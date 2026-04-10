# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, groups=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, k, s, p,
            dilation=dilation, bias=False, groups=groups
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# =====================
# SCAS: Scale-Aware Cross Attention Skip Connection
# =====================
class SCASkip(nn.Module):
    """增强版跨尺度注意力跳连"""

    def __init__(self, enc_ch, dec_ch, heads=4, attn_ratio=0.5):
        super().__init__()
        proj_ch = max(int(min(enc_ch, dec_ch) * attn_ratio), 16)
        assert proj_ch % heads == 0, \
            f"proj_ch ({proj_ch}) must be divisible by heads ({heads})"

        self.proj_ch = proj_ch
        self.heads = heads
        self.head_dim = proj_ch // heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Conv2d(dec_ch, proj_ch, 1, bias=False)
        self.k = nn.Conv2d(enc_ch, proj_ch, 1, bias=False)
        self.v = nn.Conv2d(enc_ch, proj_ch, 1, bias=False)

        # 如果后续想显式回投影可以使用，目前保留
        self.out = nn.Conv2d(proj_ch, enc_ch, 1, bias=False)

        # 多尺度下采样
        self.ds2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.ds4 = nn.AvgPool2d(kernel_size=4, stride=4)

        self.us = lambda t, size: F.interpolate(
            t, size=size, mode='bilinear', align_corners=False
        )

        # 门控生成器
        self.gate_gen = nn.Sequential(
            nn.Conv2d(proj_ch, proj_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(proj_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(proj_ch, proj_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(proj_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(proj_ch, 1, 1)
        )

        # 残差保守重标定系数
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def _multi_head_attention(self, q, k, v):
        """
        q, k, v: [B, P, H, W]
        return: [B, P, H, W]
        """
        B, P, h, w = q.shape
        N = h * w

        q = q.view(B, self.heads, self.head_dim, N)
        k = k.view(B, self.heads, self.head_dim, N)
        v = v.view(B, self.heads, self.head_dim, N)

        # attn: [B, heads, N, N]
        attn = torch.matmul(q.transpose(-1, -2) * self.scale, k)
        attn = F.softmax(attn, dim=-1)

        # out: [B, heads, head_dim, N] -> [B, P, h, w]
        out = torch.matmul(attn, v.transpose(-1, -2))
        out = out.transpose(-1, -2).contiguous().view(B, P, h, w)
        return out

    def forward(self, enc, dec):
        B, Ce, He, We = enc.shape
        _, Cd, Hd, Wd = dec.shape
        assert He == Hd and We == Wd, "SCASkip: enc/dec spatial must match"

        q1 = self.q(dec)
        k1 = self.k(enc)
        v1 = self.v(enc)

        q2 = self.ds2(q1)
        k2 = self.ds2(k1)
        v2 = self.ds2(v1)

        q4 = self.ds4(q1)
        k4 = self.ds4(k1)
        v4 = self.ds4(v1)

        attn_out_1 = self._multi_head_attention(q1, k1, v1)
        attn_out_2 = self._multi_head_attention(q2, k2, v2)
        attn_out_4 = self._multi_head_attention(q4, k4, v4)

        # 多尺度融合
        attn_out = (
            attn_out_1
            + self.us(attn_out_2, attn_out_1.shape[-2:])
            + self.us(attn_out_4, attn_out_1.shape[-2:])
        ) / 3.0

        gate_low = torch.sigmoid(self.gate_gen(attn_out))
        gate = self.us(gate_low, size=(He, We))

        # 保守残差重标定
        enc_gated = enc * gate
        return self.alpha * enc_gated + (1.0 - self.alpha) * enc


# =====================
# D2F: Dual-Domain Fusion
# =====================
class D2F(nn.Module):
    """
    1) 低频 / 高频两路通道注意力
    2) 三分支融合权重使用 softmax 归一化
       -> 保证 w_low, w_high, w_res >= 0 且和为 1
    3) 提供 get_fusion_weights() 便于训练中记录
    """

    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        reduced_ch = max(channels // reduction_ratio, 1)

        # 低频分支：GAP -> 1x1 -> ReLU -> 1x1 -> Sigmoid
        self.low_freq_processor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_ch, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_ch, channels, 1, bias=True),
            nn.Sigmoid()
        )

        # 高频分支：x - AvgPool3x3(x) -> 1x1 -> ReLU -> 1x1 -> Sigmoid
        self.high_freq_processor = nn.Sequential(
            nn.Conv2d(channels, reduced_ch, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_ch, channels, 1, bias=True),
            nn.Sigmoid()
        )

        # 三分支融合 logits: [low, high, residual]
        # softmax 后形成凸组合，避免 collapse / 负权重
        self.fusion_logits = nn.Parameter(torch.zeros(3))

    def get_fusion_weights(self):
        """
        返回当前三分支融合权重（Python float）
        """
        weights = torch.softmax(self.fusion_logits, dim=0)
        return {
            "w_low": weights[0].item(),
            "w_high": weights[1].item(),
            "w_res": weights[2].item()
        }

    def forward(self, x):
        # 低频增强
        low_freq_weights = self.low_freq_processor(x)
        low_enhanced = x * low_freq_weights

        # 高频增强：局部高通近似
        high_freq = x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high_freq_weights = self.high_freq_processor(high_freq)
        high_enhanced = x * high_freq_weights

        # 归一化融合权重
        weights = torch.softmax(self.fusion_logits, dim=0)
        w_low, w_high, w_res = weights[0], weights[1], weights[2]

        # 三路凸组合融合
        out = w_low * low_enhanced + w_high * high_enhanced + w_res * x
        return out


# =====================
# ResNet 
# =====================
class ResNetEncoder(nn.Module):
    """使用本地预训练 ResNet34 作为编码器"""

    def __init__(self, weights_path="resnet34-b627a593.pth"):
        super().__init__()
        resnet = resnet34(pretrained=False)

        if weights_path is not None:
            state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
            resnet.load_state_dict(state_dict)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.maxpool(x0)     # 1/4
        x2 = self.layer1(x1)      # 1/4
        x3 = self.layer2(x2)      # 1/8
        x4 = self.layer3(x3)      # 1/16
        x5 = self.layer4(x4)      # 1/32
        return [x2, x3, x4, x5]


# =====================
# Decoder Block
# =====================
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid = max(out_ch, in_ch // 2)
        self.conv1 = ConvBNReLU(in_ch, mid, 3, 1, 1)
        self.conv2 = ConvBNReLU(mid, out_ch, 3, 1, 1)

        self.residual = nn.Sequential()
        if in_ch != out_ch:
            self.residual = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        residual = self.residual(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual



class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(in_ch, out_ch, 3, 1, 1),
            ConvBNReLU(out_ch, out_ch, 3, 1, 1),
            ConvBNReLU(out_ch, out_ch, 3, 1, 1),
        )
        self.down = nn.MaxPool2d(2)

    def forward(self, x):
        feat = self.conv(x)
        down = self.down(feat)
        return feat, down



class EnhancedPolypSegNet(nn.Module):
    def __init__(self,
                 base_ch=64,
                 use_scas=True,
                 use_d2f=True,
                 use_resnet=True,
                 verbose=False):
        super().__init__()
        self.verbose = verbose

        if use_resnet:
            self.encoder = ResNetEncoder("resnet34-b627a593.pth")
            c1, c2, c3, c4 = 64, 128, 256, 512
        else:
            c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8
            self.enc1 = EncoderBlock(3, c1)
            self.enc2 = EncoderBlock(c1, c2)
            self.enc3 = EncoderBlock(c2, c3)
            self.enc4 = EncoderBlock(c3, c4)

        self.use_resnet = use_resnet
        self.use_d2f = use_d2f
        self.use_scas = use_scas

        self.bot_conv = ConvBNReLU(c4, c4, 3, 1, 1)
        self.d2f = D2F(c4) if use_d2f else nn.Identity()

        if use_scas:
            self.sca4 = SCASkip(c4, c4)
            self.sca3 = SCASkip(c3, c4)
            self.sca2 = SCASkip(c2, c4)
            self.sca1 = SCASkip(c1, c4)

        self.dec4 = DecoderBlock(c4 + c4, c4)
        self.dec3 = DecoderBlock(c4 + c3, c4)
        self.dec2 = DecoderBlock(c4 + c2, c4)
        self.dec1 = DecoderBlock(c4 + c1, c4)

        self.final_conv = nn.Sequential(
            ConvBNReLU(c4, c4 // 2, 3, 1, 1),
            nn.Dropout2d(0.1),
            ConvBNReLU(c4 // 2, c4 // 4, 3, 1, 1),
            nn.Conv2d(c4 // 4, 1, 1)
        )

        self.ds4 = nn.Conv2d(c4, 1, 1)
        self.ds3 = nn.Conv2d(c4, 1, 1)
        self.ds2 = nn.Conv2d(c4, 1, 1)
        self.ds1 = nn.Conv2d(c4, 1, 1)

        self.up = lambda t, size: F.interpolate(
            t, size=size, mode='bilinear', align_corners=False
        )

    def get_d2f_weights(self):
        """
        训练/验证阶段调用，用于记录论文中的 α/β/残差分支权重分布。
        若未启用 D2F，则返回 None。
        """
        if self.use_d2f and isinstance(self.d2f, D2F):
            return self.d2f.get_fusion_weights()
        return None

    def forward(self, x):
        B, _, H, W = x.shape

        # ===== Encoder =====
        if self.use_resnet:
            e1, e2, e3, e4 = self.encoder(x)

            # 对齐到统一尺度
            e1 = self.up(e1, (H // 4, W // 4))     # 1/4
            e2 = self.up(e2, (H // 8, W // 8))     # 1/8
            e3 = self.up(e3, (H // 16, W // 16))   # 1/16
            e4 = self.up(e4, (H // 16, W // 16))   # 1/16
        else:
            e1, x1 = self.enc1(x)   # H/2
            e2, x2 = self.enc2(x1)  # H/4
            e3, x3 = self.enc3(x2)  # H/8
            e4, x4 = self.enc4(x3)  # H/16

        if self.verbose:
            print("e1:", e1.shape, "e2:", e2.shape, "e3:", e3.shape, "e4:", e4.shape)

        # ===== Bottleneck =====
        b = self.bot_conv(e4)
        if self.use_d2f:
            b = self.d2f(b)

        ds4 = torch.sigmoid(self.up(self.ds4(b), (H, W)))

        # ===== Decoder stage 4 =====
        d4 = self.up(b, e4.shape[-2:])
        skip4 = self.sca4(e4, d4) if self.use_scas else e4
        d4 = torch.cat([d4, skip4], dim=1)
        d4 = self.dec4(d4)
        ds3 = torch.sigmoid(self.up(self.ds3(d4), (H, W)))

        # ===== Decoder stage 3 =====
        d3 = self.up(d4, e3.shape[-2:])
        skip3 = self.sca3(e3, d3) if self.use_scas else e3
        d3 = torch.cat([d3, skip3], dim=1)
        d3 = self.dec3(d3)
        ds2 = torch.sigmoid(self.up(self.ds2(d3), (H, W)))

        # ===== Decoder stage 2 =====
        d2 = self.up(d3, e2.shape[-2:])
        skip2 = self.sca2(e2, d2) if self.use_scas else e2
        d2 = torch.cat([d2, skip2], dim=1)
        d2 = self.dec2(d2)
        ds1 = torch.sigmoid(self.up(self.ds1(d2), (H, W)))

        # ===== Decoder stage 1 =====
        d1 = self.up(d2, e1.shape[-2:])
        skip1 = self.sca1(e1, d1) if self.use_scas else e1
        d1 = torch.cat([d1, skip1], dim=1)
        d1 = self.dec1(d1)

        # ===== Final output =====
        out = self.final_conv(d1)
        out = self.up(out, (H, W))
        out = torch.sigmoid(out)

        if self.verbose:
            print("d4:", d4.shape, "d3:", d3.shape, "d2:", d2.shape, "d1:", d1.shape, "out:", out.shape)

        return {
            'main': out,
            'ds1': ds1,
            'ds2': ds2,
            'ds3': ds3,
            'ds4': ds4
        }



def PolypSegNet_Full(base_ch=64, verbose=False):
    return EnhancedPolypSegNet(
        base_ch=base_ch,
        use_scas=True,
        use_d2f=True,
        use_resnet=True,
        verbose=verbose
    )


def PolypSegNet_Resnet_SCAS(base_ch=64, verbose=False):
    return EnhancedPolypSegNet(
        base_ch=base_ch,
        use_scas=True,
        use_d2f=False,
        use_resnet=True,
        verbose=verbose
    )


def PolypSegNet_Resnet_D2F(base_ch=64, verbose=False):
    return EnhancedPolypSegNet(
        base_ch=base_ch,
        use_scas=False,
        use_d2f=True,
        use_resnet=True,
        verbose=verbose
    )


def PolypSegNet_Resnet(base_ch=64, verbose=False):
    return EnhancedPolypSegNet(
        base_ch=base_ch,
        use_scas=False,
        use_d2f=True,
        use_resnet=False,
        verbose=verbose
    )


def PolypSegNet_D2F(base_ch=64, verbose=False):
    return EnhancedPolypSegNet(
        base_ch=base_ch,
        use_scas=False,
        use_d2f=True,
        use_resnet=False,
        verbose=verbose
    )


def PolypSegNet_Baseline(base_ch=64, verbose=False):
    return EnhancedPolypSegNet(
        base_ch=base_ch,
        use_scas=False,
        use_d2f=False,
        use_resnet=False,
        verbose=verbose
    )



if __name__ == "__main__":
    x = torch.randn(2, 3, 352, 352)

    for name, builder in [
        ("Full", PolypSegNet_Full),
        # ("Resnet+SCAS", PolypSegNet_Resnet_SCAS),
        # ("Resnet+D2F", PolypSegNet_Resnet_D2F),
        # ("Resnet", PolypSegNet_Resnet),
        # ("D2F", PolypSegNet_D2F),
        # ("Baseline", PolypSegNet_Baseline),
    ]:
        model = builder(verbose=False)
        y = model(x)

        print(f"{name:>12s} | input: {tuple(x.shape)} -> output: {tuple(y['main'].shape)}")

        weights = model.get_d2f_weights()
        if weights is not None:
            print(f"{name:>12s} | D2F weights = {weights}")

        assert y['main'].shape == (2, 1, 352, 352), f"{name} output shape mismatch!"

    print("所有模型测试通过!")
