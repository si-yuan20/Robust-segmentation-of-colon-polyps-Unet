# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import resnet34


# ---------------------
# 基础积木 - 增强版
# ---------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, groups=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, dilation=dilation, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# ---------------------
# 创新二：增强版SCAS 尺度感知跨注意力跳连
# ---------------------
class SCASkip(nn.Module):
    """增强版跨尺度注意力"""

    def __init__(self, enc_ch, dec_ch, heads=4, attn_ratio=0.5):
        super().__init__()
        proj_ch = max(int(min(enc_ch, dec_ch) * attn_ratio), 16)
        self.q = nn.Conv2d(dec_ch, proj_ch, 1, bias=False)
        self.k = nn.Conv2d(enc_ch, proj_ch, 1, bias=False)
        self.v = nn.Conv2d(enc_ch, proj_ch, 1, bias=False)
        self.out = nn.Conv2d(proj_ch, enc_ch, 1, bias=False)
        self.heads = heads
        self.scale = (proj_ch // heads) ** -0.5

        # 多尺度下采样
        self.ds2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.ds4 = nn.AvgPool2d(kernel_size=4, stride=4)
        # self.ds8 = nn.AvgPool2d(kernel_size=8, stride=8)

        self.us = lambda t, size: F.interpolate(t, size=size, mode='bilinear', align_corners=False)

        # 增强门控生成器
        self.gate_gen = nn.Sequential(
            nn.Conv2d(proj_ch, proj_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(proj_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(proj_ch, proj_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(proj_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(proj_ch, 1, 1)
        )

        # 残差连接权重
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, enc, dec):
        B, Ce, He, We = enc.shape
        _, Cd, Hd, Wd = dec.shape
        assert He == Hd and We == Wd, "SCASkip: enc/dec spatial must match"

        # 多尺度注意力
        q1 = self.q(dec)
        k1 = self.k(enc)
        v1 = self.v(enc)

        # 不同尺度的注意力
        q2 = self.ds2(q1)
        k2 = self.ds2(k1)
        v2 = self.ds2(v1)

        q4 = self.ds4(q1)
        k4 = self.ds4(k1)
        v4 = self.ds4(v1)

        # 计算多尺度注意力
        attn_outputs = []
        for q, k, v in [(q1, k1, v1), (q2, k2, v2), (q4, k4, v4)]:
            B_, P, h, w = q.shape

            # 重塑为多头注意力格式 [B, heads, C, N]
            q = q.view(B_, self.heads, P // self.heads, h * w)
            k = k.view(B_, self.heads, P // self.heads, h * w)
            v = v.view(B_, self.heads, P // self.heads, h * w)

            # 注意力计算
            attn = torch.matmul(q.transpose(-1, -2) * self.scale, k)
            attn = F.softmax(attn, dim=-1)

            # 应用注意力权重到V
            out = torch.matmul(attn, v.transpose(-1, -2))
            out = out.transpose(-1, -2).contiguous().view(B_, P, h, w)
            attn_outputs.append(out)

        # 融合多尺度注意力
        attn_out = attn_outputs[0]
        if len(attn_outputs) > 1:
            attn_out += self.us(attn_outputs[1], attn_out.shape[-2:])
            attn_out += self.us(attn_outputs[2], attn_out.shape[-2:])
            attn_out /= len(attn_outputs)

        # 生成门控图
        gate_low = torch.sigmoid(self.gate_gen(attn_out))
        gate = self.us(gate_low, size=(He, We))

        # 对encoder skip做门控 (保留少量原始信息)
        enc_gated = enc * gate
        return self.alpha * enc_gated + (1 - self.alpha) * enc


# ---------------------
# 创新三：增强版D2F 双域融合模块
# ---------------------
class D2F(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        # 低频处理
        self.low_freq_processor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1),
            nn.Sigmoid()
        )

        # 高频处理
        self.high_freq_processor = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1),
            nn.Sigmoid()
        )

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        # 低频分量
        low_freq_weights = self.low_freq_processor(x)
        low_enhanced = x * low_freq_weights

        # 高频分量 (通过残差获取)
        high_freq = x - F.avg_pool2d(x, 3, stride=1, padding=1)
        high_freq_weights = self.high_freq_processor(high_freq)
        high_enhanced = x * high_freq_weights

        # 双域融合
        return self.alpha * low_enhanced + self.beta * high_enhanced + (1 - self.alpha - self.beta) * x


# ---------------------
# 增强版编解码骨架
# ---------------------
import torch
import torch.nn as nn
from torchvision.models import resnet34


class ResNetEncoder(nn.Module):
    """使用本地预训练ResNet作为编码器"""

    def __init__(self, weights_path="resnet34-b627a593.pth"):
        super().__init__()
        # 初始化未加载预训练权重的ResNet
        resnet = resnet34(pretrained=False)

        # 如果提供了权重路径，则加载本地权重
        if weights_path is not None:
            # 加载本地权重文件
            state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
            # 加载权重到模型
            resnet.load_state_dict(state_dict)

        # 提取ResNet的各个层
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        # 初始卷积
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.maxpool(x0)  # 1/4

        # 残差层
        x2 = self.layer1(x1)  # 1/4
        x3 = self.layer2(x2)  # 1/8
        x4 = self.layer3(x3)  # 1/16
        x5 = self.layer4(x4)  # 1/32

        return [x2, x3, x4, x5]


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid = max(out_ch, in_ch // 2)
        self.conv1 = ConvBNReLU(in_ch, mid, 3, 1, 1)
        self.conv2 = ConvBNReLU(mid, out_ch, 3, 1, 1)

        # 残差连接
        self.residual = nn.Sequential()
        if in_ch != out_ch:
            print("使用残差连接！！！！！！！！！！！！！！")
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
            ConvBNReLU(out_ch, out_ch, 3, 1, 1),  # 增加一层卷积
        )
        self.down = nn.MaxPool2d(2)
    def forward(self, x):
        feat = self.conv(x)
        down = self.down(feat)
        return feat, down

# ---------------------
# 主模型：增强版PolypSegNet
# ---------------------
class EnhancedPolypSegNet(nn.Module):
    def __init__(self,
                 base_ch=64,
                 use_scas=True,
                 use_d2f=True,
                 use_resnet=True):
        super().__init__()

        if use_resnet:
            # 使用ResNet作为编码器
            self.encoder = ResNetEncoder("resnet34-b627a593.pth")
            c1, c2, c3, c4 = 64, 128, 256, 512  # ResNet34各层输出通道数
        else:
            # 原始编码器
            c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8
            self.enc1 = EncoderBlock(3, c1)  # 1/2
            self.enc2 = EncoderBlock(c1, c2)  # 1/4
            self.enc3 = EncoderBlock(c2, c3)  # 1/8
            self.enc4 = EncoderBlock(c3, c4)  # 1/16

        self.use_resnet = use_resnet

        # Bottleneck (包含D2F模块)
        self.use_d2f = use_d2f
        self.bot_conv = ConvBNReLU(c4, c4, 3, 1, 1)
        self.d2f = D2F(c4) if use_d2f else nn.Identity()

        # SCAS gates
        self.use_scas = use_scas
        if use_scas:
            self.sca4 = SCASkip(c4, c4)
            self.sca3 = SCASkip(c3, c4)  # 修改：decoder输出通道是c4
            self.sca2 = SCASkip(c2, c4)  # 修改：decoder输出通道是c4
            self.sca1 = SCASkip(c1, c4)  # 修改：decoder输出通道是c4

        # 修改解码器通道数，确保一致性
        self.dec4 = DecoderBlock(c4 + c4, c4)
        self.dec3 = DecoderBlock(c4 + c3, c4)
        self.dec2 = DecoderBlock(c4 + c2, c4)
        self.dec1 = DecoderBlock(c4 + c1, c4)

        # 最终输出卷积
        self.final_conv = nn.Sequential(
            ConvBNReLU(c4, c4 // 2, 3, 1, 1),
            nn.Dropout2d(0.1),
            ConvBNReLU(c4 // 2, c4 // 4, 3, 1, 1),
            nn.Conv2d(c4 // 4, 1, 1)
        )

        # 深度监督输出 - 修改为与解码器输出通道一致
        self.ds4 = nn.Conv2d(c4, 1, 1)
        self.ds3 = nn.Conv2d(c4, 1, 1)
        self.ds2 = nn.Conv2d(c4, 1, 1)
        self.ds1 = nn.Conv2d(c4, 1, 1)

        # 上采样函数
        self.up = lambda t, size: F.interpolate(t, size=size, mode='bilinear', align_corners=False)

    def forward(self, x):
        B, _, H, W = x.shape

        # Encoder
        if self.use_resnet:
            e1, e2, e3, e4 = self.encoder(x)
            # 修复：确保所有编码器特征的空间尺寸正确
            e1 = self.up(e1, (H // 4, W // 4))  # 1/4
            e2 = self.up(e2, (H // 8, W // 8))  # 1/8
            e3 = self.up(e3, (H // 16, W // 16))  # 1/16
            e4 = self.up(e4, (H // 16, W // 16))  # 1/16
            print("e1.shape:", e1.shape)
            print("e2.shape:" , e2.shape)
            print("e3.shape:" , e3.shape)
            print("e4.shape:" , e4.shape)

        else:
            e1, x1 = self.enc1(x)  # e1: H/2
            e2, x2 = self.enc2(x1)  # e2: H/4
            e3, x3 = self.enc3(x2)  # e3: H/8
            e4, x4 = self.enc4(x3)  # e4: H/16

        # Bottleneck (包含D2F处理)
        b = self.bot_conv(e4)  # H/16
        if self.use_d2f:
            b = self.d2f(b)

        # 深度监督输出
        ds4 = torch.sigmoid(self.up(self.ds4(b), (H, W)))

        # Decoder stage 4: 确保尺寸匹配
        d4 = self.up(b, e4.shape[-2:])  # 确保与e4尺寸相同
        skip4 = e4
        if self.use_scas:
            skip4 = self.sca4(e4, d4)
        d4 = torch.cat([d4, skip4], dim=1)
        d4 = self.dec4(d4)
        ds3 = torch.sigmoid(self.up(self.ds3(d4), (H, W)))
        print("d4.shape:" , d4.shape)
        # Decoder stage 3: 确保尺寸匹配
        d3 = self.up(d4, e3.shape[-2:])  # 确保与e3尺寸相同
        skip3 = e3
        if self.use_scas:
            skip3 = self.sca3(e3, d3)
        d3 = torch.cat([d3, skip3], dim=1)
        d3 = self.dec3(d3)
        ds2 = torch.sigmoid(self.up(self.ds2(d3), (H, W)))
        print("d3.shape:" , d3.shape)

        # Decoder stage 2: 确保尺寸匹配
        d2 = self.up(d3, e2.shape[-2:])  # 确保与e2尺寸相同
        skip2 = e2
        if self.use_scas:
            skip2 = self.sca2(e2, d2)
        d2 = torch.cat([d2, skip2], dim=1)
        d2 = self.dec2(d2)
        ds1 = torch.sigmoid(self.up(self.ds1(d2), (H, W)))
        print("d2.shape:" , d2.shape)

        # Decoder stage 1: 确保尺寸匹配
        d1 = self.up(d2, e1.shape[-2:])  # 确保与e1尺寸相同
        skip1 = e1
        if self.use_scas:
            skip1 = self.sca1(e1, d1)
        d1 = torch.cat([d1, skip1], dim=1)
        d1 = self.dec1(d1)
        print("d1.shape:" , d1.shape)

        # 最终输出
        out = self.final_conv(d1)
        out = self.up(out, (H, W))
        out = torch.sigmoid(out)
        print("out.shape:" , out.shape)

        return {
            'main': out,
            'ds1': ds1,
            'ds2': ds2,
            'ds3': ds3,
            'ds4': ds4
        }

# ---------------------
# 各种模型构造器（含消融）
# ---------------------
def PolypSegNet_Full(base_ch=64):
    """完整模型：包含所有创新点"""
    return EnhancedPolypSegNet(
        base_ch=base_ch,
        use_scas=True,
        use_d2f=True,
        use_resnet=True
    )


def PolypSegNet_Resnet_SCAS(base_ch=64):
    """含Resnet+SCAS创新点"""
    return EnhancedPolypSegNet(
        base_ch=base_ch,
        use_scas=True,
        use_d2f=False,
        use_resnet=True
    )


def PolypSegNet_Resnet_D2F(base_ch=64):
    """含Resnet+D2F创新点"""
    return EnhancedPolypSegNet(
        base_ch=base_ch,
        use_scas=False,
        use_d2f=True,
        use_resnet=True
    )

def PolypSegNet_Resnet(base_ch=64):
    """仅含Resnet创新点"""
    return EnhancedPolypSegNet(
        base_ch=base_ch,
        use_scas=False,
        use_d2f=True,
        use_resnet=False
    )

def PolypSegNet_D2F(base_ch=64):
    """仅含D2F创新点"""
    return EnhancedPolypSegNet(
        base_ch=base_ch,
        use_scas=False,
        use_d2f=True,
        use_resnet=False
    )


def PolypSegNet_Baseline(base_ch=64):
    """基线模型：不含任何创新点"""
    return EnhancedPolypSegNet(
        base_ch=base_ch,
        use_scas=False,
        use_d2f=False,
        use_resnet=False
    )


# ---------------------
# 快速自测
# ---------------------
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
        model = builder()
        y = model(x)
        print(f"{name:>12s} | input: {tuple(x.shape)} -> output: {y['main'].shape}")
        assert y['main'].shape == (2, 1, 352, 352), f"{name} output shape mismatch!"

    print("所有模型测试通过!")