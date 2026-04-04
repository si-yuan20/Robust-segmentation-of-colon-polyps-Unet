import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from model import PolypSegNet_Full
from ImageLoader2D import load_data
import config
import cv2


def visualize_feature_maps(seed):
    """可视化FULL模型的每个特征层的热力图"""
    # 设置随机种子以确保可重复性
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 定义数据集列表
    datasets = ['cvc-clinicdb', 'cvc-colondb', 'ETIS-LaribPolypDB', 'cvc-t']

    # 创建模型
    model = PolypSegNet_Full()
    model.to(config.DEVICE)

    # 加载模型权重
    model_path = os.path.join(config.CHECKPOINT_DIR, f'best_model_cvc-clinicdb_PolypSegNet_Full.pth')
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model weights from {model_path}")
    else:
        print(f"Model {model_path} not found. Using random weights.")

    model.eval()

    # 创建图形 - 增加左侧边距以容纳数据集名称
    fig, axes = plt.subplots(4, 10, figsize=(25, 12))
    fig.suptitle('Feature Maps Visualization for PolypSegNet_Full', fontsize=16, fontweight='bold')

    # 增加左侧边距以容纳数据集名称
    plt.subplots_adjust(left=0.08)

    # 设置列标题
    col_titles = ['Original', 'Ground Truth', 'Encoder Layer 1', 'Encoder Layer 2',
                  'Encoder Layer 3', 'Encoder Layer 4', 'Decoder Layer 4',
                  'Decoder Layer 3', 'Decoder Layer 2', 'Decoder Layer 1']

    for ax, col_title in zip(axes[0], col_titles):
        ax.set_title(col_title, fontsize=12, fontweight='bold')

    # 为每个数据集处理一张图像
    for dataset_idx, dataset_name in enumerate(datasets):
        print(f"Processing {dataset_name} dataset...")

        # 加载数据
        X, Y = load_data(
            config.IMG_HEIGHT,
            config.IMG_WIDTH,
            -1,  # 加载所有图像
            dataset_name
        )

        # 随机选择一张图像
        random_idx = random.randint(0, len(X) - 1)
        sample_image = X[random_idx]
        sample_mask = Y[random_idx]

        # 在左侧添加数据集名称（垂直方向）
        fig.text(0.07, 0.85 - dataset_idx * 0.25, dataset_name,
                 rotation=90, fontsize=10, fontweight='bold',
                 verticalalignment='center', horizontalalignment='center')

        # 显示原始图像
        axes[dataset_idx, 0].imshow(sample_image)
        axes[dataset_idx, 0].axis('off')

        # 显示真实标签
        axes[dataset_idx, 1].imshow(sample_mask[:, :, 0], cmap='gray')
        axes[dataset_idx, 1].axis('off')

        # 准备输入数据
        input_tensor = torch.from_numpy(sample_image).permute(2, 0, 1).unsqueeze(0).float().to(config.DEVICE)

        # 创建钩子来获取中间特征
        encoder_features = []
        decoder_features = []

        def get_encoder_hook(layer_idx):
            def hook(module, input, output):
                encoder_features.append(output.detach().cpu().numpy())

            return hook

        def get_decoder_hook(layer_idx):
            def hook(module, input, output):
                decoder_features.append(output.detach().cpu().numpy())

            return hook

        # 注册钩子
        hooks = []
        if hasattr(model, 'encoder') and model.use_resnet:
            # ResNet编码器
            hooks.append(model.encoder.layer1.register_forward_hook(get_encoder_hook(0)))
            hooks.append(model.encoder.layer2.register_forward_hook(get_encoder_hook(1)))
            hooks.append(model.encoder.layer3.register_forward_hook(get_encoder_hook(2)))
            hooks.append(model.encoder.layer4.register_forward_hook(get_encoder_hook(3)))
        else:
            # 原始编码器
            hooks.append(model.enc1.register_forward_hook(get_encoder_hook(0)))
            hooks.append(model.enc2.register_forward_hook(get_encoder_hook(1)))
            hooks.append(model.enc3.register_forward_hook(get_encoder_hook(2)))
            hooks.append(model.enc4.register_forward_hook(get_encoder_hook(3)))

        # 解码器钩子
        hooks.append(model.dec4.register_forward_hook(get_decoder_hook(0)))
        hooks.append(model.dec3.register_forward_hook(get_decoder_hook(1)))
        hooks.append(model.dec2.register_forward_hook(get_decoder_hook(2)))
        hooks.append(model.dec1.register_forward_hook(get_decoder_hook(3)))

        # 预测
        with torch.no_grad():
            output = model(input_tensor)

        # 移除钩子
        for hook in hooks:
            hook.remove()

        # 可视化编码器特征
        for i, feat in enumerate(encoder_features):
            # 计算特征图的平均激活
            avg_activation = np.mean(feat[0], axis=0)

            # 上采样到原始图像大小
            resized_activation = cv2.resize(avg_activation, (config.IMG_WIDTH, config.IMG_HEIGHT))

            # 显示热力图
            axes[dataset_idx, i + 2].imshow(resized_activation, cmap='hot')
            axes[dataset_idx, i + 2].axis('off')

        # 可视化解码器特征
        for i, feat in enumerate(decoder_features):
            # 计算特征图的平均激活
            avg_activation = np.mean(feat[0], axis=0)

            # 上采样到原始图像大小
            resized_activation = cv2.resize(avg_activation, (config.IMG_WIDTH, config.IMG_HEIGHT))

            # 显示热力图
            axes[dataset_idx, i + 6].imshow(resized_activation, cmap='hot')
            axes[dataset_idx, i + 6].axis('off')

    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, left=0.08)  # 调整顶部和左侧边距

    # 保存图像
    output_dir = "feature_visualization"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"feature_maps_visualization_{seed}.png"), dpi=300, bbox_inches='tight')

    print(f"Feature maps visualization saved to {output_dir}")
    plt.show()


if __name__ == "__main__":
    visualize_feature_maps(2002)