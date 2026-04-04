import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from train import create_model
from ImageLoader2D import load_data
import config
from main import split_data


def visualize_comparison(seed):
    """可视化三个数据集在六个模型上的分割结果对比"""
    # 设置随机种子以确保可重复性
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 定义数据集列表
    datasets = ['cvc-clinicdb', 'cvc-colondb', 'ETIS-LaribPolypDB', 'cvc-t']

    # 定义模型列表
    model_names = [
        'PolypSegNet_Baseline',
        'PolypSegNet_D2F',
        'PolypSegNet_Resnet',
        'PolypSegNet_Resnet_D2F',
        'PolypSegNet_Resnet_SCAS',
        'PolypSegNet_Full',
    ]

    # 创建图形 - 增加左侧边距以容纳数据集名称
    fig, axes = plt.subplots(8, 8, figsize=(24, 18))  # 高度增加为18以适应6行
    # fig.suptitle('Segmentation Results Comparison Across Datasets and Models', fontsize=15, fontweight='bold')

    # 增加左侧边距以容纳数据集名称
    plt.subplots_adjust(left=0.08)

    # 设置列标题
    col_titles = ['Original Image', 'Ground Truth'] + [name.replace('PolypSegNet_', '').replace("_", "+") for name in model_names]
    col_titles[-1] = "Ours"
    col_titles[3] = "+D2F"
    col_titles[4] = "+Resnet"
    col_titles[5] = "+Resnet+D2F"
    col_titles[6] = "+Resnet+SCAS"

    print(col_titles)
    for ax, col_title in zip(axes[0], col_titles):
        ax.set_title(col_title, fontsize=15, fontweight='bold')

    # 加载每个数据集并随机选择两张图像
    for dataset_idx, dataset_name in enumerate(datasets):
        print(f"Processing {dataset_name} dataset...")

        # 加载数据
        X, Y = load_data(
            config.IMG_HEIGHT,
            config.IMG_WIDTH,
            -1,  # 加载所有图像
            dataset_name
        )

        # 随机选择两张不同的图像
        random_indices = random.sample(range(len(X)), 2)

        for sample_idx, random_idx in enumerate(random_indices):
            sample_image = X[random_idx]
            sample_mask = Y[random_idx]

            # 计算当前行索引
            row_idx = dataset_idx * 2 + sample_idx

            # 在左侧添加数据集名称（垂直方向）- 只在每个数据集的第一行添加
            if sample_idx == 0:
                fig.text(0.07, 0.80 - dataset_idx * 0.215, dataset_name,
                         rotation=90, fontsize=14, fontweight='bold',
                         verticalalignment='center', horizontalalignment='center')

            # 显示原始图像
            axes[row_idx, 0].imshow(sample_image)
            axes[row_idx, 0].axis('off')

            # 显示真实标签
            axes[row_idx, 1].imshow(sample_mask[:, :, 0], cmap='gray')
            axes[row_idx, 1].axis('off')

            # 为每个模型预测并显示结果
            for col_idx, model_name in enumerate(model_names, start=2):
                # 加载模型
                model_path = os.path.join(config.CHECKPOINT_DIR, f'best_model_cvc-clinicdb_{model_name}.pth')

                if not os.path.exists(model_path):
                    print(f"Model {model_path} not found. Skipping...")
                    axes[row_idx, col_idx].text(0.5, 0.5, 'Model\nNot Found',
                                                ha='center', va='center', transform=axes[row_idx, col_idx].transAxes)
                    axes[row_idx, col_idx].axis('off')
                    continue

                model = create_model(model_name)
                model.to(config.DEVICE)

                # 加载模型权重
                checkpoint = torch.load(model_path, map_location=config.DEVICE)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()

                # 准备输入数据 - 完全按照test.py中的方式处理
                # 将图像从(H, W, C)转换为(C, H, W)并添加批次维度，然后转换为float
                input_tensor = torch.from_numpy(sample_image).permute(2, 0, 1).unsqueeze(0).float().to(config.DEVICE)

                # 预测 - 完全按照test.py中的方式处理
                with torch.no_grad():
                    output = model(input_tensor)

                    # 关键修改：直接使用output['main']，不应用sigmoid
                    # 因为test.py中的save_predictions函数也是直接使用output['main']
                    prediction = output['main']

                    # 移除批次维度并转换为numpy数组
                    prediction = prediction.squeeze(0).permute(1, 2, 0).cpu().numpy()

                    # 二值化 - 按照test.py中的方式
                    binary_pred = (prediction > 0.5).astype(np.uint8) * 255

                # 显示预测结果
                axes[row_idx, col_idx].imshow(binary_pred[:, :, 0], cmap='gray')
                axes[row_idx, col_idx].axis('off')

                # 打印预测结果的统计信息以便调试
                print(f"Model {model_name} on {dataset_name} sample {sample_idx + 1}: "
                      f"Prediction range: [{prediction.min():.4f}, {prediction.max():.4f}], "
                      f"Binary pixels: {np.sum(binary_pred > 0)}/{binary_pred.size}")

    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, left=0.08)  # 调整顶部和左侧边距

    # 保存图像
    output_dir = "comparison_results"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"dataset_model_comparison_{seed}.png"), dpi=300, bbox_inches='tight')

    print(f"Comparison figure saved to {output_dir}")
    plt.show()


if __name__ == "__main__":
    visualize_comparison(2002)