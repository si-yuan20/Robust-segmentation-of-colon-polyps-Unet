import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
from sklearn.model_selection import train_test_split
import os
import time
import cv2
from train import create_model
from ImageLoader2D import load_data
from utils import calculate_metrics
import config
from main import  split_data


def save_predictions(images, masks, predictions, output_dir, batch_idx):
    """保存原始图像、真实标签和预测结果"""
    batch_size = images.shape[0]

    for i in range(batch_size):
        # 创建唯一的文件名
        idx = batch_idx * config.BATCH_SIZE + i
        filename = f"{idx:04d}"

        # 保存原始图像
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, 'images', f"{filename}_image.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # 保存真实标签
        mask = masks[i].permute(1, 2, 0).cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, 'masks', f"{filename}_mask.png"), mask)

        # 保存预测结果
        pred = predictions[i].permute(1, 2, 0).cpu().numpy()
        pred = (pred > 0.5).astype(np.uint8) * 255  # 二值化
        cv2.imwrite(os.path.join(output_dir, 'predictions', f"{filename}_pred.png"), pred)


def validate_model(X_val, y_val, dataset_name, model_name, model_path, save_results=True, save_predictions_flag=True):
    """验证模型并计算多个指标，可选择保存预测结果"""
    print(f"Loading {dataset_name} dataset...")

    # 转换为PyTorch张量
    X_val_tensor = torch.from_numpy(X_val).permute(0, 3, 1, 2).float()
    Y_val_tensor = torch.from_numpy(y_val).permute(0, 3, 1, 2).float()

    # 创建数据集和数据加载器
    dataset = TensorDataset(X_val_tensor, Y_val_tensor)
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        drop_last=True  # 确保不会出现batch size为1的情况
    )

    # 创建模型并加载权重
    model = create_model(model_name)
    model.to(config.DEVICE)

    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 初始化指标收集
    all_metrics = {
        'iou': [],
        'f1': [],
        'precision': [],
        'recall': [],
        'accuracy': []
    }

    # 创建保存预测结果的目录
    if save_predictions_flag:
        predictions_dir = os.path.join("validation_results", f"{model_name}_{dataset_name}_predictions")
        os.makedirs(os.path.join(predictions_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(predictions_dir, 'masks'), exist_ok=True)
        os.makedirs(os.path.join(predictions_dir, 'predictions'), exist_ok=True)
        print(f"Predictions will be saved to: {predictions_dir}")

    # 预热GPU（如果有）
    if config.DEVICE == 'cuda':
        dummy_input = torch.randn(1, 3, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]).to(config.DEVICE)
        for _ in range(10):
            _ = model(dummy_input)

    # 计算FPS
    fps_values = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)

            # 前向传播计时
            start_time = time.time()
            output = model(data)

            # 如果是CUDA设备，等待所有内核完成
            if config.DEVICE == 'cuda':
                torch.cuda.synchronize()

            end_time = time.time()

            # 计算FPS
            batch_time = end_time - start_time
            batch_fps = data.size(0) / batch_time
            fps_values.append(batch_fps)

            # 计算指标
            metrics = calculate_metrics(target, output['main'])

            for key in all_metrics.keys():
                if key in metrics:
                    all_metrics[key].append(metrics[key])

            # 保存预测结果
            if save_predictions_flag:
                save_predictions(data, target, output['main'], predictions_dir, batch_idx)

    # 计算平均指标
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    avg_metrics['fps'] = np.mean(fps_values)

    # 打印结果
    print(f"Validation Results for {model_name} on {dataset_name}:")
    for key, value in avg_metrics.items():
        print(f"{key}: {value:.4f}")

    # 保存结果到txt文件
    if save_results:
        results_dir = "validation_results"
        os.makedirs(results_dir, exist_ok=True)
        result_file = os.path.join(results_dir, f"{model_name}_{dataset_name}_results.txt")

        with open(result_file, 'w') as f:
            f.write(f"Validation Results for {model_name} on {dataset_name}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Device: {config.DEVICE}\n")
            f.write(f"Batch size: {config.BATCH_SIZE}\n")
            f.write(f"Number of validation samples: {len(dataset)}\n\n")

            for key, value in avg_metrics.items():
                if key == 'fps':
                    f.write(f"{key}: {value:.2f}\n")
                else:
                    f.write(f"{key}: {value:.4f}\n")

        print(f"Results saved to {result_file}")

    return avg_metrics


if __name__ == "__main__":
    # 首先需要加载数据
    X, Y = load_data(
        config.IMG_HEIGHT,
        config.IMG_WIDTH,
        -1,  # 加载所有图像
        "ETIS-LaribPolypDB"
    )
    # 分割训练集、测试集、验证集，并确保随机种子和训练过程的一致，防止数据混乱

    # X_train, X_test, X_val, y_train, y_test, y_val = split_data(
    #     X, Y,
    #     train_size=0.8,
    #     test_size=0.1,
    #     val_size=0.1,
    #     random_state=42
    # )


    model_path = os.path.join(config.CHECKPOINT_DIR, f'best_model_cvc-clinicdb_{config.MODEL_NAME}.pth')
    validate_model(X, Y, 'ETIS-LaribPolypDB', config.MODEL_NAME, model_path, save_predictions_flag=True)