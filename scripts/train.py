# train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import time
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import cv2
from torch.cuda import amp
import json
from datetime import datetime
warnings.filterwarnings('ignore')

from model import PolypSegNet_Full, PolypSegNet_Resnet_SCAS, PolypSegNet_Resnet_D2F, PolypSegNet_Resnet, PolypSegNet_D2F, PolypSegNet_Baseline
from ImageLoader2D import load_data
from utils import MultiScaleLoss, calculate_metrics, calculate_mae, calculate_accuracy
import config

def create_model(model_name):
    """创建模型"""
    model_dict = {
        'PolypSegNet_Full': PolypSegNet_Full(),
        'PolypSegNet_Resnet_SCAS': PolypSegNet_Resnet_SCAS(),
        'PolypSegNet_Resnet_D2F': PolypSegNet_Resnet_D2F(),
        'PolypSegNet_Resnet': PolypSegNet_Resnet(),
        'PolypSegNet_D2F': PolypSegNet_D2F(),
        'PolypSegNet_Baseline': PolypSegNet_Baseline()
    }
    
    if model_name not in model_dict:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model_dict[model_name]

def evaluate_model(model, dataloader, criterion):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    all_metrics = {
        'iou': [], 'f1': [], 'precision': [], 'recall': [], 'accuracy': [], 'mae': []
    }
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)
            
            # 前向传播
            output = model(data)
            
            # 计算损失
            loss = criterion(output, target)
            total_loss += loss.item()
            
            # 计算指标
            metrics = calculate_metrics(target, output['main'])
            metrics['mae'] = calculate_mae(target, output['main'])
            metrics['accuracy'] = calculate_accuracy(target, output['main'])
            
            for key in all_metrics.keys():
                all_metrics[key].append(metrics[key])
    
    # 计算平均指标
    avg_loss = total_loss / len(dataloader)
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    
    return avg_loss, avg_metrics

def save_visualizations(model, dataloader, epoch, dataset_name, model_name, num_images=4):
    """保存可视化结果"""
    model.eval()
    vis_dir = os.path.join(config.RESULTS_DIR, "visualizations", f"{dataset_name}_{model_name}")
    os.makedirs(vis_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, (data, target) in enumerate(dataloader):
            if i >= num_images:  # 只处理前num_images个样本
                break
                
            data, target = data.to(config.DEVICE), target.to(config.DEVICE)
            
            # 前向传播
            output = model(data)
            
            # 转换为numpy数组
            image = data[0].cpu().permute(1, 2, 0).numpy()
            true_mask = target[0, 0].cpu().numpy()
            pred_mask = output['main'][0, 0].cpu().numpy()
            
            # 创建热力图
            plt.figure(figsize=(15, 5))
            
            # 原始图像
            plt.subplot(1, 3, 1)
            plt.imshow(image)
            plt.title('Original Image')
            plt.axis('off')
            
            # 真实分割
            plt.subplot(1, 3, 2)
            plt.imshow(true_mask, cmap='gray')
            plt.title('Ground Truth')
            plt.axis('off')
            
            # 预测分割
            plt.subplot(1, 3, 3)
            plt.imshow(pred_mask, cmap='jet')
            plt.title('Prediction')
            plt.axis('off')
            
            # 保存图像
            plt.savefig(os.path.join(
                vis_dir, 
                f"epoch{epoch}_sample{i}_iou{calculate_metrics(target, output['main'])['iou']:.4f}.png"
            ), bbox_inches='tight', dpi=300)
            plt.close()
            
            # 保存不同层的热力图
            for layer_name in ['ds1', 'ds2', 'ds3', 'ds4']:
                layer_output = output[layer_name][0, 0].cpu().numpy()
                
                plt.figure(figsize=(5, 5))
                plt.imshow(layer_output, cmap='jet')
                plt.title(f'{layer_name} Output')
                plt.axis('off')
                plt.colorbar()
                
                plt.savefig(os.path.join(
                    vis_dir, 
                    f"epoch{epoch}_sample{i}_{layer_name}.png"
                ), bbox_inches='tight', dpi=300)
                plt.close()

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    """学习率预热"""
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
        
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

def train_model(X_train, X_test, y_train, y_test, dataset_name, model_name, num_epochs, patience):
    """训练模型"""
    # 创建目录
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # 创建唯一的实验ID
    experiment_id = f"{dataset_name}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 创建日志文件
    log_file = os.path.join(config.LOG_DIR, f'training_log_{experiment_id}.txt')
    metrics_file = os.path.join(config.LOG_DIR, f'metrics_{experiment_id}.json')
    
    # 初始化混合精度训练
    scaler = amp.GradScaler(enabled=config.USE_AMP)
    
    # 记录训练配置
    config_dict = {
        'dataset': dataset_name,
        'model': model_name,
        'epochs': num_epochs,
        'batch_size': config.BATCH_SIZE,
        'learning_rate': config.LEARNING_RATE,
        'patience': patience,
        'use_amp': config.USE_AMP,
        'gradient_accumulation_steps': config.GRADIENT_ACCUMULATION_STEPS,
        'experiment_id': experiment_id,
        'start_time': datetime.now().isoformat()
    }
    
    with open(log_file, 'w') as f:
        f.write(f"Training Log for {experiment_id}\n")
        f.write("="*50 + "\n")
        f.write("Configuration:\n")
        for key, value in config_dict.items():
            f.write(f"{key}: {value}\n")
        f.write("="*50 + "\n")
    
    # 保存配置
    with open(os.path.join(config.LOG_DIR, f'config_{experiment_id}.json'), 'w') as f:
        json.dump(config_dict, f, indent=4)
    
    # 加载训练数据
    print(f"Loading {dataset_name} dataset...")
    X_train, Y_train = X_train, y_train
    
    # 转换为PyTorch张量
    X_train = torch.from_numpy(X_train).permute(0, 3, 1, 2).float()
    Y_train = torch.from_numpy(Y_train).permute(0, 3, 1, 2).float()
    
    # 创建数据集和数据加载器
    dataset = TensorDataset(X_train, Y_train)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=config.NUM_WORKERS,
        pin_memory=True,  # 加速数据加载
        drop_last=True
    )
    
    # 加载测试数据
    print(f"Loading test data...")
    X_test, Y_test = X_test, y_test
    
    X_test = torch.from_numpy(X_test).permute(0, 3, 1, 2).float()
    Y_test = torch.from_numpy(Y_test).permute(0, 3, 1, 2).float()
    
    test_dataset = TensorDataset(X_test, Y_test)
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=config.NUM_WORKERS,
        pin_memory=True,  # 加速数据加载
        drop_last=True
    )
    
    # 创建模型
    model = create_model(model_name)
    model.to(config.DEVICE)
    
    # 定义损失函数和优化器
    criterion = MultiScaleLoss(nn.BCEWithLogitsLoss())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=config.MIN_LR
    )
    
    # 学习率预热
    warmup_scheduler = warmup_lr_scheduler(optimizer, config.WARMUP_EPOCHS, config.WARMUP_FACTOR)
    
    # 恢复训练（如果存在检查点）
    start_epoch = 0
    best_iou = 0.0
    best_loss = float('inf')
    metrics_history = {'train': [], 'test': []}
    
    if config.RESUME and os.path.exists(config.RESUME_CHECKPOINT):
        print(f"Resuming from checkpoint: {config.RESUME_CHECKPOINT}")
        checkpoint = torch.load(config.RESUME_CHECKPOINT, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_iou = checkpoint['best_iou']
        best_loss = checkpoint['best_loss']
        metrics_history = checkpoint['metrics_history']
        
        print(f"Resumed from epoch {start_epoch}, best IoU: {best_iou:.4f}")
    
    # 训练循环
    patience_counter = 0
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        accumulation_steps = 0
        
        # 用于记录训练指标
        train_metrics = {
            'iou': [], 'f1': [], 'precision': [], 'recall': [], 'accuracy': [], 'mae': []
        }
        
        # 训练批次
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(config.DEVICE, non_blocking=True), target.to(config.DEVICE, non_blocking=True)
            
            # 混合精度训练
            with amp.autocast(enabled=config.USE_AMP):
                # 前向传播
                output = model(data)
                
                # 计算损失
                loss = criterion(output, target)
                loss = loss / config.GRADIENT_ACCUMULATION_STEPS  # 梯度累积
            
            # 反向传播
            scaler.scale(loss).backward()
            
            # 梯度累积
            accumulation_steps += 1
            if accumulation_steps % config.GRADIENT_ACCUMULATION_STEPS == 0:
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                
                # 更新参数
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # 更新学习率（预热阶段）
                if epoch < config.WARMUP_EPOCHS:
                    warmup_scheduler.step()
            
            epoch_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS
            
            # 计算训练指标
            with torch.no_grad():
                metrics = calculate_metrics(target, output['main'])
                metrics['mae'] = calculate_mae(target, output['main'])
                metrics['accuracy'] = calculate_accuracy(target, output['main'])
                
                for key in train_metrics.keys():
                    train_metrics[key].append(metrics[key])
            
            # 更新进度条
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss.item() * config.GRADIENT_ACCUMULATION_STEPS:.4f}',
                'iou': f'{np.mean(train_metrics["iou"]):.4f}',
                'lr': f'{current_lr:.2e}'
            })
        
        # 更新学习率（余弦退火）
        if epoch >= config.WARMUP_EPOCHS:
            scheduler.step()
        
        # 计算平均训练损失和指标
        avg_train_loss = epoch_loss / len(dataloader)
        avg_train_metrics = {key: np.mean(values) for key, values in train_metrics.items()}
        
        # 记录训练指标
        train_metrics_record = {
            'epoch': epoch + 1,
            'loss': avg_train_loss,
            **avg_train_metrics
        }
        metrics_history['train'].append(train_metrics_record)
        
        # 输出训练指标
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}')
        print(f'Train Metrics - IoU: {avg_train_metrics["iou"]:.4f}, '
              f'Dice: {avg_train_metrics["f1"]:.4f}, '
              f'Accuracy: {avg_train_metrics["accuracy"]:.4f}, '
              f'MAE: {avg_train_metrics["mae"]:.4f}, '
              f'Recall: {avg_train_metrics["recall"]:.4f}, '
              f'Precision: {avg_train_metrics["precision"]:.4f}')
        
        # 保存训练指标到日志文件
        with open(log_file, 'a') as f:
            f.write(f'Epoch {epoch+1}/{num_epochs}\n')
            f.write(f'Train Loss: {avg_train_loss:.4f}\n')
            f.write(f'Train Metrics - IoU: {avg_train_metrics["iou"]:.4f}, '
                    f'Dice: {avg_train_metrics["f1"]:.4f}, '
                    f'Accuracy: {avg_train_metrics["accuracy"]:.4f}, '
                    f'MAE: {avg_train_metrics["mae"]:.4f}, '
                    f'Recall: {avg_train_metrics["recall"]:.4f}, '
                    f'Precision: {avg_train_metrics["precision"]:.4f}\n')
        
        # 每N个epoch或最后一个epoch，计算测试指标并保存可视化结果
        if (epoch + 1) % config.EVAL_INTERVAL == 0 or epoch == num_epochs - 1:
            test_loss, test_metrics = evaluate_model(model, test_dataloader, criterion)
            
            # 记录测试指标
            test_metrics_record = {
                'epoch': epoch + 1,
                'loss': test_loss,
                **test_metrics
            }
            metrics_history['test'].append(test_metrics_record)
            
            print(f'Test Loss: {test_loss:.4f}')
            print(f'Test Metrics - IoU: {test_metrics["iou"]:.4f}, '
                  f'Dice: {test_metrics["f1"]:.4f}, '
                  f'Accuracy: {test_metrics["accuracy"]:.4f}, '
                  f'MAE: {test_metrics["mae"]:.4f}, '
                  f'Recall: {test_metrics["recall"]:.4f}, '
                  f'Precision: {test_metrics["precision"]:.4f}')
            
            # 保存测试指标到日志文件
            with open(log_file, 'a') as f:
                f.write(f'Test Loss: {test_loss:.4f}\n')
                f.write(f'Test Metrics - IoU: {test_metrics["iou"]:.4f}, '
                        f'Dice: {test_metrics["f1"]:.4f}, '
                        f'Accuracy: {test_metrics["accuracy"]:.4f}, '
                        f'MAE: {test_metrics["mae"]:.4f}, '
                        f'Recall: {test_metrics["recall"]:.4f}, '
                        f'Precision: {test_metrics["precision"]:.4f}\n')
            
            # 保存可视化结果
            if (epoch + 1) % config.VIS_INTERVAL == 0 or epoch == num_epochs - 1:
                save_visualizations(model, test_dataloader, epoch+1, dataset_name, model_name)
            
            # 保存检查点
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': test_loss,
                'metrics': test_metrics,
                'best_iou': best_iou,
                'best_loss': best_loss,
                'metrics_history': metrics_history,
                'config': config_dict
            }
            
#             # 保存常规检查点
#             torch.save(checkpoint, os.path.join(
#                 config.CHECKPOINT_DIR, 
#                 f'checkpoint_{experiment_id}_epoch{epoch+1}_iou{test_metrics["iou"]:.4f}.pth'
#             ))
            
            # 保存最佳模型（基于IoU）
            if test_metrics["iou"] > best_iou:
                best_iou = test_metrics["iou"]
                best_loss = test_loss
                patience_counter = 0
                
                torch.save(checkpoint, os.path.join(
                    config.CHECKPOINT_DIR, 
                    f'best_model_{dataset_name}_{model_name}.pth'
                ))
                print(f"New best model saved with IoU: {best_iou:.4f}")
            else:
                patience_counter += 1
                
            # 早停检查
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # 保存指标历史
        with open(metrics_file, 'w') as f:
            json.dump(metrics_history, f, indent=4)
    
#     # 保存最终模型
#     final_checkpoint = {
#         'epoch': num_epochs,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'scheduler_state_dict': scheduler.state_dict(),
#         'scaler_state_dict': scaler.state_dict(),
#         'best_iou': best_iou,
#         'best_loss': best_loss,
#         'metrics_history': metrics_history,
#         'config': config_dict
#     }
    
#     torch.save(final_checkpoint, os.path.join(
#         config.CHECKPOINT_DIR, 
#         f'final_model_{experiment_id}_iou{best_iou:.4f}.pth'
#     ))
    
    # 记录训练结束时间
    config_dict['end_time'] = datetime.now().isoformat()
    config_dict['training_duration'] = str(datetime.now() - datetime.fromisoformat(config_dict['start_time']))
    config_dict['best_iou'] = best_iou
    config_dict['best_loss'] = best_loss
    
    with open(os.path.join(config.LOG_DIR, f'config_{experiment_id}.json'), 'w') as f:
        json.dump(config_dict, f, indent=4)
    
    print(f"Training completed! Best IoU: {best_iou:.4f}")

if __name__ == "__main__":
    # 示例：训练Kvasir数据集
    # 注意：这里需要先加载数据并划分训练集和测试集
    # 这里只是一个示例，实际使用时需要根据您的数据加载方式进行调整
    X_train, y_train = load_data(config.IMG_HEIGHT, config.IMG_WIDTH, -1, 'kvasir')
    X_test, y_test = load_data(config.IMG_HEIGHT, config.IMG_WIDTH, -1, 'cvc-clinicdb')  # 示例中使用不同的测试集
    
    train_model(X_train, X_test, y_train, y_test, 'kvasir', config.MODEL_NAME, config.NUM_EPOCHS, config.PATIENCE)
