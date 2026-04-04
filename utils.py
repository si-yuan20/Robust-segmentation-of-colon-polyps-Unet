# utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score, accuracy_score, mean_absolute_error

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        
        # 展平预测和目标
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class BCEDiceLoss(nn.Module):
    def __init__(self, weight=0.5, smooth=1e-5):
        super(BCEDiceLoss, self).__init__()
        self.dice = DiceLoss(smooth)
        self.bce = nn.BCEWithLogitsLoss()
        self.weight = weight

    def forward(self, inputs, targets):
        return self.weight * self.bce(inputs, targets) + (1 - self.weight) * self.dice(inputs, targets)

class MultiScaleLoss(nn.Module):
    def __init__(self, loss_fn, weights=None):
        super(MultiScaleLoss, self).__init__()
        self.loss_fn = loss_fn
        self.weights = weights if weights is not None else [1.0, 0.8, 0.6, 0.4, 0.2]

    def forward(self, outputs, targets):
        main_loss = self.loss_fn(outputs['main'], targets)
        ds1_loss = self.loss_fn(outputs['ds1'], targets)
        ds2_loss = self.loss_fn(outputs['ds2'], targets)
        ds3_loss = self.loss_fn(outputs['ds3'], targets)
        ds4_loss = self.loss_fn(outputs['ds4'], targets)
        
        total_loss = (
            self.weights[0] * main_loss +
            self.weights[1] * ds1_loss +
            self.weights[2] * ds2_loss +
            self.weights[3] * ds3_loss +
            self.weights[4] * ds4_loss
        )
        
        return total_loss

def calculate_metrics(y_true, y_pred):
    """计算评估指标"""
    y_pred_bin = y_pred > 0.5  # 二值化
    
    # 转换为numpy数组并展平
    y_true_np = y_true.cpu().numpy().flatten()
    y_pred_np = y_pred_bin.cpu().numpy().flatten()
    
    # 计算指标
    iou = jaccard_score(y_true_np, y_pred_np, zero_division=0)
    f1 = f1_score(y_true_np, y_pred_np, zero_division=0)
    precision = precision_score(y_true_np, y_pred_np, zero_division=0)
    recall = recall_score(y_true_np, y_pred_np, zero_division=0)
    accuracy = calculate_accuracy(y_true_np, y_pred_np)

    return {
        'iou': iou,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy
    }

def calculate_mae(y_true, y_pred):
    """计算MAE"""
    y_pred_sig = torch.sigmoid(y_pred)
    mae = torch.abs(y_pred_sig - y_true).mean().item()
    return mae

# def calculate_accuracy(y_true, y_pred):
#     """计算Accuracy"""
#     y_pred_bin = y_pred > 0.5  # 二值化
#     accuracy = (y_pred_bin == y_true).float().mean().item()
#     return accuracy


def calculate_accuracy(y_true, y_pred):
    # 转换为PyTorch张量
    y_true_tensor = torch.tensor(y_true, dtype=torch.float32)
    y_pred_tensor = torch.tensor(y_pred, dtype=torch.float32)

    # 使用PyTorch的方式计算准确率
    accuracy = (y_pred_tensor == y_true_tensor).float().mean().item()
    return accuracy

def get_loss_function(loss_name='bce_dice'):
    """获取损失函数"""
    if loss_name == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss_name == 'dice':
        return DiceLoss()
    elif loss_name == 'bce_dice':
        return BCEDiceLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")