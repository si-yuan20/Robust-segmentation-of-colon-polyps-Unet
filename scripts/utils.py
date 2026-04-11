import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import label, regionprops
from sklearn.metrics import jaccard_score, f1_score, precision_score, recall_score

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
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

def compute_hd95(y_true, y_pred):
    """计算95%豪斯多夫距离"""
    y_true = np.squeeze(y_true).astype(np.bool_)
    y_pred = np.squeeze(y_pred).astype(np.bool_)

    if not np.any(y_true) and not np.any(y_pred):
        return 0.0
    if not np.any(y_true) or not np.any(y_pred):
        return 100.0

    y_true_dist = distance_transform_edt(~y_true)
    y_pred_dist = distance_transform_edt(~y_pred)

    border_true = np.logical_xor(y_true, distance_transform_edt(y_true) < 1)
    border_pred = np.logical_xor(y_pred, distance_transform_edt(y_pred) < 1)

    true_points = np.argwhere(border_true)
    pred_points = np.argwhere(border_pred)

    dist_true = np.min(y_pred_dist[true_points[:, 0], true_points[:, 1]]) if len(true_points) > 0 else 0
    dist_pred = np.min(y_true_dist[pred_points[:, 0], pred_points[:, 1]]) if len(pred_points) > 0 else 0

    hd95 = max(np.percentile(y_pred_dist[true_points], 95) if len(true_points) > 0 else 0,
                np.percentile(y_true_dist[pred_points], 95) if len(pred_points) > 0 else 0)
    return hd95

def compute_assd(y_true, y_pred):
    """计算平均对称表面距离 ASSD"""
    y_true = np.squeeze(y_true).astype(np.bool_)
    y_pred = np.squeeze(y_pred).astype(np.bool_)

    if not np.any(y_true) and not np.any(y_pred):
        return 0.0
    if not np.any(y_true) or not np.any(y_pred):
        return 100.0

    y_true_dist = distance_transform_edt(~y_true)
    y_pred_dist = distance_transform_edt(~y_pred)

    border_true = np.argwhere(np.logical_xor(y_true, distance_transform_edt(y_true) < 1))
    border_pred = np.argwhere(np.logical_xor(y_pred, distance_transform_edt(y_pred) < 1))

    assd = (np.mean(y_pred_dist[border_true[:, 0], border_true[:, 1]]) +
            np.mean(y_true_dist[border_pred[:, 0], border_pred[:, 1]])) / 2
    return assd

def compute_bf_score(y_true, y_pred):
    """计算BF分数 (Boundary F1-Score)"""
    y_true = np.squeeze(y_true).astype(np.bool_)
    y_pred = np.squeeze(y_pred).astype(np.bool_)

    if not np.any(y_true) and not np.any(y_pred):
        return 1.0
    if not np.any(y_true) or not np.any(y_pred):
        return 0.0

    dt = distance_transform_edt(~y_true)
    dp = distance_transform_edt(~y_pred)

    border_t = np.logical_xor(y_true, dt < 1)
    border_p = np.logical_xor(y_pred, dp < 1)

    t2p = dt[border_p].sum() / max(border_p.sum(), 1)
    p2t = dp[border_t].sum() / max(border_t.sum(), 1)

    bf = 2 / (1 / (1e-5 + t2p) + 1 / (1e-5 + p2t))
    return np.clip(bf, 0, 1)

def calculate_metrics(y_true, y_pred):
    """计算所有分割指标（含新增HD95/ASSD/BF-Score）"""
    y_pred_bin = (torch.sigmoid(y_pred) > 0.5).float()
    
    y_true_np = y_true.cpu().numpy().astype(np.uint8)
    y_pred_np = y_pred_bin.cpu().numpy().astype(np.uint8)
    
    iou = jaccard_score(y_true_np.flatten(), y_pred_np.flatten(), zero_division=0)
    f1 = f1_score(y_true_np.flatten(), y_pred_np.flatten(), zero_division=0)
    precision = precision_score(y_true_np.flatten(), y_pred_np.flatten(), zero_division=0)
    recall = recall_score(y_true_np.flatten(), y_pred_np.flatten(), zero_division=0)
    accuracy = calculate_accuracy(y_true_np.flatten(), y_pred_np.flatten())

    hd95_list, assd_list, bf_list = [], [], []
    for bt, bp in zip(y_true_np, y_pred_np):
        hd95_list.append(compute_hd95(bt, bp))
        assd_list.append(compute_assd(bt, bp))
        bf_list.append(compute_bf_score(bt, bp))

    return {
        'iou': iou, 'f1': f1, 'precision': precision, 'recall': recall, 'accuracy': accuracy,
        'hd95': float(np.mean(hd95_list)),
        'assd': float(np.mean(assd_list)),
        'bf_score': float(np.mean(bf_list))
    }

def calculate_mae(y_true, y_pred):
    y_pred_sig = torch.sigmoid(y_pred)
    return torch.abs(y_pred_sig - y_true).mean().item()

def calculate_accuracy(y_true, y_pred):
    if isinstance(y_true, np.ndarray):
        y_true = torch.tensor(y_true)
        y_pred = torch.tensor(y_pred)
    return (y_pred == y_true).float().mean().item()

def get_loss_function(loss_name='bce_dice'):
    if loss_name == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss_name == 'dice':
        return DiceLoss()
    elif loss_name == 'bce_dice':
        return BCEDiceLoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
