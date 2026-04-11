import glob
import json
import os
import numpy as np
from PIL import Image
from skimage.io import imread
from tqdm import tqdm
import concurrent.futures
import config

root_path = config.dataset_root_path

# 各数据集路径配置
dataset_path_config = {
   
    'cvc-clinicdb': {
        'img_dir': 'cvc-clinicdb/images/',
        'mask_dir': 'cvc-clinicdb/masks/',
        'img_ext': '.png'
    },

}

# split_indices 文件夹路径
split_indices_root = os.path.join(root_path, 'split_indices')


def process_image_mask_pair(args):
    """
    处理单个图像-掩码对的辅助函数
    Args:
        args: (img_path, mask_path, img_height, img_width, n)
    Returns:
        n: 索引, image_array: 归一化图像, mask_array: 二值掩码
    """
    img_path, mask_path, img_height, img_width, n = args
    
    # 读取图像 + resize + 归一化
    image = imread(img_path)
    pillow_image = Image.fromarray(image).resize((img_width, img_height), resample=Image.BILINEAR)
    image_array = np.array(pillow_image, dtype=np.float32) / 255.0  # 归一化到[0,1]
    
    # 读取掩码 + 灰度化 + resize(最近邻) + 二值化
    mask_ = imread(mask_path)
    pillow_mask = Image.fromarray(mask_).convert('L').resize(
        (img_width, img_height), resample=Image.NEAREST  # 最近邻插值保持二值
    )
    mask_array = np.array(pillow_mask, dtype=np.uint8)
    mask_array = (mask_array >= 127).astype(np.uint8)  # 阈值化，确保0/1
    
    return n, image_array, mask_array


def get_split_file_paths(dataset_name, split_type):
    """
    从split_indices文件夹读取划分，获取对应split的图像/掩码路径列表
    Args:
        dataset_name: 数据集名称 (kvasir/cvc-clinicdb/...)
        split_type: 划分类型 (train/val/test)
    Returns:
        img_paths: 图像路径列表, mask_paths: 掩码路径列表
    """
    # 1. 读取split文件（优先txt，兼容json）
    split_file = os.path.join(split_indices_root, f'{split_type}.txt')
    if not os.path.exists(split_file):
        split_file = os.path.join(split_indices_root, f'split_indices.json')
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"划分文件不存在: {split_indices_root} 下无 {split_type}.txt 或 split_indices.json")
        
        # 从json读取划分
        with open(split_file, 'r', encoding='utf-8') as f:
            split_data = json.load(f)
        if split_type not in split_data:
            raise KeyError(f"json文件中无 {split_type} 划分")
        split_names = split_data[split_type]
    else:
        # 从txt读取划分（每行一个文件名，不含后缀）
        with open(split_file, 'r', encoding='utf-8') as f:
            split_names = [line.strip() for line in f if line.strip()]
    
    # 2. 获取数据集路径配置
    if dataset_name not in dataset_path_config:
        raise ValueError(f"不支持的数据集: {dataset_name}，支持列表: {list(dataset_path_config.keys())}")
    cfg = dataset_path_config[dataset_name]
    img_dir = os.path.join(root_path, cfg['img_dir'])
    mask_dir = os.path.join(root_path, cfg['mask_dir'])
    img_ext = cfg['img_ext']
    
    # 3. 生成完整路径
    img_paths = []
    mask_paths = []
    for name in split_names:
        # 自动补全后缀（兼容txt中带/不带后缀的情况）
        if not name.endswith(img_ext):
            img_name = f'{name}{img_ext}'
            mask_name = f'{name}{img_ext}'  # 掩码文件名与图像一致
        else:
            img_name = name
            mask_name = name
        
        img_path = os.path.join(img_dir, img_name)
        mask_path = os.path.join(mask_dir, mask_name)
        
        # 校验文件存在
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图像文件不存在: {img_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"掩码文件不存在: {mask_path}")
        
        img_paths.append(img_path)
        mask_paths.append(mask_path)
    
    return img_paths, mask_paths


def load_split_data(img_height, img_width, dataset, split_type, images_to_be_loaded=-1, max_workers=4):
    """
    按split划分加载数据（核心入口函数）
    Args:
        img_height: 目标高度
        img_width: 目标宽度
        dataset: 数据集名称
        split_type: 划分类型 (train/val/test)
        images_to_be_loaded: 加载数量，-1表示全部
        max_workers: 多线程最大线程数
    Returns:
        X: 图像数组 (N, H, W, 3) float32
        Y: 掩码数组 (N, H, W, 1) uint8
    """
    # 1. 获取划分后的路径列表
    img_paths, mask_paths = get_split_file_paths(dataset, split_type)
    total_samples = len(img_paths)
    
    # 2. 确定加载数量
    if images_to_be_loaded == -1:
        load_num = total_samples
    else:
        load_num = min(images_to_be_loaded, total_samples)
    
    # 截断到指定数量
    img_paths = img_paths[:load_num]
    mask_paths = mask_paths[:load_num]
    
    # 3. 预分配内存（提升效率）
    X = np.zeros((load_num, img_height, img_width, 3), dtype=np.float32)
    Y = np.zeros((load_num, img_height, img_width, 1), dtype=np.uint8)
    
    print(f'[Dataset: {dataset} | Split: {split_type}] 正在加载 {load_num} 个样本...')
    
    # 4. 多线程并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 准备参数列表
        args_list = [
            (img_paths[n], mask_paths[n], img_height, img_width, n)
            for n in range(load_num)
        ]
        
        # 提交任务 + 进度条
        futures = {executor.submit(process_image_mask_pair, args): args for args in args_list}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            n, image, mask = future.result()
            X[n] = image
            Y[n, :, :, 0] = mask  # 直接赋值到预分配数组
    
    print(f'[Dataset: {dataset} | Split: {split_type}] 加载完成，样本数: {load_num}')
    return X, Y


# ------------------------------
# 兼容原函数的兼容层（可选，保留原调用方式）
# ------------------------------
def load_data(img_height, img_width, images_to_be_loaded, dataset):
    """
    兼容原函数接口，默认加载train划分（向后兼容）
    """
    return load_split_data(
        img_height=img_height,
        img_width=img_width,
        dataset=dataset,
        split_type='train',
        images_to_be_loaded=images_to_be_loaded
    )


# ------------------------------
# 示例调用
# ------------------------------
if __name__ == '__main__':
    h, w = 352, 352
    dataset_name = 'cvc-clinicdb'
    
    # 加载训练集
    X_train, Y_train = load_split_data(h, w, dataset_name, 'train')
    # 加载验证集
    X_val, Y_val = load_split_data(h, w, dataset_name, 'val')
    # 加载测试集
    X_test, Y_test = load_split_data(h, w, dataset_name, 'test')
    
    print(f"Train shape: {X_train.shape}, {Y_train.shape}")
    print(f"Val shape: {X_val.shape}, {Y_val.shape}")
    print(f"Test shape: {X_test.shape}, {Y_test.shape}")
