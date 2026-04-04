import glob
import numpy as np
from PIL import Image
from skimage.io import imread
from tqdm import tqdm
import concurrent.futures
import config

root_path = config.dataset_root_path
kvasir_folder_path = "kvasir/"
clinicdb_folder_path = "cvc-clinicdb/"
colondb_folder_path = "cvc-colondb/"
cvc_300_folder_path = "cvc-t/"
larib_folder_path = "ETIS-LaribPolypDB/"

def process_image_mask_pair(args):
    """处理单个图像-掩码对的辅助函数"""
    img_path, img_height, img_width, n = args
    mask_path = img_path.replace("images", "masks")
    
    # 读取图像
    image = imread(img_path)
    pillow_image = Image.fromarray(image).resize((img_width, img_height))  # 注意尺寸顺序
    image_array = np.array(pillow_image, dtype=np.float32) / 255.0
    
    # 读取并处理掩码
    mask_ = imread(mask_path)
    pillow_mask = Image.fromarray(mask_).convert('L').resize(
        (img_width, img_height), resample=Image.NEAREST  # 使用NEAREST保持二值特性
    )
    mask_array = np.array(pillow_mask, dtype=np.uint8)
    mask_array = (mask_array >= 127).astype(np.uint8)  # 向量化阈值处理
    
    return n, image_array, mask_array

def load_data(img_height, img_width, images_to_be_loaded, dataset):
    """加载图像和掩码数据的高效实现"""
    
    # 确定数据集路径
    if dataset == 'kvasir':
        IMAGES_PATH = root_path + kvasir_folder_path + 'images/'
        train_ids = glob.glob(IMAGES_PATH + "*.jpg")
    elif dataset == 'cvc-clinicdb':
        IMAGES_PATH = root_path + clinicdb_folder_path + 'images/'
        train_ids = glob.glob(IMAGES_PATH + "*.png")
    elif dataset == 'cvc-colondb':
        IMAGES_PATH = root_path + colondb_folder_path + "images/"
        train_ids = glob.glob(IMAGES_PATH + "*.png")
    elif dataset == 'cvc-t':
        IMAGES_PATH = root_path + cvc_300_folder_path + 'images/'
        print(IMAGES_PATH)
        train_ids = glob.glob(IMAGES_PATH + "*.png")
    elif dataset == 'ETIS-LaribPolypDB':
        IMAGES_PATH = root_path + larib_folder_path + "images/"
        train_ids = glob.glob(IMAGES_PATH + "*.png")
    else:
        raise ValueError(f"未知的数据集: {dataset}")
    
    # 确定要加载的图像数量
    if images_to_be_loaded == -1:
        images_to_be_loaded = len(train_ids)
    else:
        images_to_be_loaded = min(images_to_be_loaded, len(train_ids))
    
    # 预分配内存
    X_train = np.zeros((images_to_be_loaded, img_height, img_width, 3), dtype=np.float32)
    Y_train = np.zeros((images_to_be_loaded, img_height, img_width, 1), dtype=np.uint8)  # 直接添加最后一个维度
    
    print(f'正在加载和调整 {images_to_be_loaded} 个训练图像和掩码...')
    
    # 使用多线程处理图像
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # 准备参数
        args_list = [(id_, img_height, img_width, n) for n, id_ in enumerate(train_ids[:images_to_be_loaded])]
        
        # 使用tqdm显示进度
        futures = {executor.submit(process_image_mask_pair, args): args for args in args_list}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            n, image, mask = future.result()
            X_train[n] = image
            Y_train[n, :, :, 0] = mask  # 直接赋值到预分配的数组中
    
    return X_train, Y_train