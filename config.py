import torch

# 数据集配置
dataset_root_path = "datasets/"  # 请修改为实际路径
kvasir_folder_path = "kvasir/"
clinicdb_folder_path = "cvc-clinicdb/"
colondb_folder_path = "cvc-colondb/"
cvc_300_folder_path = "cvc-t/"
larib_folder_path = "ETIS-LaribPolypDB/"


# 训练超参数
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 60
PATIENCE = 10  # 早停耐心值
WEIGHT_DECAY = 1e-3 # 权重衰减
MIN_LR = 1e-6  # 最小学习率

# 模型配置
MODEL_NAME = "PolypSegNet_Full"  # 可选
BASE_CH = 64
USE_RESNET = True

# 图像尺寸
IMG_HEIGHT = 352
IMG_WIDTH = 352

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4

# 路径配置
CHECKPOINT_DIR = "./checkpoints/"
LOG_DIR = "./logs/"
RESULTS_DIR = "./results/"

# 数据集划分
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# 数据增强
USE_AUGMENTATION = True
AUGMENTATION_PROB = 0.5

# 可视化配置
VISUALIZATION_NUM_IMAGES = 4  # 每10个epoch可视化的图像数量
VIS_INTERVAL = 10  # 可视化间隔（epoch）
EVAL_INTERVAL = 5  # 评估间隔（epoch）

# 训练优化配置
USE_AMP = True  # 使用自动混合精度
GRADIENT_ACCUMULATION_STEPS = 2  # 梯度累积步数
MAX_GRAD_NORM = 1.0  # 梯度裁剪

# 学习率预热配置
WARMUP_EPOCHS = 5  # 预热epoch数
WARMUP_FACTOR = 0.1  # 预热起始学习率因子

# 恢复训练配置
RESUME = False  # 是否从检查点恢复训练
RESUME_CHECKPOINT = "./checkpoints/latest_checkpoint.pth"  # 恢复训练的检查点路径