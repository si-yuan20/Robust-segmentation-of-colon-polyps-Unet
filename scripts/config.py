import torch

dataset_root_path = "datasets/"  
clinicdb_folder_path = "cvc-clinicdb/"
colondb_folder_path = "cvc-colondb/"
cvc_300_folder_path = "cvc-t/"
larib_folder_path = "ETIS-LaribPolypDB/"


BATCH_SIZE = 8
LEARNING_RATE = 1e-3
NUM_EPOCHS = 200
PATIENCE = 10 
WEIGHT_DECAY = 1e-4 
MIN_LR = 1e-6  

MODEL_NAME = "PolypSegNet_Full"  
BASE_CH = 8
USE_RESNET = True

IMG_HEIGHT = 352
IMG_WIDTH = 352

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4

CHECKPOINT_DIR = "./checkpoints/"
LOG_DIR = "./logs/"
RESULTS_DIR = "./results/"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

USE_AUGMENTATION = True
AUGMENTATION_PROB = 0.5

VISUALIZATION_NUM_IMAGES = 4  
VIS_INTERVAL = 10  
EVAL_INTERVAL = 5 

USE_AMP = True  
GRADIENT_ACCUMULATION_STEPS = 2 
MAX_GRAD_NORM = 1.0  

WARMUP_EPOCHS = 5  
WARMUP_FACTOR = 0.1  

RESUME = False
RESUME_CHECKPOINT = "./checkpoints/latest_checkpoint.pth" 
