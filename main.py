import argparse
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import config
from train import train_model
from val import validate_model
from ImageLoader2D import load_data
from sklearn.model_selection import train_test_split

def split_data(X, y, train_size=0.8, test_size=0.1, val_size=0.1, random_state=None):
    
    if not np.isclose(train_size + test_size + val_size, 1.0):
        raise ValueError("训练集、测试集和验证集的比例之和必须为1.0")
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=test_size + val_size, 
        random_state=random_state
    )
    
    test_proportion = test_size / (test_size + val_size)
    
    X_test, X_val, y_test, y_val = train_test_split(
        X_temp, y_temp, 
        test_size=1 - test_proportion, 
        random_state=random_state
    )
    
    return X_train, X_test, X_val, y_train, y_test, y_val

def main():
    parser = argparse.ArgumentParser(description='Polyp Segmentation Training')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'val'],
                        help='Mode: train, test, or val')
    parser.add_argument('--dataset', type=str, default='kvasir', 
                        choices=['kvasir', 'cvc-clinicdb', 'cvc-colondb'],
                        help='Dataset to use')
    parser.add_argument('--model', type=str, default=config.MODEL_NAME,
                        choices=['PolypSegNet_Full', 'PolypSegNet_Resnet_SCAS', 'PolypSegNet_Resnet_D2F', 
                                'PolypSegNet_Resnet', 'PolypSegNet_D2F', 'PolypSegNet_Baseline'],
                        help='Model architecture')
    parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE,
                        help='Learning rate')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint for testing or validation')

    parser.add_argument('--seed', type=str, default=2024,
                        help='seed')
    args = parser.parse_args()
    
    config.NUM_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    
    print(f"Using device: {config.DEVICE}")
    
    X, Y = load_data(
    config.IMG_HEIGHT, 
    config.IMG_WIDTH, 
    -1,  # 加载所有图像
    args.dataset
    )
    
    X_train, X_test, X_val, y_train, y_test, y_val = split_data(
        X, Y, 
        train_size=0.8, 
        test_size=0.1, 
        val_size=0.1,
        random_state=42
    )
    
    # print(f"Training {args.model} on {args.dataset} dataset...")
    # train_model(X_train, X_test, y_train, y_test,args.dataset,args.model, args.epochs, config.PATIENCE)

    checkpoint_path = f"./checkpoints/best_model_{args.dataset}_{args.model}.pth"
    validate_model(X_val,y_val, args.dataset,args.model, checkpoint_path)

if __name__ == "__main__":
    main()
