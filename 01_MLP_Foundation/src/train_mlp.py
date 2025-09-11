# 標準庫導入
import os
import argparse
import logging
import json
from datetime import datetime

# 第三方庫導入
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 本地模組導入
from .mlp_models import get_model
from .data_pipeline import get_dataloaders
from .utils import EarlyStopping
from .helpers import plot_accuracy, plot_loss
from . import mlp_config as cfg

# 全域變數
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_logging(experiment_name):
    """為每個實驗設定一個簡單的日誌記錄器。"""
    # 移除現有的所有 handlers，避免重複日誌
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_filename = f"mlp_experiment_{experiment_name}.log"
    log_filepath = os.path.join(cfg.LOG_SAVE_DIR, log_filename)
    os.makedirs(cfg.LOG_SAVE_DIR, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def get_optimizer(model, optimizer_name, lr, wd):
    """根據名稱和參數獲取優化器。"""
    if optimizer_name.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_name.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        # 預設返回 Adam
        logging.warning(f"不支援的優化器 '{optimizer_name}'，將使用 Adam。")
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

def run_experiment(experiment_name: str):
    """
    執行單一實驗的主函式。
    從設定中心加載所有配置，執行訓練和評估流程。
    """
    # 1. 獲取完整的、合併後的實驗設定
    try:
        config = cfg.get_config(experiment_name)
    except ValueError as e:
        logging.error(e)
        return

    # 2. 設定日誌
    setup_logging(experiment_name)
    
    logging.info(f"--- 開始MLP實驗: {experiment_name} ---")
    logging.info(f"使用設備: {device}")

    # 3. 準備資料
    train_loader, val_loader, data_dims = get_dataloaders(config)
    
    # 4. 初始化模型
    # 將所有模型相關的設定合併到一個字典中
    model_config = {**config['model'], **data_dims}
    model = get_model(model_config)
    model.to(device)
    logging.info(f"模型 '{config['model']['model_class']}' 初始化成功。")

    # 5. 設定優化器和損失函數
    train_params = config['train']
    optimizer = get_optimizer(model, train_params['optimizer'], train_params['lr'], train_params['wd'])
    criterion = nn.CrossEntropyLoss()
    
    # 6. 設定 Early Stopping 和存檔路徑
    checkpoint_name = f"mlp_{experiment_name}.pth"
    checkpoint_path = os.path.join(config['model_save_dir'], checkpoint_name)
    early_stopping = EarlyStopping(patience=train_params['patience'], verbose=True, path=checkpoint_path)

    # 7. 訓練迴圈
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(train_params['epochs']):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

        epoch_train_loss = total_loss / total_samples
        epoch_train_acc = total_correct / total_samples
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)

        # 驗證迴圈
        model.eval()
        total_val_loss, total_val_correct, total_val_samples = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                total_val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val_samples += labels.size(0)
                total_val_correct += (predicted == labels).sum().item()

        epoch_val_loss = total_val_loss / total_val_samples
        epoch_val_acc = total_val_correct / total_val_samples
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        logging.info(f'Epoch {epoch+1}/{train_params["epochs"]} | '
                     f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | '
                     f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}')
        
        early_stopping(epoch_val_loss, model)
        if early_stopping.early_stop:
            logging.info("Early stopping 觸發")
            break

    logging.info("--- 訓練完成 ---")
    
    # 載入最佳模型
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    logging.info(f"已從 '{checkpoint_path}' 載入最佳模型。")

    # 儲存訓練歷史圖
    plot_acc_path = os.path.join(config['plot_save_dir'], f"mlp_{experiment_name}_accuracy.png")
    plot_loss_path = os.path.join(config['plot_save_dir'], f"mlp_{experiment_name}_loss.png")
    plot_accuracy(history, plot_acc_path)
    plot_loss(history, plot_loss_path)
    
    # 儲存最終的設定檔
    config_save_path = os.path.join(config['model_save_dir'], f"mlp_{experiment_name}.json")
    with open(config_save_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    logging.info(f"最終設定已儲存至: {config_save_path}")

def main():
    parser = argparse.ArgumentParser(description="執行 MLP 分類模型的訓練實驗。")
    parser.add_argument(
        'experiment_name', 
        type=str, 
        help='要運行的實驗名稱 (定義於 mlp_config.py)。',
        choices=list(cfg.EXPERIMENTS.keys())
    )
    args = parser.parse_args()
    
    # 直接呼叫 run_experiment
    run_experiment(args.experiment_name)

if __name__ == '__main__':
    main()