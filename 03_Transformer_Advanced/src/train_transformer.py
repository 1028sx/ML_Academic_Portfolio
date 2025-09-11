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
import matplotlib.pyplot as plt

# 本地模組導入
import transformer_config as cfg
from models import get_model
from data_pipeline import get_dataloaders
from utils import EarlyStopping
from helpers import plot_accuracy, plot_loss

def setup_logging(experiment_name):
    """為每個實驗設定專用的日誌記錄器。"""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    log_filename = f"experiment_{experiment_name}.log"
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

def get_optimizer(model, exp_config):
    """根據實驗配置獲取優化器。"""
    optimizer_type = exp_config.get('optimizer', 'AdamW')
    lr = exp_config.get('lr') # 無預設值
    wd = exp_config.get('wd', 1e-5)

    # 如果未提供學習率，假設排程器會處理它。
    # 在這種情況下，我們為優化器初始化提供一個佔位符值。
    if lr is None:
        if not exp_config.get('scheduler'):
            raise ValueError("Config error: 'lr' is missing and no scheduler is defined.")
        # 這是需要初始學習率的優化器的佔位符值。
        # 排程器（如 OneCycleLR）會立即覆蓋這個值。
        lr = 1e-4 

    if optimizer_type == 'AdamW':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_type == 'Adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"不支援的優化器類型: {optimizer_type}")

def get_scheduler(optimizer, exp_config, train_loader):
    """根據實驗配置獲取學習率排程器。"""
    scheduler_config = exp_config.get('scheduler')
    if not scheduler_config:
        return None
    
    scheduler_name = scheduler_config.get('name')
    if scheduler_name == 'OneCycleLR':
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=scheduler_config.get('max_lr', 1e-3),
            epochs=exp_config.get('epochs'),
            steps_per_epoch=len(train_loader),
            pct_start=scheduler_config.get('pct_start', 0.3),
            div_factor=scheduler_config.get('div_factor', 25),
            final_div_factor=scheduler_config.get('final_div_factor', 1e4)
        )
    else:
        raise ValueError(f"不支援的排程器類型: {scheduler_name}")

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device):
    """處理單個訓練週期的訓練迴圈。"""
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler and isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        if torch.isnan(loss):
            logging.error("損失值變成 NaN。停止訓練。")
            raise ValueError("檢測到 NaN 損失")

        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

    train_loss = total_loss / total_samples
    train_acc = total_correct / total_samples
    return train_loss, train_acc

def validate_one_epoch(model, val_loader, criterion, device):
    """處理單個驗證週期的驗證迴圈。"""
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

    val_loss = total_val_loss / total_val_samples
    val_acc = total_val_correct / total_val_samples
    return val_loss, val_acc

def run_experiment(experiment_name: str):
    """
    執行單一實驗的主函式。
    從設定中心載入所有配置，執行訓練和評估流程。
    """
    try:
        exp_config = cfg.get_config(experiment_name)
    except ValueError as e:
        logging.error(str(e))
        return

    setup_logging(experiment_name)
    
    logging.info(f"--- 開始實驗: {experiment_name} ---")
    logging.info(f"使用設備: {exp_config['device']}")
    logging.info("本次執行的完整配置:")
    logging.info(json.dumps(exp_config, indent=4, default=str))

    # --- 準備資料 ---
    # 將資料路徑加入到配置中供資料載入器使用
    exp_config['data_dir'] = cfg.DATA_DIR
    exp_config['csv_path'] = cfg.TRAIN_CSV_PATH
    train_loader, val_loader = get_dataloaders(exp_config)

    # --- 初始化模型 ---
    model = get_model(exp_config['model_blueprint'], exp_config['audio_settings'])
    model.to(exp_config['device'])
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"模型 '{exp_config['model_blueprint']['model_class']}' 初始化成功。")
    logging.info(f"可訓練參數數量: {num_params:,}")

    # --- 設定優化器、損失函式和排程器 ---
    optimizer = get_optimizer(model, exp_config)
    
    # 增加標籤平滑支援
    label_smoothing = exp_config.get('label_smoothing', 0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if label_smoothing > 0.0:
        logging.info(f"使用標籤平滑，因子為 {label_smoothing}")
        
    scheduler = get_scheduler(optimizer, exp_config, train_loader)
    
    # --- 設定早停機制 ---
    checkpoint_name = f"checkpoint_{experiment_name}.pth"
    checkpoint_path = os.path.join(cfg.MODEL_SAVE_DIR, checkpoint_name)
    os.makedirs(cfg.MODEL_SAVE_DIR, exist_ok=True)
    
    patience = exp_config.get('patience', 15)
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=checkpoint_path)

    # --- 訓練迴圈 ---
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    epochs = exp_config.get('epochs', 100)
    device = exp_config['device']

    for epoch in range(epochs):
        try:
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
        except ValueError as e:
            logging.error(f"因第 {epoch+1} 個週期出現錯誤而停止: {e}")
            return # 停止實驗

        logging.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if scheduler and not isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            logging.info("早停機制觸發！")
            break

    # --- 儲存最終模型和結果 ---
    logging.info(f"從檢查點載入最佳模型: '{early_stopping.path}'")
    model.load_state_dict(torch.load(early_stopping.path, weights_only=True))
    
    # 儲存最終模型狀態
    final_model_name = f"{experiment_name}.pth"
    final_model_path = os.path.join(cfg.MODEL_SAVE_DIR, final_model_name)
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"最終最佳模型已儲存至: {final_model_path}")

    # 儲存實驗配置
    config_save_path = os.path.join(cfg.MODEL_SAVE_DIR, f"{experiment_name}.json")
    with open(config_save_path, 'w', encoding='utf-8') as f:
        json.dump(exp_config, f, indent=4, default=str)
    logging.info(f"實驗配置已儲存至: {config_save_path}")

    # 儲存圖表
    plot_accuracy(history, experiment_name, cfg.PLOT_SAVE_DIR)
    plot_loss(history, experiment_name, cfg.PLOT_SAVE_DIR)
    logging.info(f"準確率和損失圖表已儲存至: {cfg.PLOT_SAVE_DIR}")
    
    logging.info(f"--- 實驗 {experiment_name} 完成 ---")

def main():
    """解析參數並執行指定的實驗。"""
    parser = argparse.ArgumentParser(description="執行 Transformer 音訊分類實驗。")
    parser.add_argument(
        "--experiment", 
        type=str, 
        required=True, 
        choices=list(cfg.EXPERIMENTS.keys()),
        help="要執行的實驗名稱，定義在 transformer_config.py 中"
    )
    args = parser.parse_args()
    
    run_experiment(args.experiment)

if __name__ == '__main__':
    main() 