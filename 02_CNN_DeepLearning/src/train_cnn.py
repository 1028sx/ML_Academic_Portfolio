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
import config as cfg
from models import get_model
from data_pipeline import get_dataloaders
from utils import EarlyStopping
from helpers import plot_accuracy, plot_loss

# 全域變數
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_logging(experiment_name):
    """
    為每個實驗設定一個簡單的日誌記錄器。
    """
    # 移除現有的所有 handlers，避免重複日誌
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

def calculate_vcreg_loss(x, vcreg_params):
    """
    計算 VCReg 損失 (Variance-Covariance Regularization)。
    """
    lambda_weight = vcreg_params.get('lambda', 25.0)
    mu_weight = vcreg_params.get('mu', 25.0)
    
    # 變異數損失
    std_x = torch.sqrt(x.var(dim=0) + 1e-4) # 添加 epsilon 防止 NaN
    std_loss = torch.mean(torch.relu(1 - std_x))

    # 協方差損失
    x = x - x.mean(dim=0)
    cov_x = (x.T @ x) / (x.size(0) - 1)
    cov_loss = (cov_x.pow(2).sum() - cov_x.diag().pow(2).sum()) / x.size(1)
    
    return std_loss * lambda_weight, cov_loss * mu_weight

def get_optimizer(model, exp_config):
    """
    根據實驗設定獲取優化器。
    """
    optimizer_config = exp_config.get('optimizer', 'AdamW')
    
    if isinstance(optimizer_config, dict):
        # 新版巢狀格式
        optimizer_type = optimizer_config.get('type', 'AdamW')
        lr = optimizer_config.get('lr', 1e-4)
        wd = optimizer_config.get('wd', 1e-3)
    else:
        # 舊版扁平格式
        optimizer_type = optimizer_config
        lr = exp_config.get('lr', 1e-4)
        wd = exp_config.get('wd', 1e-3)

    if optimizer_type == 'AdamW':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_type == 'Adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_type == 'SGD':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        raise ValueError(f"不支援的優化器類型: {optimizer_type}")

def get_scheduler(optimizer, exp_config):
    """根據實驗設定獲取學習率排程器"""
    scheduler_config = exp_config.get('scheduler')
    if not scheduler_config:
        return None

    scheduler_type = scheduler_config['type']
    scheduler_params = scheduler_config.get('params', {})
    if scheduler_type == 'CosineAnnealingLR':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
    else:
        raise ValueError(f"不支援的排程器類型: {scheduler_type}")

def run_experiment(experiment_name, checkpoint_path):
    """
    執行單一實驗的主函式。
    從設定中心加載所有配置，執行訓練和評估流程。
    """
    # 1. 獲取實驗設定
    try:
        exp_config = cfg.EXPERIMENTS[experiment_name]
    except KeyError:
        logging.error(f"在 config.py 中找不到名為 '{experiment_name}' 的實驗。")
        return

    # 2. 設定日誌
    setup_logging(experiment_name)
    
    logging.info(f"--- 開始實驗: {experiment_name} ---")
    logging.info(f"使用設備: {device}")

    # 3. 準備資料
    train_loader, val_loader, data_dims = get_dataloaders(experiment_name)

    # 4. 初始化模型
    # 將所有模型相關的設定合併到一個字典中
    model_config = {**exp_config['model_blueprint'], **data_dims}
    # 如果實驗設定中有 dropout_rate (來自 Optuna)，則更新藍圖
    if 'dropout_rate' in exp_config:
        model_config['dropout_p'] = exp_config['dropout_rate']

    model = get_model(model_config)
    model.to(device)
    logging.info(f"模型 '{model_config['model_class']}' 初始化成功。")

    # 5. 設定優化器、損失函數、排程器
    optimizer = get_optimizer(model, exp_config)
    criterion = nn.CrossEntropyLoss()
    scheduler = get_scheduler(optimizer, exp_config)
    
    # 6. 設定 Early Stopping
    patience = exp_config.get('patience', 10)
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=checkpoint_path)

    # 7. 訓練迴圈
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    epochs = exp_config.get('epochs', 50)
    grad_clip_val = exp_config.get('gradient_clip_val', None)

    # 檢查是否啟用 VCReg
    vcreg_params = next((reg for reg in exp_config.get('regularizers', []) if reg.get('type') == 'vcreg'), None)

    for epoch in range(epochs):
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        total_main_loss, total_std_loss, total_cov_loss = 0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            if vcreg_params:
                outputs, features = model(inputs, return_features=True)
                main_loss = criterion(outputs, labels)
                std_loss, cov_loss = calculate_vcreg_loss(features, vcreg_params)
                loss = main_loss + std_loss + cov_loss
                
                total_std_loss += std_loss.item() * inputs.size(0)
                total_cov_loss += cov_loss.item() * inputs.size(0)
            else:
                outputs = model(inputs)
                main_loss = criterion(outputs, labels)
                loss = main_loss
            
            loss.backward()
            
            if grad_clip_val:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)

            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            total_main_loss += main_loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

        train_loss = total_loss / total_samples
        train_main_loss = total_main_loss / total_samples
        train_std_loss = total_std_loss / total_samples
        train_cov_loss = total_cov_loss / total_samples
        train_acc = total_correct / total_samples
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # 驗證
        model.eval()
        total_val_loss, total_val_correct, total_val_samples = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs) # 驗證時不需要特徵
                loss = criterion(outputs, labels)
                
                total_val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val_samples += labels.size(0)
                total_val_correct += (predicted == labels).sum().item()

        val_loss = total_val_loss / total_val_samples
        val_acc = total_val_correct / total_val_samples
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 建立一個字典來結構化地記錄 Epoch 資訊
        epoch_log = {
            'epoch': epoch + 1,
            'total_epochs': epochs,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
        }
        
        # 如果是 VCReg 實驗，加入額外日誌
        if vcreg_params:
            epoch_log['train_main_loss'] = train_main_loss
            epoch_log['train_std_loss'] = train_std_loss
            epoch_log['train_cov_loss'] = train_cov_loss

        logging.info(f"Epoch {epoch+1}/{epochs} summary: {json.dumps(epoch_log)}")

        if scheduler:
            scheduler.step()

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            logging.info("Early stopping 觸發！")
            break

    # 8. 載入最佳模型並儲存
    logging.info(f"從 '{early_stopping.path}' 載入驗證集上表現最佳的模型...")
    model.load_state_dict(torch.load(early_stopping.path, weights_only=True))
    
    model_save_name = f"{experiment_name}.pth"
    model_save_path = os.path.join(cfg.MODEL_SAVE_DIR, model_save_name)
    os.makedirs(cfg.MODEL_SAVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"最佳模型已儲存至: {model_save_path}")

    # --- 新增步驟：儲存與模型對應的實驗設定檔 (.json) ---
    config_save_path = os.path.join(cfg.MODEL_SAVE_DIR, f"{experiment_name}.json")
    # 為了儲存，需要將設定中的 transform 物件轉為字串描述
    if 'transform' in exp_config and not isinstance(exp_config['transform'], str):
         exp_config['transform'] = "CustomTransform" # 或其他描述
    with open(config_save_path, 'w', encoding='utf-8') as f:
        json.dump(exp_config, f, indent=4, ensure_ascii=False, default=lambda o: '<not serializable>')
    logging.info(f"實驗設定已儲存至: {config_save_path}")
    
    # 繪製並儲存歷史圖表
    plot_acc_path = os.path.join(cfg.PLOT_SAVE_DIR, f"{experiment_name}_accuracy.png")
    plot_loss_path = os.path.join(cfg.PLOT_SAVE_DIR, f"{experiment_name}_loss.png")
    os.makedirs(cfg.PLOT_SAVE_DIR, exist_ok=True)
    plot_accuracy(history, plot_acc_path)
    plot_loss(history, plot_loss_path)

    logging.info(f"--- 實驗 {experiment_name} 完成 ---")


def main():
    parser = argparse.ArgumentParser(description="執行 CNN 音訊分類模型的訓練實驗。")
    parser.add_argument(
        '-e', '--experiment', 
        type=str, 
        required=True,
        help='要運行的實驗名稱 (定義於 cnn_config.py)。',
        choices=list(cfg.EXPERIMENTS.keys())
    )
    args = parser.parse_args()

    # 為每個實驗建立一個固定的存檔路徑
    checkpoint_dir = cfg.MODEL_SAVE_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{args.experiment}.pt")
    
    run_experiment(args.experiment, checkpoint_path)

if __name__ == '__main__':
    main()