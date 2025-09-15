# 標準庫
import os
import argparse
import json
from datetime import datetime

# 第三方庫
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 本地模組
from . import config as cfg
from .models import get_model
from .data_pipeline import get_dataloaders
from .helpers import plot_accuracy, plot_loss

# 共享工具
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from shared_utils import EarlyStopping

# 全域變數
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_vcreg_loss(x, vcreg_params):
    """計算 VCReg 損失"""
    lambda_weight = vcreg_params.get('lambda', 25.0)
    mu_weight = vcreg_params.get('mu', 25.0)

    # 變異數損失
    std_x = torch.sqrt(x.var(dim=0) + 1e-4)  # 防止 NaN
    std_loss = torch.mean(torch.relu(1 - std_x))

    # 協方差損失
    x = x - x.mean(dim=0)
    cov_x = (x.T @ x) / (x.size(0) - 1)
    cov_loss = (cov_x.pow(2).sum() - cov_x.diag().pow(2).sum()) / x.size(1)

    return std_loss * lambda_weight, cov_loss * mu_weight

def get_optimizer(model, config):
    """獲取優化器"""
    optimizer_config = config.get('optimizer', 'AdamW')

    if isinstance(optimizer_config, dict):
        # 新版嵌套格式 (V25後期實驗採用)
        optimizer_type = optimizer_config.get('type', 'AdamW')
        lr = optimizer_config.get('lr', 1e-4)
        wd = optimizer_config.get('wd', 1e-3)
    else:
        # 舊版扁平格式 (V16-V24早期實驗使用)
        optimizer_type = optimizer_config
        lr = config.get('lr', 1e-4)
        wd = config.get('wd', 1e-3)

    if optimizer_type == 'Adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if optimizer_type == 'AdamW':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

def get_scheduler(optimizer, config):
    """獲取學習率排程器"""
    scheduler_config = config.get('scheduler')
    if not scheduler_config:
        return None

    scheduler_type = scheduler_config['type']
    scheduler_params = scheduler_config.get('params', {})
    # 創建 CosineAnnealingLR 排程器
    if scheduler_type == 'CosineAnnealingLR':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
    else:
        return None

def run_experiment(experiment_name, checkpoint_path):
    """執行單一實驗的主函式"""
    # 獲取實驗設定
    config = cfg.EXPERIMENTS[experiment_name]

    print(f"--- 開始實驗: {experiment_name} ---")

    # 準備資料
    train_loader, val_loader, data_dims = get_dataloaders(experiment_name)

    # 初始化模型
    model_config = {**config['model_blueprint'], **data_dims}
    # dropout_rate 設定
    if 'dropout_rate' in config:
        model_config['dropout_p'] = config['dropout_rate']

    model = get_model(model_config)
    model.to(device)
    print(f"模型 '{model_config['model_class']}' 初始化成功")

    # 優化器、損失函數、排程器
    optimizer = get_optimizer(model, config)

    criterion = nn.CrossEntropyLoss()

    scheduler = get_scheduler(optimizer, config)

    # early_stopping機制
    patience = config.get('patience', 10)
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=checkpoint_path)

    # 訓練迴圈
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    epochs = config.get('epochs', 50)
    grad_clip_val = config.get('gradient_clip_val', None)

    # 檢查 VCReg 設定
    vcreg_params = next((reg for reg in config.get('regularizers', []) if reg.get('type') == 'vcreg'), None)

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
                outputs = model(inputs)  # 驗證時不需要特徵
                loss = criterion(outputs, labels)

                total_val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val_samples += labels.size(0)

                total_val_correct += (predicted == labels).sum().item()

        val_loss = total_val_loss / total_val_samples
        val_acc = total_val_correct / total_val_samples
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 輸出訓練進度
        print(f'第 {epoch+1}/{epochs} 輪 | '
              f'訓練損失: {train_loss:.4f}, 訓練準確率: {train_acc:.4f} | '
              f'驗證損失: {val_loss:.4f}, 驗證準確率: {val_acc:.4f}')


        if scheduler:
            scheduler.step()

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("提前停止訓練")
            break

    # 載入最佳模型並儲存
    print(f"已從 '{early_stopping.path}' 載入最佳模型")
    model.load_state_dict(torch.load(early_stopping.path, weights_only=True))

    model_save_name = f"{experiment_name}.pth"
    model_save_path = os.path.join(cfg.MODEL_SAVE_DIR, model_save_name)
    os.makedirs(cfg.MODEL_SAVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"儲存至: {model_save_path}")

    # 儲存實驗設定檔
    config_save_path = os.path.join(cfg.MODEL_SAVE_DIR, f"{experiment_name}.json")
    # 處理無法序列化的物件
    if 'transform' in config and not isinstance(config['transform'], str):
         config['transform'] = "CustomTransform"
    with open(config_save_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False, default=lambda o: '<not serializable>')
    print(f"設定已儲存至: {config_save_path}")

    # 繪製並儲存圖表
    plot_acc_path = os.path.join(cfg.PLOT_SAVE_DIR, f"{experiment_name}_accuracy.png")
    plot_loss_path = os.path.join(cfg.PLOT_SAVE_DIR, f"{experiment_name}_loss.png")
    os.makedirs(cfg.PLOT_SAVE_DIR, exist_ok=True)
    plot_accuracy(history, plot_acc_path)
    plot_loss(history, plot_loss_path)

    print("--- 訓練完成 ---")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e', '--experiment',
        type=str,
        required=True,
        choices=list(cfg.EXPERIMENTS.keys())
    )
    args = parser.parse_args()

    # 建立存檔路徑
    checkpoint_dir = cfg.MODEL_SAVE_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{args.experiment}.pt")

    run_experiment(args.experiment, checkpoint_path)

if __name__ == '__main__':
    main()