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


def get_optimizer(model, config):
    """獲取優化器"""
    optimizer_type = config.get('optimizer', 'AdamW')
    lr = config.get('lr', 1e-4)
    wd = config.get('wd', 1e-5)

    if optimizer_type == 'Adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if optimizer_type == 'AdamW':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

def get_scheduler(optimizer, exp_config, train_loader):
    """獲取學習率排程器"""
    scheduler_config = exp_config.get('scheduler')
    if not scheduler_config:
        return None

    scheduler_name = scheduler_config.get('name')
    # 創建 OneCycleLR 排程器，如果類型不匹配則返回 None
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
        return None

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device):
    """訓練一輪"""
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

        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

    train_loss = total_loss / total_samples
    train_acc = total_correct / total_samples
    return train_loss, train_acc

def validate_one_epoch(model, val_loader, criterion, device):
    """驗證一輪"""
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
    """執行實驗"""
    exp_config = cfg.get_config(experiment_name)


    print(f"--- 開始實驗: {experiment_name} ---")
    print(f"使用設備: {exp_config['device']}")

    # 準備資料
    exp_config['data_dir'] = cfg.DATA_DIR
    exp_config['csv_path'] = cfg.TRAIN_CSV_PATH
    train_loader, val_loader = get_dataloaders(exp_config)

    # 初始化模型
    model = get_model(exp_config['model_blueprint'], exp_config['audio_settings'])
    model.to(exp_config['device'])
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型 '{exp_config['model_blueprint']['model_class']}' 初始化成功")

    # 優化器、損失函式和排程器
    optimizer = get_optimizer(model, exp_config)

    # 標籤平滑
    label_smoothing = exp_config.get('label_smoothing', 0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if label_smoothing > 0.0:
        print(f"使用標籤平滑，因子為 {label_smoothing}")

    scheduler = get_scheduler(optimizer, exp_config, train_loader)

    # early_stopping
    checkpoint_name = f"checkpoint_{experiment_name}.pth"
    checkpoint_path = os.path.join(cfg.MODEL_SAVE_DIR, checkpoint_name)
    os.makedirs(cfg.MODEL_SAVE_DIR, exist_ok=True)

    patience = exp_config.get('patience', 15)
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=checkpoint_path)

    # 訓練迴圈
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    epochs = exp_config.get('epochs', 100)
    device = exp_config['device']

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f'第 {epoch+1}/{epochs} 輪 | '
              f'訓練損失: {train_loss:.4f}, 訓練準確率: {train_acc:.4f} | '
              f'驗證損失: {val_loss:.4f}, 驗證準確率: {val_acc:.4f}')

        if scheduler and not isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
            scheduler.step()

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("提前停止訓練")
            break

    # 儲存最終模型和結果
    print(f"已從 '{early_stopping.path}' 載入最佳模型")
    model.load_state_dict(torch.load(early_stopping.path, weights_only=True))

    # 儲存最終模型狀態
    final_model_name = f"{experiment_name}.pth"
    final_model_path = os.path.join(cfg.MODEL_SAVE_DIR, final_model_name)
    torch.save(model.state_dict(), final_model_path)
    print(f"最終最佳模型已儲存至: {final_model_path}")

    # 儲存實驗配置
    config_save_path = os.path.join(cfg.MODEL_SAVE_DIR, f"{experiment_name}.json")
    with open(config_save_path, 'w', encoding='utf-8') as f:
        json.dump(exp_config, f, indent=4, default=str)
    print(f"實驗配置已儲存至: {config_save_path}")

    # 儲存圖表
    plot_accuracy(history, experiment_name, cfg.PLOT_SAVE_DIR)
    plot_loss(history, experiment_name, cfg.PLOT_SAVE_DIR)
    print(f"準確率和損失圖表已儲存至: {cfg.PLOT_SAVE_DIR}")

    print("--- 訓練完成 ---")

def main():
    """解析參數並執行指定的實驗。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=list(cfg.EXPERIMENTS.keys())
    )
    args = parser.parse_args()

    run_experiment(args.experiment)

if __name__ == '__main__':
    main()