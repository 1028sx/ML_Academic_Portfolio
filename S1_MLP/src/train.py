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

# 本地模組
from .models import get_model
from .data_pipeline import get_dataloaders
from .helpers import plot_accuracy, plot_loss
from . import config as cfg

# 共享工具
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from shared_utils import EarlyStopping

# 全域變數
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def get_optimizer(model, config):
    """獲取優化器"""
    optimizer_type = config.get('optimizer', 'AdamW')
    lr = config.get('lr', 1e-4)
    wd = config.get('wd', 1e-3)

    if optimizer_type == 'Adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if optimizer_type == 'AdamW':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

def run_experiment(experiment_name: str):
    """加載配置，執行訓練和評估流程。"""
    # 獲取設定
    config = cfg.get_config(experiment_name)


    # 準備資料
    train_loader, val_loader, data_dims = get_dataloaders(config)

    # 初始化模型
    model_config = {**config['model'], **data_dims}
    model = get_model(model_config)
    model.to(device)

    # 優化器和損失函數
    train_params = config['train']
    optimizer = get_optimizer(model, train_params)
    criterion = nn.CrossEntropyLoss()

    # Early Stopping 和存檔路徑
    checkpoint_name = f"mlp_{experiment_name}.pth"
    checkpoint_path = os.path.join(config['model_save_dir'], checkpoint_name)
    early_stopping = EarlyStopping(patience=train_params['patience'], verbose=True, path=checkpoint_path)

    # 訓練迴圈
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

        print(f'第 {epoch+1}/{train_params["epochs"]} 輪 | '
              f'訓練損失: {epoch_train_loss:.4f}, 訓練準確率: {epoch_train_acc:.4f} | '
              f'驗證損失: {epoch_val_loss:.4f}, 驗證準確率: {epoch_val_acc:.4f}')

        early_stopping(epoch_val_loss, model)
        if early_stopping.early_stop:
            print("提前停止訓練")
            break

    print("--- 訓練完成 ---")

    # 載入最佳模型
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    print(f"已從 '{checkpoint_path}' 載入最佳模型")

    # 儲存訓練歷史圖
    plot_acc_path = os.path.join(config['plot_save_dir'], f"mlp_{experiment_name}_accuracy.png")
    plot_loss_path = os.path.join(config['plot_save_dir'], f"mlp_{experiment_name}_loss.png")
    plot_accuracy(history, plot_acc_path)
    plot_loss(history, plot_loss_path)

    # 儲存最終的設定檔
    config_save_path = os.path.join(config['model_save_dir'], f"mlp_{experiment_name}.json")
    with open(config_save_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print(f"儲存至: {config_save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'experiment_name',
        type=str,
        choices=list(cfg.EXPERIMENTS.keys())
    )
    args = parser.parse_args()
    run_experiment(args.experiment_name)

if __name__ == '__main__':
    main()