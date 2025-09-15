import optuna
import torch
import torch.optim as optim
import torch.nn as nn
import copy
import time
import os
import argparse
from tqdm import tqdm

from . import config as cfg
from .data_pipeline import get_dataloaders
from .models import get_model
from .utils import setup_seed

def objective(trial, base_config):
    """Optuna 的 objective 函式，用於單次訓練與評估。 """
    # 避免污染全域設定
    cfg = copy.deepcopy(base_config)
    cfg['lr'] = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    cfg['wd'] = trial.suggest_float('wd', 1e-5, 1e-2, log=True)

    # 優化 Dropout 的 p 值
    dropout_p = trial.suggest_float('dropout_p', 0.1, 0.5)
    cfg['model_blueprint']['dropout_p'] = dropout_p

    # 為此試驗記錄超參數
    print(f"--- 試驗 {trial.number} ---")
    print(f"  - lr: {cfg['lr']:.6f}")
    print(f"  - wd: {cfg['wd']:.6f}")
    print(f"  - dropout_p: {dropout_p:.4f}")

    # 確保每次 trial 的隨機種子不同但可控，以便複現
    setup_seed(trial.number)

    # 設置裝置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 準備資料
    train_loader, val_loader, data_dims = get_dataloaders(cfg['experiment_name'])

    # 建立模型
    model_config = {**cfg['model_blueprint'], **data_dims}
    model = get_model(model_config)
    model.to(device)

    # 優化器與損失函數
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    # 訓練迴圈
    for epoch in range(cfg['epochs']):
        model.train()
        # 用 tqdm 包裝 train_loader 以顯示進度條
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']} [Train]", leave=False)
        for features, labels in pbar_train:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 驗證迴圈
        model.eval()
        val_correct = 0
        val_total = 0
        # 同上的進度條顯示
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']} [Val]", leave=False)
        with torch.no_grad():
            for features, labels in pbar_val:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        print(f"第 {epoch+1}/{cfg['epochs']} 輪, 驗證準確率: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        # Optuna 剪枝
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # 返回要優化的目標值
    return best_val_acc

def main_tuning():
    """啟動 Optuna 超參數搜索。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--n-trials",
        type=int,
        default=50
    )
    parser.add_argument(
        "-e", "--experiment",
        type=str,
        default="V25_Heavy_Regularization"
    )
    args = parser.parse_args()

    # 設定記錄
    log_dir = cfg.LOG_SAVE_DIR
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"tuning_{args.experiment}_{timestamp}.log")


    print(f"日誌檔案位於: {log_file}")
    print(f"將以 '{args.experiment}' 的設定為基礎，執行 {args.n_trials} 次試驗。")
    # 結束記錄設定

    # 載入基礎設定檔
    base_config = cfg.EXPERIMENTS[args.experiment]
    base_config['experiment_name'] = args.experiment

    # 在超參數搜索時減少 epoch 數量
    base_config['epochs'] = 15
    print(f"為加速搜尋，暫時將 epochs 數量設為: {base_config['epochs']}")

    # 創建 Optuna study剪枝器
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction='maximize', pruner=pruner)

    # 啟動優化
    study.optimize(lambda trial: objective(trial, base_config), n_trials=args.n_trials)

    # 輸出最佳結果
    print("=" * 30)
    print("超參數搜尋完成")
    print(f"完成試驗數量: {len(study.trials)}")
    print("最佳試驗:")
    trial = study.best_trial

    print(f"  值 (最佳驗證準確率): {trial.value:.4f}%")
    print("  參數: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    print("=" * 30)

if __name__ == '__main__':
    main_tuning()