import optuna
import torch
import torch.optim as optim
import torch.nn as nn
import copy
import time
import os
import argparse
import logging
from tqdm import tqdm

# --- FIX: 使用新的設定中心 ---
# 移除舊的 get_config 和 get_path，直接導入完整的 config 模組。
# 這確保了我們總是能從單一的真相來源 (Single Source of Truth) 獲取設定。
from . import config as cfg
from .data_pipeline import get_dataloaders
from .models import get_model
from .utils import setup_seed

def objective(trial, base_config, logger):
    """
    Optuna 的 objective 函式，用於單次超參數組合的訓練與評估。
    """
    # 複製一份基礎設定，避免污染全域設定
    cfg = copy.deepcopy(base_config)

    # --- FIX: 適應新的設定結構 ---
    # 從 cfg.EXPERIMENTS 讀取到的設定，其訓練參數 (如 lr, wd, epochs) 是在頂層的，
    # 而不是像舊的 get_config 那樣被包在一個 'train' 鍵裡面。
    # 我們直接在頂層讀寫這些參數。
    cfg['lr'] = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    cfg['wd'] = trial.suggest_float('wd', 1e-5, 1e-2, log=True)
    
    # 優化 Dropout 的 p 值
    # 我們需要將這個值放入 model_blueprint 中，以便 get_model 能夠讀取到
    dropout_p = trial.suggest_float('dropout_p', 0.1, 0.5)
    cfg['model_blueprint']['dropout_p'] = dropout_p
    
    # 新增：為此試驗記錄超參數
    logger.info(f"--- Trial {trial.number} ---")
    logger.info(f"  - lr: {cfg['lr']:.6f}")
    logger.info(f"  - wd: {cfg['wd']:.6f}")
    logger.info(f"  - dropout_p: {dropout_p:.4f}")

    # 確保每次 trial 的隨機種子不同但可控，以便複現
    setup_seed(trial.number)

    # 設置裝置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 準備資料
    # 使用設定檔中定義的資料集
    train_loader, val_loader, data_dims = get_dataloaders(cfg['experiment_name'])

    # 3. 建立模型
    # --- FIX: 使用重構後的 get_model ---
    model = get_model(cfg['model_blueprint'], data_dims)
    model.to(device)

    # 4. 定義優化器與損失函數
    # 同樣，直接從頂層讀取 lr 和 wd
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0

    # 5. 訓練迴圈
    for epoch in range(cfg['epochs']):
        model.train()
        # 新增：用 tqdm 包裝 train_loader 以顯示進度條
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
        # 新增：用 tqdm 包裝 val_loader 以顯示進度條
        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg['epochs']} [Val]", leave=False)
        with torch.no_grad():
            for features, labels in pbar_val:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        logger.info(f"Epoch {epoch+1}/{cfg['epochs']}, Validation Accuracy: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        # Optuna 剪枝 (Pruning) 機制，可以提早中止沒有希望的 trial
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # 6. 返回要優化的目標值
    return best_val_acc

def main_tuning():
    """
    主函式，用於啟動 Optuna 超參數搜索。
    """
    # --- NEW: 命令列參數解析 ---
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Tuning Script")
    parser.add_argument(
        "-n", "--n-trials", 
        type=int, 
        default=50, 
        help="Number of trials to run."
    )
    # 新增：為實驗名稱增加參數
    parser.add_argument(
        "-e", "--experiment",
        type=str,
        default="V25_Heavy_Regularization",
        help="Name of the experiment configuration to use as a base."
    )
    args = parser.parse_args()
    
    # --- 新增：設定記錄 ---
    log_dir = cfg.LOG_SAVE_DIR
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"tuning_{args.experiment}_{timestamp}.log")
    
    # 配置根記錄器
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler() # Also print to console
                        ])
    logger = logging.getLogger(__name__)
    
    logger.info(f"日誌檔案位於: {log_file}")
    logger.info(f"將以 '{args.experiment}' 的設定為基礎，執行 {args.n_trials} 次試驗。")
    # --- 結束記錄設定 ---

    # 載入基礎設定檔
    try:
        # 新增：使用來自 CLI 參數的實驗名稱
        base_config = cfg.EXPERIMENTS[args.experiment]
    except KeyError:
        logger.error(f"錯誤: 在 config.py 中找不到名為 '{args.experiment}' 的實驗。")
        return

    # 將實驗名稱加入設定中，以便 get_dataloaders 等函式使用
    base_config['experiment_name'] = args.experiment

    # 為了節省時間，在超參數搜索時可以減少 epoch 數量
    # 我們將 epochs 縮短為 15，以加速搜尋過程
    base_config['epochs'] = 15 
    logger.info(f"為加速搜尋，暫時將 epochs 數量設為: {base_config['epochs']}")

    # 創建 Optuna study
    # pruner 可以在早期階段就砍掉沒有潛力的試驗，節省大量時間
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction='maximize', pruner=pruner)
    
    # 啟動優化
    # 使用 lambda 將 base_config 和 logger 傳遞給 objective 函式
    study.optimize(lambda trial: objective(trial, base_config, logger), n_trials=args.n_trials)

    # 輸出最佳結果
    logger.info("=" * 30)
    logger.info("超參數搜尋完成")
    logger.info(f"Number of finished trials: {len(study.trials)}")
    logger.info("Best trial:")
    trial = study.best_trial

    logger.info(f"  Value (Best Validation Accuracy): {trial.value:.4f}%")
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")
    logger.info("=" * 30)

if __name__ == '__main__':
    main_tuning() 