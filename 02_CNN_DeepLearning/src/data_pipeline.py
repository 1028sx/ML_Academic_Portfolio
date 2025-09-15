import torch
from torch.utils.data import DataLoader
import pandas as pd
import os
from sklearn.model_selection import train_test_split

from utils import AudioDataset
from helpers import calculate_scaler_stats
# 導入新的設定中心
import config as cfg

def get_dataloaders(experiment_name: str, force_recalc_scaler: bool = False):
    """
    根據指定的實驗名稱，從設定中心加載配置，
    並返回對應的訓練和驗證 DataLoader。

    Args:
        experiment_name (str): 在 config.py 中定義的實驗名稱。
        force_recalc_scaler (bool): 是否強制重新計算 scaler。

    Returns:
        tuple: (train_loader, val_loader, data_dims)
    """
    print(f"--- 正在為實驗 '{experiment_name}' 準備資料 ---")
    
    # 1. 從設定中心獲取實驗設定
    try:
        exp_config = cfg.EXPERIMENTS[experiment_name]
    except KeyError:
        print(f"錯誤: 在 config.py 中找不到名為 '{experiment_name}' 的實驗。")
        raise
        
    data_pipeline_config = exp_config['data_pipeline']
    batch_size = exp_config.get('batch_size', 64)

    # 新增：找出管道名稱以建立唯一的標準化器檔名
    pipeline_name = "default" # 備用名稱
    for name, config_dict in cfg.DATA_PIPELINES.items():
        if config_dict == data_pipeline_config:
            pipeline_name = name
            break
    print(f"找到對應的資料管道名稱: {pipeline_name}")

    # 2. 路徑設定
    train_csv_path = cfg.TRAIN_CSV_PATH
    # BUG FIX: 強制使用精確的 train_set 路徑，避免潛在的路徑拼接錯誤
    train_data_path = cfg.TRAIN_SET_DIR 
    # 新增：為每個管道使用唯一的標準化器路徑
    scaler_path = os.path.join(cfg.PROCESSED_DATA_DIR, f'scaler_{pipeline_name}.pt')

    # 3. 載入和分割數據
    df = pd.read_csv(train_csv_path)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    # 4. Scaler 計算或載入 (MODIFIED LOGIC)
    if os.path.exists(scaler_path) and not force_recalc_scaler:
        print(f"從 '{scaler_path}' 載入已有的 scaler...")
        scaler = torch.load(scaler_path, weights_only=False)
    else:
        if force_recalc_scaler:
            print("強制重新計算 scaler...")
        else:
            print(f"未找到 scaler for pipeline '{pipeline_name}'，正在計算新的 scaler...")
        
        # 新增：使用*正確的*管道配置進行計算，而非硬編碼
        # 用於標準化器計算的數據集不應使用增強 (is_train=False)
        # 我們使用整個數據集 (df) 來計算標準化器統計以獲得更穩定的結果
        temp_dataset_for_scaler = AudioDataset(
            df, train_data_path, scaler=None, is_train=False, 
            pipeline_config=data_pipeline_config
        )
        scaler = calculate_scaler_stats(temp_dataset_for_scaler)
        
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        torch.save(scaler, scaler_path)
        print(f"新的 scaler 已計算並儲存至 '{scaler_path}'")

    # 5. 創建 Datasets (根據指定的實驗設定)
    print(f"使用資料管道: {data_pipeline_config}")
    train_dataset = AudioDataset(
        train_df, train_data_path, scaler=scaler, is_train=True,
        pipeline_config=data_pipeline_config
    )
    # 驗證集不應使用增強，我們透過 is_train=False 在 Dataset 內部控制
    val_dataset = AudioDataset(
        val_df, train_data_path, scaler=scaler, is_train=False,
        pipeline_config=data_pipeline_config
    )

    # 6. 創建 DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 7. 獲取並返回資料維度
    sample_spec, _ = train_dataset[0]
    data_dims = {
        'input_h': sample_spec.shape[1],
        'input_w': sample_spec.shape[2],
        'output_dim': cfg.AUDIO['num_classes']
    }
    
    print(f"資料準備完成。輸入維度 (H, W): ({data_dims['input_h']}, {data_dims['input_w']})")
    return train_loader, val_loader, data_dims

# 主執行區塊 (用於獨立測試)
if __name__ == '__main__':
    print("正在執行 data_pipeline.py 以進行獨立測試...")
    # 測試一個標準實驗和一個VAD實驗
    for test_experiment in ['V7_CNN_Advanced_Benchmark', 'V15_Advanced_VAD']:
        try:
            train_loader, val_loader, dims = get_dataloaders(experiment_name=test_experiment)
            
            print(f"\n--- 測試資訊 ({test_experiment}) ---")
            print(f"資料維度: {dims}")
            
            train_features, train_labels = next(iter(train_loader))
            print(f"Train DataLoader - 一個批次的特徵維度: {train_features.size()}")
            print(f"Train DataLoader - 一個批次的標籤維度: {train_labels.size()}")
            
            val_features, val_labels = next(iter(val_loader))
            print(f"Validation DataLoader - 一個批次的特徵維度: {val_features.size()}")
            print(f"Validation DataLoader - 一個批次的標籤維度: {val_labels.size()}")
            print("-" * 40)
        except Exception as e:
            print(f"測試實驗 '{test_experiment}' 失敗: {e}")

    print("\n測試完成。") 