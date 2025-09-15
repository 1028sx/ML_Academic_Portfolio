import torch
from torch.utils.data import DataLoader
import pandas as pd
import os
from sklearn.model_selection import train_test_split

from .utils import AudioDataset
from .helpers import calculate_scaler_stats
from . import config as cfg

def get_dataloaders(experiment_name: str, force_recalc_scaler: bool = False):
    """獲取資料加載器"""
    print(f"準備資料: {experiment_name}")

    # 獲取實驗設定
    config = cfg.EXPERIMENTS[experiment_name]
    data_pipeline_config = config['data_pipeline']
    batch_size = config.get('batch_size', 64)
    pipeline_name = "default"
    for name, config_dict in cfg.DATA_PIPELINES.items():
        if config_dict == data_pipeline_config:
            pipeline_name = name
            break

    # 路徑設定
    train_csv_path = cfg.TRAIN_CSV_PATH
    train_data_path = cfg.TRAIN_SET_DIR
    scaler_path = os.path.join(cfg.PROCESSED_DATA_DIR, f'scaler_{pipeline_name}.pt')

    # 載入和分割數據
    df = pd.read_csv(train_csv_path)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    # Scaler 計算或載入
    if os.path.exists(scaler_path) and not force_recalc_scaler:
        print(f"從 '{scaler_path}' 載入已有的 scaler...")
        scaler = torch.load(scaler_path, weights_only=False)
    else:
        if force_recalc_scaler:
            print("強制重新計算 scaler...")
        else:
            print(f"未找到 scaler for pipeline '{pipeline_name}'，正在計算新的 scaler...")
        temp_dataset_for_scaler = AudioDataset(
            df, train_data_path, scaler=None, is_train=False,
            pipeline_config=data_pipeline_config
        )
        scaler = calculate_scaler_stats(temp_dataset_for_scaler)

        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        torch.save(scaler, scaler_path)
        print(f"新的 scaler 已計算並儲存至 '{scaler_path}'")

    # 創建 Datasets
    print(f"使用資料管道: {data_pipeline_config}")
    train_dataset = AudioDataset(
        train_df, train_data_path, scaler=scaler, is_train=True,
        pipeline_config=data_pipeline_config
    )
    val_dataset = AudioDataset(
        val_df, train_data_path, scaler=scaler, is_train=False,
        pipeline_config=data_pipeline_config
    )

    # 創建 DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # 獲取並返回資料維度
    sample_spec, _ = train_dataset[0]
    data_dims = {
        'input_h': sample_spec.shape[1],
        'input_w': sample_spec.shape[2],
        'output_dim': cfg.AUDIO['num_classes']
    }

    print(f"資料準備完成。輸入維度 (H, W): ({data_dims['input_h']}, {data_dims['input_w']})")
    return train_loader, val_loader, data_dims