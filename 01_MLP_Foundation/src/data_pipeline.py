import torch
from torch.utils.data import DataLoader
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import glob

# 導入 MLP 版本的模組
from .utils import AudioDataset
from .helpers import calculate_scaler_stats
from . import mlp_config as cfg

def collate_fn(batch):
    """
    自訂的 collate_fn，用於過濾掉無效的樣本。
    AudioDataset 在遇到問題時會返回 label 為 -1 的樣本。
    """
    batch = [item for item in batch if item[1] != -1]
    if not batch:
        return None, None
    return torch.utils.data.dataloader.default_collate(batch)

def get_dataloaders(config: dict, force_recalc_scaler: bool = False):
    """
    根據統一的設定物件，返回對應的訓練和驗證 DataLoader。
    這是為 MLP 基準模型設計的簡化版本。
    """
    print(f"--- 正在為 MLP 實驗 '{config['experiment_name']}' 準備資料 ---")
    
    data_config = config['data']
    train_config = config['train']
    audio_config = config['audio']
    
    batch_size = train_config.get('batch_size', 64)
    
    pipeline_name = next(iter(cfg.DATA_PIPELINES))
    pipeline_config = cfg.DATA_PIPELINES[pipeline_name]

    print(f"使用的資料管道: {pipeline_name}")

    train_csv_path = data_config['train_csv_path']
    # 音檔目錄從 CSV 路徑推導得出
    voice_data_root = os.path.dirname(os.path.dirname(train_csv_path)) 
    
    train_data_path = os.path.join(voice_data_root, 'voice_dataset', 'train_set')
    scaler_path = os.path.join(data_config['processed_data_dir'], f'scaler_{pipeline_name}.pt')


    df = pd.read_csv(train_csv_path)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    if os.path.exists(scaler_path) and not force_recalc_scaler:
        print(f"從 '{scaler_path}' 載入已有的 scaler...")
        scaler = torch.load(scaler_path)
    else:
        print(f"為管道 '{pipeline_name}' 計算新的 scaler...")
        
        temp_dataset_for_scaler = AudioDataset(
            df, train_data_path, scaler=None, 
            pipeline_config=pipeline_config
        )
        scaler = calculate_scaler_stats(temp_dataset_for_scaler)
        
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        torch.save(scaler, scaler_path)
        print(f"新的 scaler 已計算並儲存至 '{scaler_path}'")

    train_dataset = AudioDataset(
        train_df, train_data_path, scaler=scaler,
        pipeline_config=pipeline_config
    )
    val_dataset = AudioDataset(
        val_df, train_data_path, scaler=scaler,
        pipeline_config=pipeline_config
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_fn)
    
    input_dim = audio_config['n_mels'] * data_config['target_len']
    config['model']['input_dim'] = input_dim

    data_dims = {
        'input_dim': input_dim,
        'output_dim': audio_config['num_classes']
    }
    
    print(f"資料準備完成。展平後的輸入維度: {data_dims['input_dim']}")
    return train_loader, val_loader, data_dims 