import torch
from torch.utils.data import DataLoader
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import glob

# 導入 MLP 版本的模組
from .utils import AudioDataset
from .helpers import calculate_scaler_stats
from . import config as cfg

def collate_fn(batch):
    #過濾無效樣本(遇到問題返回 label-1)
    batch = [item for item in batch if item[1] != -1]
    if not batch:
        return None, None
    return torch.utils.data.default_collate(batch)

def get_dataloaders(config: dict, force_recalc_scaler: bool = False):
    data_config = config['data']
    train_config = config['train']
    audio_config = config['audio']

    batch_size = train_config.get('batch_size', 64)

    pipeline_name = next(iter(cfg.DATA_PIPELINES))
    pipeline_config = cfg.DATA_PIPELINES[pipeline_name]

    train_csv_path = data_config['train_csv_path']
    train_data_path = os.path.join(cfg.DATA_DIR, 'voice_dataset', 'train_set')
    scaler_path = os.path.join(data_config['processed_data_dir'], f'scaler_{pipeline_name}.pt')


    df = pd.read_csv(train_csv_path)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    if os.path.exists(scaler_path) and not force_recalc_scaler:
        print("載入已有的 scaler")
        scaler = torch.load(scaler_path)
    else:
        print("計算新的 scaler")

        temp_dataset_for_scaler = AudioDataset(
            df, train_data_path, scaler=None,
            pipeline_config=pipeline_config
        )
        scaler = calculate_scaler_stats(temp_dataset_for_scaler)

        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        torch.save(scaler, scaler_path)
        print(f"已儲存至 '{scaler_path}'")

    train_dataset = AudioDataset(
        train_df, train_data_path, scaler=scaler,
        pipeline_config=pipeline_config
    )
    val_dataset = AudioDataset(
        val_df, train_data_path, scaler=scaler,
        pipeline_config=pipeline_config
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)

    input_dim = audio_config['n_mels'] * data_config['target_len']
    config['model']['input_dim'] = input_dim

    data_dims = {
        'input_dim': input_dim,
        'output_dim': audio_config['num_classes']
    }

    print("資料準備完成")
    return train_loader, val_loader, data_dims