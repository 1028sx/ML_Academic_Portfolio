import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import numpy as np
import os

# 導入 MLP 版本的輔助函式
from .helpers import (
    load_audio, get_mel_spectrogram
)
# 導入 MLP 版本的設定
from . import mlp_config as cfg

class AudioDataset(Dataset):
    """
    客製化 PyTorch Dataset，用於處理音訊檔案。
    這是為 MLP 基準模型設計的簡化版本。
    """
    def __init__(self, dataframe, data_dir, scaler, pipeline_config):
        self.df = dataframe
        self.data_dir = data_dir
        self.scaler = scaler
        
        # 從 pipeline_config 解構出所有需要的參數
        self.pipeline_config = pipeline_config
        self.target_len = pipeline_config.get('target_len', 215)

        # 音訊參數，直接從 MLP 設定檔讀取
        self.sample_rate = cfg.AUDIO['sample_rate']
        self.n_fft = cfg.AUDIO['n_fft']
        self.hop_length = cfg.AUDIO['hop_length']
        self.n_mels = cfg.AUDIO['n_mels']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.data_dir, row['wav'])
        label = row['label']

        # 1. 載入音訊
        waveform = load_audio(file_path, self.sample_rate)
        if waveform is None or waveform.shape[1] < self.n_fft:
            # 如果音訊載入失敗或長度不足以計算一幀 FFT，則返回一個零張量和無效標籤
            # 這需要後續在 Dataloader 中過濾掉
            print(f"警告：檔案 {file_path} 長度不足或載入失敗，將跳過。")
            return torch.zeros(1, self.n_mels, self.target_len), torch.tensor(-1, dtype=torch.long)

        # 2. 轉換為梅爾頻譜圖
        mel_spec = get_mel_spectrogram(
            waveform, 
            sample_rate=self.sample_rate, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            n_mels=self.n_mels
        )
        
        log_mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)

        # 3. 標準化長度
        if log_mel_spec.shape[2] < self.target_len:
            pad_size = self.target_len - log_mel_spec.shape[2]
            log_mel_spec = torch.nn.functional.pad(log_mel_spec, (0, pad_size))
        else:
            log_mel_spec = log_mel_spec[:, :, :self.target_len]
        
        # 4. 標準化
        if self.scaler:
            mean = self.scaler['mean']
            std = self.scaler['std']
            # .squeeze(0) 移除 channel 維度, 因為 MLP 不需要
            standardized_spec = (log_mel_spec.squeeze(0) - mean) / std
        else: 
            standardized_spec = log_mel_spec.squeeze(0)

        # MLP 模型在 forward pass 中自行展平，這裡返回 2D 張量 (n_mels, target_len)
        return standardized_spec, torch.tensor(label, dtype=torch.long)

class EarlyStopping:
    """早停法以避免過擬合"""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model to {self.path} ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss 