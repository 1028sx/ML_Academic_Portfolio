import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import numpy as np
import os

# 導入設定
from . import config as cfg

# 共享工具導入
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from shared_utils import load_audio, get_mel_spectrogram

class AudioDataset(Dataset):
    # 超載PyTorch Dataset
    def __init__(self, dataframe, data_dir, scaler, pipeline_config):
        self.df = dataframe
        self.data_dir = data_dir
        self.scaler = scaler

        # 從 pipeline_config 解構參數
        self.pipeline_config = pipeline_config
        self.target_len = pipeline_config.get('target_len', 215)

        # 讀取音訊參數
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

        # 載入音訊
        waveform = load_audio(file_path, self.sample_rate)
        if waveform is None or waveform.shape[1] < self.n_fft:
            # 失敗則返回零張量和無效標籤
            print(f"警告：檔案 {file_path} 長度不足或載入失敗，將跳過。")
            return torch.zeros(1, self.n_mels, self.target_len), torch.tensor(-1, dtype=torch.long)

        # 轉換為梅爾頻譜圖
        mel_spec = get_mel_spectrogram(
            waveform,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )

        log_mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)

        # 標準化長度
        if log_mel_spec.shape[2] < self.target_len:
            pad_size = self.target_len - log_mel_spec.shape[2]
            log_mel_spec = torch.nn.functional.pad(log_mel_spec, (0, pad_size))
        else:
            log_mel_spec = log_mel_spec[:, :, :self.target_len]

        # 標準化
        if self.scaler:
            mean = self.scaler['mean']
            std = self.scaler['std']
            # MLP 不需要channel 維度
            standardized_spec = (log_mel_spec.squeeze(0) - mean) / std
        else:
            standardized_spec = log_mel_spec.squeeze(0)

        return standardized_spec, torch.tensor(label, dtype=torch.long)