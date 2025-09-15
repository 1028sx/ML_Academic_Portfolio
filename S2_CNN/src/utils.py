import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchaudio
import numpy as np
import random
import torchaudio.transforms as T
from .helpers import (
    apply_spec_augment,
    add_background_noise, shift_time, change_pitch_and_speed
)
from . import config as cfg

# 共享工具導入
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from shared_utils import load_audio, get_mel_spectrogram

# 用於儲存已經記錄過的檔案路徑，避免在多個 epoch 中重複記錄
_logged_short_files = set()

def setup_seed(seed):
    """設定隨機種子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AudioDataset(Dataset):
    """客製化 PyTorch Dataset，用於處理音訊檔案"""
    def __init__(self, dataframe, data_dir, scaler, is_train, pipeline_config):
        self.df = dataframe
        self.data_dir = data_dir
        self.scaler = scaler
        self.is_train = is_train

        # 解構需要的參數
        self.pipeline_config = pipeline_config
        self.use_vad = pipeline_config.get('use_vad', False)
        self.vad_config = pipeline_config.get('vad_config', {})
        self.target_len = pipeline_config.get('target_len', 215)
        self.augmentation_mode = pipeline_config.get('augmentation_mode', None) if self.is_train else None

        # 優先從 pipeline_config 讀取音訊參數
        self.sample_rate = pipeline_config.get('sample_rate', cfg.AUDIO['sample_rate'])
        self.n_fft = pipeline_config.get('n_fft', cfg.AUDIO['n_fft'])
        self.hop_length = pipeline_config.get('hop_length', cfg.AUDIO['hop_length'])
        self.n_mels = pipeline_config.get('n_mels', cfg.AUDIO['n_mels'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.data_dir, row['wav'])
        label = row['label']

        # 載入音訊
        waveform = load_audio(file_path, self.sample_rate)

        # VAD 處理
        if self.use_vad:
            waveform = torchaudio.functional.vad(waveform, sample_rate=self.sample_rate)
        # 檢測並重新補齊長度
        if waveform.shape[1] < self.n_fft:
            pad_size = self.n_fft - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))

        # 音訊增強
        if self.is_train and self.augmentation_mode == 'gentle_audio':
            choice = np.random.rand()
            if choice < 0.3:
                waveform = add_background_noise(waveform, self.sample_rate)
            elif choice < 0.6:
                waveform = shift_time(waveform, self.sample_rate, shift_limit=0.2)
            else:
                waveform = change_pitch_and_speed(
                    waveform,
                    self.sample_rate,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    pitch_factor=np.random.uniform(-2, 2)
                )

        # 檢測並重新補齊長度
        if waveform.shape[1] < self.n_fft:
            pad_size = self.n_fft - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))

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
        actual_len = log_mel_spec.shape[2]
        if actual_len < self.target_len:
            if file_path not in _logged_short_files:
                # 檢測長度不足以避免崩潰
                print(f"檔案長度不足: {file_path} (實際長度: {actual_len}, 目標長度: {self.target_len})")
                _logged_short_files.add(file_path)

        if log_mel_spec.shape[2] < self.target_len:
            pad_size = self.target_len - log_mel_spec.shape[2]
            log_mel_spec = torch.nn.functional.pad(log_mel_spec, (0, pad_size))
        else:
            log_mel_spec = log_mel_spec[:, :, :self.target_len]

        # 頻譜增強
        if self.is_train and self.augmentation_mode == 'spec_augment':
            log_mel_spec = apply_spec_augment(log_mel_spec)

        # 標準化
        if self.scaler:
            mean = self.scaler['mean']
            std = self.scaler['std']
            log_mel_spec_np = log_mel_spec.squeeze(0).numpy()
            standardized_spec_np = (log_mel_spec_np - mean) / std
            standardized_spec_tensor = torch.from_numpy(standardized_spec_np).float()
        else:
            standardized_spec_tensor = log_mel_spec.squeeze(0).float()

        # 回傳前驗證頻譜圖的寬度是否符合預期
        if standardized_spec_tensor.shape[1] != self.target_len:
            raise ValueError(
                f"尺寸驗證失敗！檔案 '{file_path}' 的頻譜圖寬度為 "
                f"{standardized_spec_tensor.shape[1]}，但期望的 target_len 為 {self.target_len}。"
            )

        return standardized_spec_tensor.unsqueeze(0), torch.tensor(label, dtype=torch.long)