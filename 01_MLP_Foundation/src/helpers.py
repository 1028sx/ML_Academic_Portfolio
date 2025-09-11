import torch
import torchaudio
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import torchaudio.transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import logging

# 導入 AudioDataset，但只用於類型提示，避免循環導入
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from utils import AudioDataset

# ==============================================================================
#  音訊載入與基礎轉換 (Audio Loading & Basic Transforms)
# ==============================================================================

def load_audio(file_path: str, sample_rate: int) -> torch.Tensor:
    """載入音訊檔案並視需要進行重採樣。"""
    waveform, sr = torchaudio.load(file_path)
    if sr != sample_rate:
        resampler = T.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    return waveform

def get_mel_spectrogram(waveform: torch.Tensor, sample_rate: int, n_fft: int, hop_length: int, n_mels: int) -> torch.Tensor:
    """從波形計算梅爾頻譜圖。"""
    mel_spectrogram_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    return mel_spectrogram_transform(waveform)

# ==============================================================================
#  統計計算 (Statistics Calculation)
# ==============================================================================

def calculate_scaler_stats(dataset: 'AudioDataset'):
    """計算數據集的均值和標準差以進行標準化。"""
    # 使用 tqdm 顯示進度條
    print("正在計算 MLP 數據集的平均值與標準差...")
    # 創建一個 DataLoader 來迭代數據集
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 初始化
    mean = 0.
    std = 0.
    total_samples = 0

    for spec, _ in tqdm(loader):
        # 確保 spec 至少是 2D 的
        if spec.dim() >= 2:
            # 對於 MLP (2D Tensor: [batch, features])
            if spec.dim() == 2:
                batch_samples = spec.size(0)
                mean += spec.mean(dim=1).sum()
                std += spec.std(dim=1).sum()
                total_samples += batch_samples
            # 對於 CNN (4D Tensor: [batch, channels, height, width])
            elif spec.dim() == 4:
                batch_samples = spec.size(0)
                # 我們只關心頻率和時間維度
                mean += spec.mean([0, 2, 3]).sum()
                std += spec.std([0, 2, 3]).sum()
                total_samples += batch_samples
    
    if total_samples > 0:
        mean /= total_samples
        std /= total_samples
    else:
        mean = torch.tensor(0.0)
        std = torch.tensor(1.0)

    return {'mean': mean, 'std': std}

# ==============================================================================
#  繪圖函式 (Plotting Functions)
# ==============================================================================

def plot_accuracy(history, save_path):
    """繪製並儲存準確率歷史圖表。"""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs_range = range(1, len(history['train_acc']) + 1)

    ax.plot(epochs_range, history['train_acc'], '-o', label='Train Accuracy', markersize=4)
    ax.plot(epochs_range, history['val_acc'], '-o', label='Validation Accuracy', markersize=4)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Epochs")
    ax.set_title("Training and Validation Accuracy")
    ax.legend()
    ax.grid(True)
    
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    logging.info(f"準確率圖表已儲存至: {save_path}")

def plot_loss(history, save_path):
    """繪製並儲存損失歷史圖表。"""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs_range = range(1, len(history['train_loss']) + 1)

    ax.plot(epochs_range, history['train_loss'], '--', label='Train Loss')
    ax.plot(epochs_range, history['val_loss'], '--', label='Validation Loss')
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epochs")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True)
    
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    logging.info(f"損失圖表已儲存至: {save_path}") 