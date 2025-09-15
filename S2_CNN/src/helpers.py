import torch
import torchaudio
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import torchaudio.transforms as T
import torchaudio.functional as F
from torch import nn

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from utils import AudioDataset

#  資料增強輔助函式 (Data Augmentation Helpers)
def apply_spec_augment(mel_spec: torch.Tensor) -> torch.Tensor:
    """頻譜增強(向後相容於SpecAugment)"""
    transform = nn.Sequential(
        T.FrequencyMasking(freq_mask_param=30),
        T.TimeMasking(time_mask_param=50),
    )
    return transform(mel_spec)


def add_background_noise(waveform: torch.Tensor, sample_rate: int, noise_factor=0.005) -> torch.Tensor:
    """隨機背景噪音。"""
    noise = torch.randn_like(waveform) * noise_factor
    return waveform + noise

def shift_time(waveform: torch.Tensor, sample_rate: int, shift_limit=0.2) -> torch.Tensor:
    """隨機時間平移。"""
    shift_amt = int(np.random.rand() * shift_limit * waveform.shape[1])
    return torch.roll(waveform, shifts=shift_amt, dims=1)

def change_pitch_and_speed(waveform: torch.Tensor, sample_rate: int, n_fft: int, hop_length: int, pitch_factor=None, speed_factor=None) -> torch.Tensor:
    """使用 torchaudio 改變音高和速度。"""
    if pitch_factor is None:
        pitch_factor = np.random.uniform(-2, 2)
    if speed_factor is None:
        speed_factor = np.random.uniform(0.9, 1.1)

    # 波形 -> 頻譜圖
    spec_transform = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None)
    spec = spec_transform(waveform)

    # 時間拉伸
    stretch_transform = T.TimeStretch(fixed_rate=False, n_freq=n_fft // 2 + 1)
    stretched_spec = stretch_transform(spec, speed_factor)

    # 頻譜圖 -> 波形
    inverse_spec_transform = T.InverseSpectrogram(n_fft=n_fft, hop_length=hop_length)
    waveform_stretched = inverse_spec_transform(stretched_spec, length=waveform.shape[-1])

    # 音高變換
    waveform_pitched = F.pitch_shift(waveform_stretched, sample_rate, n_steps=int(pitch_factor))

    # 確保輸出長度一致
    if waveform_pitched.shape[-1] < waveform.shape[-1]:
        pad_amount = waveform.shape[-1] - waveform_pitched.shape[-1]
        waveform_pitched = torch.nn.functional.pad(waveform_pitched, (0, pad_amount))
    else:
        waveform_pitched = waveform_pitched[..., :waveform.shape[-1]]

    return waveform_pitched


#  統計計算 (Statistics Calculation)
def calculate_scaler_stats(dataset: 'AudioDataset'):
    """標準化。"""
    # 使用 tqdm 顯示進度條
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)
    total_sum = 0.
    total_sum_sq = 0.
    total_samples = 0

    print("正在計算數據集的平均值與標準差...")
    for spec, _ in tqdm(loader):
        total_sum += spec.sum()
        total_sum_sq += (spec**2).sum()
        total_samples += spec.shape[2] * spec.shape[3]

    # 均值和標準差
    mean = total_sum / total_samples
    std = (total_sum_sq / total_samples - mean**2)**0.5
    print(f"計算完成: 均值={mean}, 標準差={std}")
    return {'mean': float(mean), 'std': float(std)}

#  視覺化 (Visualization)
import matplotlib.pyplot as plt

def plot_accuracy(history, save_path):
    """準確率"""
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
    print(f"準確率圖已儲存至: {save_path}")

def plot_loss(history, save_path):
    """損失函數"""
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
    print(f"損失圖已儲存至: {save_path}")