import torch
import torchaudio
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import torchaudio.transforms as T
import torchaudio.functional as F
from torch import nn

# 設定TorchAudio後端使用soundfile
torchaudio.set_audio_backend("soundfile")

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
#  資料增強輔助函式 (Data Augmentation Helpers)
# ==============================================================================

def apply_spec_augment(mel_spec: torch.Tensor) -> torch.Tensor:
    """應用頻譜增強 (SpecAugment) - 保持向後相容性。"""
    transform = nn.Sequential(
        T.FrequencyMasking(freq_mask_param=30),
        T.TimeMasking(time_mask_param=50),
    )
    return transform(mel_spec)

def apply_specmix_augmentation(mel_spec: torch.Tensor, specmix_config: dict, 
                              secondary_spec: torch.Tensor = None) -> torch.Tensor:
    """
    應用SpecMix增強技術 - 2024年最新研究，比SpecAugment高4.45%準確率
    
    Args:
        mel_spec: 主要頻譜圖
        specmix_config: SpecMix配置參數
        secondary_spec: 次要頻譜圖（用於混合）
        
    Returns:
        增強後的頻譜圖
    """
    # Import here to avoid circular imports
    import sys
    import os
    # Add the project root to Python path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    sys.path.insert(0, project_root)
    
    from specmix_augmentation import get_specmix_augmentation
    
    # Create augmentation instance
    augmenter = get_specmix_augmentation(specmix_config)
    
    # Apply augmentation
    if secondary_spec is not None:
        # For mixing augmentations, we need dummy labels
        dummy_label = torch.tensor(0)  # Will be overridden by dataset
        augmented_spec, _ = augmenter(mel_spec, secondary_spec, dummy_label, dummy_label)
    else:
        # For single-sample augmentations
        dummy_label = torch.tensor(0)
        augmented_spec, _ = augmenter(mel_spec, None, dummy_label, None)
    
    return augmented_spec

def add_background_noise(waveform: torch.Tensor, sample_rate: int, noise_factor=0.005) -> torch.Tensor:
    """添加隨機背景噪音。"""
    noise = torch.randn_like(waveform) * noise_factor
    return waveform + noise

def shift_time(waveform: torch.Tensor, sample_rate: int, shift_limit=0.2) -> torch.Tensor:
    """隨機時間平移。"""
    shift_amt = int(np.random.rand() * shift_limit * waveform.shape[1])
    return torch.roll(waveform, shifts=shift_amt, dims=1)

def change_pitch_and_speed(waveform: torch.Tensor, sample_rate: int, n_fft: int, hop_length: int, pitch_factor=None, speed_factor=None) -> torch.Tensor:
    """
    使用 torchaudio 改變音高和速度。
    此函式取代了先前基於 librosa 的實作。
    """
    if pitch_factor is None:
        pitch_factor = np.random.uniform(-2, 2)
    if speed_factor is None:
        speed_factor = np.random.uniform(0.9, 1.1)

    # 步驟 1: 時間拉伸 (改變速度，保持音高)
    # 1a. 波形 -> 頻譜圖
    spec_transform = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None)
    spec = spec_transform(waveform)

    # 1b. 時間拉伸
    stretch_transform = T.TimeStretch(fixed_rate=False, n_freq=n_fft // 2 + 1)
    stretched_spec = stretch_transform(spec, speed_factor)

    # 1c. 頻譜圖 -> 波形
    inverse_spec_transform = T.InverseSpectrogram(n_fft=n_fft, hop_length=hop_length)
    waveform_stretched = inverse_spec_transform(stretched_spec, length=waveform.shape[-1])

    # 步驟 2: 音高變換 (改變音高，保持速度)
    waveform_pitched = F.pitch_shift(waveform_stretched, sample_rate, n_steps=pitch_factor)
    
    # 確保輸出長度與輸入一致
    if waveform_pitched.shape[-1] < waveform.shape[-1]:
        pad_amount = waveform.shape[-1] - waveform_pitched.shape[-1]
        waveform_pitched = torch.nn.functional.pad(waveform_pitched, (0, pad_amount))
    else:
        waveform_pitched = waveform_pitched[..., :waveform.shape[-1]]
        
    return waveform_pitched


# ==============================================================================
#  統計計算 (Statistics Calculation)
# ==============================================================================

def calculate_scaler_stats(dataset: 'AudioDataset'):
    """
    計算數據集的均值和標準差以進行標準化。
    這個函式現在是獨立的，可以被任何需要計算統計數據的模組使用。
    """
    # 使用 tqdm 顯示進度條
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)
    
    # 初始化變量以累積總和和平方總和
    total_sum = 0.
    total_sum_sq = 0.
    total_samples = 0

    print("正在計算數據集的平均值與標準差...")
    for spec, _ in tqdm(loader):
        # spec 的維度是 (1, 1, n_mels, n_frames)
        # 我們需要在頻率和時間維度上計算
        total_sum += spec.sum()
        total_sum_sq += (spec**2).sum()
        # 梅爾頻道數 * 幀數
        total_samples += spec.shape[2] * spec.shape[3]

    # 計算均值和標準差
    mean = total_sum / total_samples
    std = (total_sum_sq / total_samples - mean**2)**0.5

    print(f"計算完成: Mean={mean}, Std={std}")
    
    return {'mean': mean.item(), 'std': std.item()} 

# ==============================================================================
#  視覺化 (Visualization)
# ==============================================================================
import matplotlib.pyplot as plt
import logging

def plot_accuracy(history, save_path):
    """繪製並儲存準確率歷史圖。"""
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
    logging.info(f"準確率圖已儲存至: {save_path}")

def plot_loss(history, save_path):
    """繪製並儲存損失歷史圖。"""
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
    logging.info(f"損失圖已儲存至: {save_path}") 