import os
import random

import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm

class SpecAugment(nn.Module):
    """對頻譜圖進行 SpecAugment """
    def __init__(self, freq_mask_param, time_mask_param, num_freq_masks=1, num_time_masks=1):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def forward(self, spectrogram):
        """回傳torch.Tensor: 經過 SpecAugment 處理後的頻譜圖。"""
        clone = spectrogram.clone()
        n_mels, time_steps = clone.shape[-2], clone.shape[-1]

        # 應用頻率遮罩
        for _ in range(self.num_freq_masks):
            f = random.randint(0, self.freq_mask_param)
            if n_mels > f:
                f0 = random.randint(0, n_mels - f)
                clone[..., f0:f0 + f, :] = 0

        # 應用時間遮罩
        for _ in range(self.num_time_masks):
            t = random.randint(0, self.time_mask_param)
            if time_steps > t:
                t0 = random.randint(0, time_steps - t)
                clone[..., :, t0:t0 + t] = 0

        return clone

#  視覺化
import matplotlib.pyplot as plt

def plot_accuracy(history, experiment_name, save_dir):
    """繪製並儲存準確率歷史圖。"""
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs_range = range(1, len(history['train_acc']) + 1)

    ax.plot(epochs_range, history['train_acc'], '-o', label='Train Accuracy', markersize=4)
    ax.plot(epochs_range, history['val_acc'], '-o', label='Validation Accuracy', markersize=4)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Epochs")
    ax.set_title(f"Training and Validation Accuracy for {experiment_name}")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    save_path = os.path.join(save_dir, f"{experiment_name}_accuracy.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"準確率圖已儲存至: {save_path}")

def plot_loss(history, experiment_name, save_dir):
    """繪製並儲存損失歷史圖。"""
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs_range = range(1, len(history['train_loss']) + 1)

    ax.plot(epochs_range, history['train_loss'], '--', label='Train Loss')
    ax.plot(epochs_range, history['val_loss'], '--', label='Validation Loss')
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epochs")
    ax.set_title(f"Training and Validation Loss for {experiment_name}")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    save_path = os.path.join(save_dir, f"{experiment_name}_loss.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"損失圖已儲存至: {save_path}")