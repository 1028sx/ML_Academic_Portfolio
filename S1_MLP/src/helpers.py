import torch
import torchaudio
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import torchaudio.transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from utils import AudioDataset

#  計算函式
def calculate_scaler_stats(dataset: 'AudioDataset'):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    mean = 0.
    std = 0.
    total_samples = 0

    for spec, _ in tqdm(loader):
        if spec.dim() >= 2:
            # MLP[批次, 特性]
            if spec.dim() == 2:
                batch_samples = spec.size(0)
                mean += spec.mean(dim=1).sum()
                std += spec.std(dim=1).sum()
                total_samples += batch_samples
            # CNN[批次, 通道, 高度, 寬度]
            elif spec.dim() == 4:
                batch_samples = spec.size(0)
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


def plot_accuracy(history, save_path):
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
    print(f"準確率圖表已儲存至: {save_path}")

def plot_loss(history, save_path):
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
    print(f"損失圖表已儲存至: {save_path}")