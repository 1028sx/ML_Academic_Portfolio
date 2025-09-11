import os
import logging
import random

import pandas as pd
import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm


# --- 設定日誌 ---
# 確保 logs 目錄存在
log_dir = '30_Transformer_Exploration/output/logs'
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, 'data_health_check.log')

# 建立 logger
health_check_logger = logging.getLogger('data_health_check')
health_check_logger.setLevel(logging.INFO)

# 防止重複添加 handler
if not health_check_logger.handlers:
    # 建立 file handler
    fh = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    health_check_logger.addHandler(fh)
    
    # 建立 console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    health_check_logger.addHandler(ch)

def run_health_check(df: pd.DataFrame, data_dir: str, sample_rate: int) -> pd.DataFrame:
    """
    對資料集中的音訊檔案進行健康檢查。

    Args:
        df (pd.DataFrame): 包含 'wav' 和 'label' 欄位的 DataFrame。
        data_dir (str): 音訊檔案所在的目錄。
        sample_rate (int): 讀取音訊時使用的取樣率。

    Returns:
        pd.DataFrame: 只包含健康檔案的 DataFrame。
    """
    health_check_logger.info("--- 開始進行音訊檔案健康檢查 ---")
    
    healthy_indices = []
    unhealthy_files = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="檢查音訊檔案"):
        file_path = os.path.join(data_dir, row['wav'])
        try:
            # 1. 檢查檔案是否存在
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"檔案不存在: {file_path}")

            # 2. 嘗試載入音訊
            waveform, sr = torchaudio.load(file_path)

            # 3. 檢查取樣率是否一致
            if sr != sample_rate:
                health_check_logger.warning(f"取樣率不匹配: {file_path} (實際: {sr}, 期望: {sample_rate})。將進行重採樣。")
            
            # 4. 檢查音訊是否為空
            if waveform.numel() == 0:
                raise ValueError("音訊檔案為空或已損壞。")
            
            healthy_indices.append(index)

        except Exception as e:
            # 捕獲所有可能的錯誤 (soundfile.LibsndfileError, RuntimeError, etc.)
            unhealthy_files.append(row['wav'])
            health_check_logger.error(f"檔案 '{row['wav']}' 不健康，將被排除。錯誤: {e}", exc_info=False)

    health_check_logger.info(f"--- 健康檢查完成 ---")
    health_check_logger.info(f"總共檢查了 {len(df)} 個檔案。")
    health_check_logger.info(f"發現 {len(healthy_indices)} 個健康檔案。")
    health_check_logger.info(f"排除 {len(unhealthy_files)} 個不健康檔案。")
    
    if unhealthy_files:
        health_check_logger.warning(f"不健康檔案列表: {unhealthy_files}")

    return df.loc[healthy_indices].copy()

class SpecAugment(nn.Module):
    """
    對頻譜圖進行 SpecAugment。

    SpecAugment是一種針對音訊頻譜圖設計的資料增強技術，
    它通過遮蔽(masking)頻譜圖中的一部分時間步和頻率通道來實現。
    這種方法可以讓模型學習到更穩健的特徵，提高對雜訊和變形的容忍度。

    參考論文: SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition
    (https://arxiv.org/abs/1904.08779)
    """
    def __init__(self, freq_mask_param, time_mask_param, num_freq_masks=1, num_time_masks=1):
        """
        Args:
            freq_mask_param (int): 頻率遮罩的最大寬度 F。
            time_mask_param (int): 時間遮罩的最大寬度 T。
            num_freq_masks (int): 要應用的頻率遮罩數量。
            num_time_masks (int): 要應用的時間遮罩數量。
        """
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def forward(self, spectrogram):
        """
        Args:
            spectrogram (torch.Tensor): 輸入的頻譜圖，形狀為 (..., n_mels, time_steps)。

        Returns:
            torch.Tensor: 經過 SpecAugment 處理後的頻譜圖。
        """
        clone = spectrogram.clone()
        n_mels, time_steps = clone.shape[-2], clone.shape[-1]

        # 應用頻率遮罩
        for _ in range(self.num_freq_masks):
            f = random.randint(0, self.freq_mask_param)
            if n_mels > f: # 確保遮罩寬度不超過總寬度
                f0 = random.randint(0, n_mels - f)
                clone[..., f0:f0 + f, :] = 0

        # 應用時間遮罩
        for _ in range(self.num_time_masks):
            t = random.randint(0, self.time_mask_param)
            if time_steps > t: # 確保遮罩寬度不超過總長度
                t0 = random.randint(0, time_steps - t)
                clone[..., :, t0:t0 + t] = 0

        return clone

# ==============================================================================
#  視覺化 (Visualization)
# ==============================================================================
import matplotlib.pyplot as plt

def plot_accuracy(history, experiment_name, save_dir):
    """繪製並儲存準確率歷史圖。"""
    plt.style.use('seaborn-v0_8-whitegrid')
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
    logging.info(f"準確率圖已儲存至: {save_path}")

def plot_loss(history, experiment_name, save_dir):
    """繪製並儲存損失歷史圖。"""
    plt.style.use('seaborn-v0_8-whitegrid')
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
    logging.info(f"損失圖已儲存至: {save_path}") 