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
import logging
from helpers import (
    load_audio, get_mel_spectrogram, apply_spec_augment, apply_specmix_augmentation,
    add_background_noise, shift_time, change_pitch_and_speed
)
# 導入新的設定中心
import config as cfg

# --- 新增：專門用於記錄過短檔案的 Logger ---
# 確保 logs 目錄存在
# --- 修復：使用來自配置的集中化記錄路徑 ---
log_dir = cfg.LOG_SAVE_DIR
os.makedirs(log_dir, exist_ok=True)

# 建立 logger
short_file_logger = logging.getLogger('short_files')
# 防止在重新載入模組時重複添加 handler
if not short_file_logger.handlers:
    short_file_logger.setLevel(logging.INFO)
    # 建立 file handler 並設定格式
    log_file_path = os.path.join(log_dir, 'short_files.log')
    fh = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    short_file_logger.addHandler(fh)

# 用於儲存已經記錄過的檔案路徑，避免在多個 epoch 中重複記錄
_logged_short_files = set()
# --- 新增結束 ---

# 舊的、用於記錄問題檔案的日誌設定已被移除，
# 因為我們現在採用了預先掃描的策略。

def setup_seed(seed):
    """
    設定隨機種子以確保實驗的可複現性。
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AudioDataset(Dataset):
    """
    客製化 PyTorch Dataset，用於處理音訊檔案。
    這個版本被重構為接收一個 pipeline_config 字典，而不是多個獨立參數。
    """
    def __init__(self, dataframe, data_dir, scaler, is_train, pipeline_config):
        self.df = dataframe
        self.data_dir = data_dir
        self.scaler = scaler
        self.is_train = is_train
        
        # 從 pipeline_config 解構出所有需要的參數
        self.pipeline_config = pipeline_config
        self.use_vad = pipeline_config.get('use_vad', False)
        self.vad_config = pipeline_config.get('vad_config', {})
        self.target_len = pipeline_config.get('target_len', 215) # 從config獲取目標長度
        self.augmentation_mode = pipeline_config.get('augmentation_mode', None) if self.is_train else None

        # --- FIX: 優先從 pipeline_config 讀取音訊參數 ---
        # 這樣的設計使得每個資料管道都可以有自己獨立的音訊處理參數，
        # 如果管道沒有特別指定，則會使用 cfg.AUDIO 中的全域預設值。
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

        # 1. 載入音訊
        waveform = load_audio(file_path, self.sample_rate)
        if waveform is None or waveform.shape[1] == 0:
            raise ValueError("音訊載入失敗或為空")

        # 2. VAD 處理 (如果啟用)
        if self.use_vad:
            # [REFACTOR] 永久性切換至 torchaudio.functional.vad
            waveform = torchaudio.functional.vad(waveform, sample_rate=self.sample_rate)
            
            if waveform.shape[1] == 0:
                raise ValueError("VAD 處理後音訊長度為 0")

        # --- FIX: VAD 後音訊長度修正 ---
        # 如果音訊長度小於 n_fft，則進行補零，以避免梅爾頻譜圖轉換失敗
        if waveform.shape[1] < self.n_fft:
            pad_size = self.n_fft - waveform.shape[1]
            # 在張量的最後一個維度（時間維度）的末尾進行補零
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))
        # --- 修復結束 ---
        
        # 3. 音訊增強 (僅在 is_train=True 時)
        if self.is_train and self.augmentation_mode == 'gentle_audio':
            # 隨機選擇一種或多種音訊層級的增強
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

        # --- RE-APPLYING FIX: 確保增強後的音訊長度依然有效 ---
        # 音訊層級的增強 (尤其是變速) 可能會縮短音訊，導致其長度小於 n_fft。
        # 因此，在計算頻譜圖之前，必須重新檢查並在必要時進行補零。
        if waveform.shape[1] < self.n_fft:
            pad_size = self.n_fft - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))
        # --- 修復結束 ---

        # 4. 轉換為梅爾頻譜圖
        mel_spec = get_mel_spectrogram(
            waveform, 
            sample_rate=self.sample_rate, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            n_mels=self.n_mels
        )
        if mel_spec.shape[2] == 0:
            raise ValueError("頻譜圖長度為 0")

        log_mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)

        # 5. 標準化長度
        # --- 新增：檢查並記錄過短的頻譜圖 ---
        actual_len = log_mel_spec.shape[2]
        if actual_len < self.target_len:
            # 只有當這個檔案路徑還沒有被記錄過時，才進行記錄
            if file_path not in _logged_short_files:
                short_file_logger.info(f"檔案長度不足: {file_path} (實際長度: {actual_len}, 目標長度: {self.target_len})")
                _logged_short_files.add(file_path)
        # --- 新增結束 ---

        if log_mel_spec.shape[2] < self.target_len:
            pad_size = self.target_len - log_mel_spec.shape[2]
            log_mel_spec = torch.nn.functional.pad(log_mel_spec, (0, pad_size))
        else:
            log_mel_spec = log_mel_spec[:, :, :self.target_len]

        # 6. 頻譜增強 (僅在 is_train=True 時)
        if self.is_train and self.augmentation_mode == 'spec_augment':
            log_mel_spec = apply_spec_augment(log_mel_spec)
        elif self.is_train and self.augmentation_mode in ['specmix', 'mixspeech', 'specmix_combined']:
            # 應用新的SpecMix增強技術
            specmix_config = self.pipeline_config.get('specmix_config', {})
            if specmix_config:
                log_mel_spec = apply_specmix_augmentation(log_mel_spec, specmix_config)
        
        # 7. 標準化
        if self.scaler: # 只有在 self.scaler 不是 None 時才進行標準化
            mean = self.scaler['mean']
            std = self.scaler['std']
            log_mel_spec_np = log_mel_spec.squeeze(0).numpy()
            standardized_spec_np = (log_mel_spec_np - mean) / std
            standardized_spec_tensor = torch.from_numpy(standardized_spec_np).float()
        else: # 在計算 scaler 時，我們只返回 log_mel_spec
            standardized_spec_tensor = log_mel_spec.squeeze(0).float()

        # 主動斷言：在回傳前，驗證頻譜圖的寬度是否符合預期。
        # 這是為了捕獲任何在處理過程中可能導致維度不匹配的潛在錯誤。
        if standardized_spec_tensor.shape[1] != self.target_len:
            raise ValueError(
                f"尺寸驗證失敗！檔案 '{file_path}' 的頻譜圖寬度為 "
                f"{standardized_spec_tensor.shape[1]}，但期望的 target_len 為 {self.target_len}。"
            )

        return standardized_spec_tensor.unsqueeze(0), torch.tensor(label, dtype=torch.long)

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