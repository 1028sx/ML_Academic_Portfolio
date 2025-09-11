"""
Transformer 實驗的數據載入和處理管道。
"""
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import torchaudio.transforms as T
from sklearn.model_selection import train_test_split
import logging
from tqdm import tqdm
from helpers import SpecAugment

# 設定日誌記錄器
pipeline_logger = logging.getLogger(__name__)

def run_health_check(df: pd.DataFrame, data_dir: str, sample_rate: int) -> pd.DataFrame:
    """
    對音頻檔案進行健康檢查，過濾掉損壞或遺失的檔案。
    """
    pipeline_logger.info("--- Starting audio file health check ---")
    healthy_indices = []
    unhealthy_files = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Checking audio files"):
        file_path = os.path.join(data_dir, row['wav'])
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            waveform, sr = torchaudio.load(file_path)
            if waveform.numel() == 0:
                raise ValueError("Audio file is empty or corrupted.")
            healthy_indices.append(index)
        except Exception as e:
            unhealthy_files.append(row['wav'])
            pipeline_logger.error(f"File '{row['wav']}' is unhealthy and will be excluded. Error: {e}", exc_info=False)
    
    pipeline_logger.info("--- Health check complete ---")
    pipeline_logger.info(f"Total files checked: {len(df)}")
    pipeline_logger.info(f"Found {len(healthy_indices)} healthy files.")
    pipeline_logger.info(f"Excluded {len(unhealthy_files)} unhealthy files.")
    if unhealthy_files:
        pipeline_logger.warning(f"List of unhealthy files: {unhealthy_files}")
    return df.loc[healthy_indices].copy()

class AudioDataset(Dataset):
    """
    自定義音頻數據集。
    - 載入音頻檔案。
    - 執行預處理，如重採樣和轉換為梅爾頻譜圖。
    - 標準化頻譜圖長度。
    - 如果指定則應用增強。
    """
    def __init__(self, dataframe, data_dir, audio_settings, augmentation_config=None):
        self.df = dataframe
        self.data_dir = data_dir
        self.audio_settings = audio_settings
        self.augmentation_config = augmentation_config

        self.resampler = T.Resample(orig_freq=32000, new_freq=self.audio_settings['sample_rate'])
        self.to_db = T.AmplitudeToDB(stype='power', top_db=80)
        self.mel_spectrogram_transform = T.MelSpectrogram(
            sample_rate=self.audio_settings['sample_rate'],
            n_fft=self.audio_settings['n_fft'],
            hop_length=self.audio_settings['hop_length'],
            n_mels=self.audio_settings['n_mels']
        )

        self.spec_augment = None
        self.specmix_augment = None
        
        # Traditional SpecAugment
        if self.augmentation_config and self.augmentation_config.get('type') == 'spec_augment':
            self.spec_augment = SpecAugment(
                freq_mask_param=self.augmentation_config['freq_mask_param'],
                time_mask_param=self.augmentation_config['time_mask_param'],
                num_freq_masks=self.augmentation_config['num_freq_masks'],
                num_time_masks=self.augmentation_config['num_time_masks'],
            )
        
        # SpecMix augmentation (2024 state-of-the-art)
        elif self.augmentation_config and self.augmentation_config.get('type') in ['specmix', 'mixspeech', 'specmix_combined']:
            # Import SpecMix augmentation
            import sys
            import os
            # Add the project root to Python path
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            sys.path.insert(0, project_root)
            
            from specmix_augmentation import get_specmix_augmentation
            self.specmix_augment = get_specmix_augmentation(self.augmentation_config)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.data_dir, row['wav'])
        label = row['label']

        try:
            waveform, sample_rate = torchaudio.load(file_path)
            
            if sample_rate != self.audio_settings['sample_rate']:
                waveform = self.resampler(waveform)

            if waveform.shape[1] < self.audio_settings['n_fft']:
                pad_size = self.audio_settings['n_fft'] - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_size))
            
            mel_spectrogram = self.mel_spectrogram_transform(waveform)
            mel_spectrogram_db = self.to_db(mel_spectrogram)

            # Apply traditional SpecAugment
            if self.spec_augment:
                mel_spectrogram_db = self.spec_augment(mel_spectrogram_db)
            
            # Apply SpecMix augmentation (2024 state-of-the-art)
            elif self.specmix_augment:
                # For single-sample augmentations, we don't need secondary samples here
                # The DataLoader level mixing will be handled separately if needed
                dummy_label = torch.tensor(label, dtype=torch.long)
                mel_spectrogram_db, _ = self.specmix_augment(
                    mel_spectrogram_db, None, dummy_label, None
                )

            target_len = self.audio_settings['target_len']
            if mel_spectrogram_db.shape[2] < target_len:
                pad_size = target_len - mel_spectrogram_db.shape[2]
                mel_spectrogram_db = torch.nn.functional.pad(mel_spectrogram_db, (0, pad_size))
            else:
                mel_spectrogram_db = mel_spectrogram_db[:, :, :target_len]

            return mel_spectrogram_db, torch.tensor(label, dtype=torch.long)
        
        except Exception as e:
            pipeline_logger.error(f"Error processing file '{file_path}' in __getitem__: {e}", exc_info=True)
            return torch.zeros(1, self.audio_settings['n_mels'], self.audio_settings['target_len']), torch.tensor(0, dtype=torch.long)

def get_dataloaders(exp_config: dict):
    """
    建立並返回訓練和驗證 DataLoaders 的主函數。
    """
    pipeline_logger.info("--- Preparing DataLoaders ---")
    
    # 從主實驗配置中提取配置
    data_dir = exp_config['data_dir']
    csv_path = exp_config['csv_path']
    audio_settings = exp_config['audio_settings']
    data_pipeline_config = exp_config['data_pipeline']
    batch_size = exp_config['batch_size']
    
    df = pd.read_csv(csv_path)

    healthy_df = run_health_check(df, data_dir, audio_settings['sample_rate'])
    
    if len(healthy_df) == 0:
        pipeline_logger.critical("No healthy audio files found. Cannot proceed.")
        raise RuntimeError("No usable data after health check.")

    train_df, val_df = train_test_split(healthy_df, test_size=0.2, random_state=42, stratify=healthy_df['label'])

    pipeline_logger.info(f"Training set size: {len(train_df)}")
    pipeline_logger.info(f"Validation set size: {len(val_df)}")

    # 僅對訓練集應用增強
    augmentation_config = data_pipeline_config.get('augmentation_config')
    
    train_dataset = AudioDataset(train_df, data_dir, audio_settings, augmentation_config=augmentation_config)
    val_dataset = AudioDataset(val_df, data_dir, audio_settings, augmentation_config=None)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    pipeline_logger.info("--- DataLoaders ready ---")
    return train_loader, val_loader 