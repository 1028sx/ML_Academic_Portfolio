import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import torchaudio.transforms as T
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from .helpers import SpecAugment


class AudioDataset(Dataset):
    """載入→預處理→增強→回傳"""
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

        # SpecAugment
        if self.augmentation_config and self.augmentation_config.get('type') == 'spec_augment':
            self.spec_augment = SpecAugment(
                freq_mask_param=self.augmentation_config['freq_mask_param'],
                time_mask_param=self.augmentation_config['time_mask_param'],
                num_freq_masks=self.augmentation_config['num_freq_masks'],
                num_time_masks=self.augmentation_config['num_time_masks'],
            )


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

            # 應用SpecAugment
            if self.spec_augment:
                mel_spectrogram_db = self.spec_augment(mel_spectrogram_db)


            target_len = self.audio_settings['target_len']
            if mel_spectrogram_db.shape[2] < target_len:
                pad_size = target_len - mel_spectrogram_db.shape[2]
                mel_spectrogram_db = torch.nn.functional.pad(mel_spectrogram_db, (0, pad_size))
            else:
                mel_spectrogram_db = mel_spectrogram_db[:, :, :target_len]

            return mel_spectrogram_db, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"處理檔案 '{file_path}' 時發生錯誤: {e}")
            return torch.zeros(1, self.audio_settings['n_mels'], self.audio_settings['target_len']), torch.tensor(0, dtype=torch.long)

def get_dataloaders(exp_config: dict):
    """建立並返回訓練和驗證 DataLoaders 的主函數。"""
    print("--- 準備資料載入器 ---")

    # 提取配置
    data_dir = exp_config['data_dir']
    csv_path = exp_config['csv_path']
    audio_settings = exp_config['audio_settings']
    data_pipeline_config = exp_config['data_pipeline']
    batch_size = exp_config['batch_size']

    df = pd.read_csv(csv_path)

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    print(f"訓練集大小: {len(train_df)}")
    print(f"驗證集大小: {len(val_df)}")

    # 僅對訓練集應用增強
    augmentation_config = data_pipeline_config.get('augmentation_config')

    train_dataset = AudioDataset(train_df, data_dir, audio_settings, augmentation_config=augmentation_config)
    val_dataset = AudioDataset(val_df, data_dir, audio_settings, augmentation_config=None)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print("--- 資料載入器準備完成 ---")
    return train_loader, val_loader