import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import argparse
import json
from tqdm import tqdm

import transformer_config as cfg
from models import get_model
import torchaudio.transforms as T

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class TestDataset(Dataset):
    """
    用於載入測試音頻檔案進行預測的數據集。
    執行與驗證集相同的預處理。
    """
    def __init__(self, file_list, test_data_dir, audio_settings):
        self.file_list = file_list
        self.test_data_dir = test_data_dir
        self.audio_settings = audio_settings

        self.resampler = T.Resample(orig_freq=32000, new_freq=self.audio_settings['sample_rate'])
        self.to_db = T.AmplitudeToDB(stype='power', top_db=80)
        self.mel_spectrogram_transform = T.MelSpectrogram(
            sample_rate=self.audio_settings['sample_rate'],
            n_fft=self.audio_settings['n_fft'],
            hop_length=self.audio_settings['hop_length'],
            n_mels=self.audio_settings['n_mels']
        )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.test_data_dir, file_name)

        try:
            waveform, sample_rate = torchaudio.load(file_path)
            
            if sample_rate != self.audio_settings['sample_rate']:
                waveform = self.resampler(waveform)

            if waveform.shape[1] < self.audio_settings['n_fft']:
                pad_size = self.audio_settings['n_fft'] - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_size))
            
            mel_spectrogram = self.mel_spectrogram_transform(waveform)
            mel_spectrogram_db = self.to_db(mel_spectrogram)

            target_len = self.audio_settings['target_len']
            if mel_spectrogram_db.shape[2] < target_len:
                pad_size = target_len - mel_spectrogram_db.shape[2]
                mel_spectrogram_db = torch.nn.functional.pad(mel_spectrogram_db, (0, pad_size))
            else:
                mel_spectrogram_db = mel_spectrogram_db[:, :, :target_len]

            return mel_spectrogram_db, file_name
        
        except Exception as e:
            print(f"Error processing file '{file_path}': {e}")
            return None, file_name

def predict(experiment_name: str, output_csv_path: str):
    """
    主要預測函數。
    """
    print(f"--- Starting Prediction for Experiment: {experiment_name} ---")
    print(f"Using device: {DEVICE}")

    # 1. 載入實驗配置
    try:
        exp_config = cfg.get_config(experiment_name)
        model_blueprint = exp_config['model_blueprint']
        audio_settings = exp_config['audio_settings']
    except ValueError as e:
        print(f"Error: {e}")
        return

    # 2. 載入模型
    model_path = os.path.join(cfg.MODEL_SAVE_DIR, f"{experiment_name}.pth")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        return
        
    model = get_model(model_blueprint, audio_settings)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(DEVICE)
    model.eval()
    print(f"Model '{experiment_name}' loaded successfully.")

    # 3. 準備測試數據
    test_data_dir = os.path.join(cfg.get_main_project_root(), '04_Data', 'voice_dataset', 'test_set')
    test_files = [f for f in os.listdir(test_data_dir) if f.endswith('.wav')]
    
    test_dataset = TestDataset(test_files, test_data_dir, audio_settings)
    # 使用基本的 collate_fn 來處理潛在錯誤
    def collate_fn(batch):
        batch = [item for item in batch if item[0] is not None]
        if not batch:
            return None, None
        tensors, filenames = zip(*batch)
        return torch.stack(tensors), filenames

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # 4. 執行預測
    predictions = []
    filenames = []
    with torch.no_grad():
        for inputs, f_names in tqdm(test_loader, desc="Predicting"):
            if inputs is None:
                continue
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted_labels = torch.max(outputs.data, 1)
            
            predictions.extend(predicted_labels.cpu().numpy())
            filenames.extend(f_names)
            
    # 5. 建立並儲存提交檔案
    submission_df = pd.DataFrame({'wav': filenames, 'label': predictions})
    submission_df.to_csv(output_csv_path, index=False)
    print(f"\nPrediction complete! Results saved to: {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predictions using a trained Transformer model.")
    parser.add_argument(
        "--experiment", 
        type=str, 
        required=True, 
        help="The name of the experiment to use for prediction."
    )
    parser.add_argument(
        "--output_csv", 
        type=str, 
        default=None, 
        help="Path to save the output submission CSV file. Defaults to submission_{experiment_name}.csv"
    )
    args = parser.parse_args()
    
    output_path = args.output_csv
    if output_path is None:
        # 與 MLP/CNN 專案統一，儲存到根目錄的 submissions 資料夾
        submission_dir = os.path.join(cfg.get_main_project_root(), "submissions")
        os.makedirs(submission_dir, exist_ok=True)
        output_path = os.path.join(submission_dir, f"submission_Transformer_Exploration_{args.experiment}.csv")
        
    predict(args.experiment, output_path) 