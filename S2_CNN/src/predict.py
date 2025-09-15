import torch
import pandas as pd
import numpy as np
import os
import torchaudio
from tqdm import tqdm
import argparse
import json

# 本地模組
from .models import get_model
from . import config as cfg
from .utils import AudioDataset  # 單一樣本預處理
from .data_pipeline import get_dataloaders  # 自動產生 scaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_single_audio(file_path, scaler, pipeline_config):
    """單一音訊預處理"""
    waveform, sr = torchaudio.load(file_path)

    # 重採樣
    if sr != cfg.AUDIO['sample_rate']:
        resampler = torchaudio.transforms.Resample(sr, cfg.AUDIO['sample_rate'])
        waveform = resampler(waveform)

    # VAD 處理
    if pipeline_config.get('use_vad', False):
        waveform = torchaudio.functional.vad(waveform, sample_rate=cfg.AUDIO['sample_rate'])
        if waveform.shape[1] == 0:
            print(f"警告：VAD 處理後檔案 {file_path} 長度為 0")
            return None

    # 轉換為梅爾頻譜圖
    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.AUDIO['sample_rate'],
        n_fft=cfg.AUDIO['n_fft'],
        hop_length=cfg.AUDIO['hop_length'],
        n_mels=cfg.AUDIO['n_mels']
    )
    mel_spec = mel_spectrogram_transform(waveform)
    log_mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)

    # 標準化長度
    target_len = pipeline_config.get('target_len', 215)
    if log_mel_spec.shape[2] < target_len:
        pad_size = target_len - log_mel_spec.shape[2]
        log_mel_spec = torch.nn.functional.pad(log_mel_spec, (0, pad_size))
    else:
        log_mel_spec = log_mel_spec[:, :, :target_len]

    # 標準化 (使用 scaler)
    mean = scaler['mean']
    std = scaler['std']
    standardized_spec_np = (log_mel_spec.squeeze(0).numpy() - mean) / std
    standardized_spec_tensor = torch.from_numpy(standardized_spec_np).float()

    return standardized_spec_tensor.unsqueeze(0).unsqueeze(0)

def predict(experiment_name, output_path=None):
    """預測流程"""

    # 建構模型路徑
    model_save_name = f"{experiment_name}.pth"
    model_path = os.path.join(cfg.MODEL_SAVE_DIR, model_save_name)

    # 載入模型和關聯的設定檔
    config_path = model_path.replace('.pth', '.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"成功載入實驗 '{experiment_name}' 的設定檔")

    model_blueprint = config['model_blueprint']
    data_pipeline_config = config['data_pipeline']

    # 找出管道名稱以載入正確的標準化器
    pipeline_name = "default"
    for name, config_dict in cfg.DATA_PIPELINES.items():
        if config_dict == data_pipeline_config:
            pipeline_name = name
            break

    data_dims = {
        'input_h': cfg.AUDIO['n_mels'],
        'input_w': data_pipeline_config['target_len'],
        'output_dim': cfg.AUDIO['num_classes']
    }

    # 合併模型藍圖和資料維度，以符合 get_model 的單一參數要求
    full_model_config = {**model_blueprint, **data_dims}

    model = get_model(full_model_config)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # 標準化器 (Scaler)
    scaler_path = os.path.join(cfg.PROCESSED_DATA_DIR, f'scaler_{pipeline_name}.pt')
    print(f"正在載入標準化器: {scaler_path}")
    scaler = torch.load(scaler_path, weights_only=False)

    # 讀取提交範本
    submission_df = pd.read_csv(cfg.SUBMISSION_CSV_PATH)
    predictions = []

    print("開始預測")
    # 迭代測試檔案進行預測
    test_data_dir = cfg.TEST_SET_DIR
    with torch.no_grad():
        for wav_filename in tqdm(submission_df['wav'], desc="預測進度"):
            file_path = os.path.join(test_data_dir, wav_filename)
            input_tensor = preprocess_single_audio(file_path, scaler, data_pipeline_config)
            if input_tensor is None:
                # 如果處理失敗，使用預設標籤0
                predictions.append(0)
                continue
            input_tensor = input_tensor.to(DEVICE)
            output = model(input_tensor)
            _, predicted_label = torch.max(output, 1)
            predictions.append(predicted_label.item())

    # 建立並儲存提交檔案
    submission_df['label'] = predictions

    submission_dir = os.path.join(cfg.MAIN_PROJECT_ROOT, "submissions")
    os.makedirs(submission_dir, exist_ok=True)
    output_path = os.path.join(submission_dir, f"submission_CNN_Audio_Classification_{experiment_name}.csv")

    submission_df.to_csv(output_path, index=False)

def save_experiment_config(experiment_name, model_save_path):
    """將設定儲存為 JSON 檔案"""
    config_save_path = model_save_path.replace('.pth', '.json')

    config_to_save = {
        'experiment_name': experiment_name,
        'model_blueprint': cfg.EXPERIMENTS[experiment_name]['model_blueprint'],
        'data_pipeline': cfg.EXPERIMENTS[experiment_name]['data_pipeline']
    }

    with open(config_save_path, 'w') as f:
        json.dump(config_to_save, f, indent=4)
    print(f"實驗設定已儲存至: {config_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default=None)
    args = parser.parse_args()

    predict(args.experiment, args.output_csv)