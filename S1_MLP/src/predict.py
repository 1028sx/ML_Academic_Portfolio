import torch
import pandas as pd
import numpy as np
import os
import argparse
import json
import torchaudio
from tqdm import tqdm

# MLP專用
from . import config as cfg
from .models import get_model
from .data_pipeline import get_dataloaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_single_audio(file_path, scaler, pipeline_config):
    waveform, sr = torchaudio.load(file_path)

    if sr != cfg.AUDIO['sample_rate']:
        resampler = torchaudio.transforms.Resample(sr, cfg.AUDIO['sample_rate'])
        waveform = resampler(waveform)

    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.AUDIO['sample_rate'],
        n_fft=cfg.AUDIO['n_fft'],
        hop_length=cfg.AUDIO['hop_length'],
        n_mels=cfg.AUDIO['n_mels']
    )
    mel_spec = mel_spectrogram_transform(waveform)
    log_mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)

    target_len = pipeline_config.get('target_len', 215)
    if log_mel_spec.shape[2] < target_len:
        pad_size = target_len - log_mel_spec.shape[2]
        log_mel_spec = torch.nn.functional.pad(log_mel_spec, (0, pad_size))
    else:
        log_mel_spec = log_mel_spec[:, :, :target_len]

    # 標準化
    if scaler:
        mean = scaler['mean']
        std = scaler['std']
        standardized_spec_np = (log_mel_spec.squeeze(0).numpy() - mean.numpy()) / std.numpy()
        input_tensor = torch.from_numpy(standardized_spec_np).float().unsqueeze(0)
    else:
        input_tensor = log_mel_spec

    return torch.flatten(input_tensor).unsqueeze(0)


def predict(experiment_name, output_path=None):
    # 建構模型和設定檔路徑
    model_name = f"mlp_{experiment_name}.pth"
    model_path = os.path.join(cfg.MODEL_SAVE_DIR, model_name)
    config_path = model_path.replace('.pth', '.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 載入模型
    model = get_model(config['model'])
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()

    # 3. 準備 Scaler
    pipeline_name = next(iter(cfg.DATA_PIPELINES))
    scaler_path = os.path.join(cfg.PROCESSED_DATA_DIR, f'scaler_{pipeline_name}.pt')
    scaler = torch.load(scaler_path)

    # 提交檔案範本和測試資料路徑
    submission_template_path = os.path.join(cfg.RAW_DATA_DIR, 'submission.csv')
    submission_df = pd.read_csv(submission_template_path)
    test_data_dir = os.path.join(cfg.DATA_DIR, 'voice_dataset', 'test_set')

    # 預測
    print("開始預測")
    predictions = []
    with torch.no_grad():
        for wav_filename in tqdm(submission_df['wav'], desc="預測進度"):
            file_path = os.path.join(test_data_dir, wav_filename)

            input_tensor = preprocess_single_audio(file_path, scaler, config['data'])

            input_tensor = input_tensor.to(DEVICE)

            output = model(input_tensor)
            _, predicted_label = torch.max(output, 1)
            predictions.append(predicted_label.item())

    # 建立提交檔
    submission_df['label'] = predictions

    submission_dir = os.path.join(cfg.MAIN_PROJECT_ROOT, "submissions")
    os.makedirs(submission_dir, exist_ok=True)
    output_path = os.path.join(submission_dir, f"submission_General_ML_{experiment_name}.csv")

    submission_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=list(cfg.EXPERIMENTS.keys())
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None
    )
    args = parser.parse_args()

    predict(args.experiment, args.output_csv)