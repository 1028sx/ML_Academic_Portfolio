import torch
import pandas as pd
import numpy as np
import os
import argparse
import json
import torchaudio
from tqdm import tqdm

# 導入 MLP 專案的相關模組
from . import mlp_config as cfg
from .mlp_models import get_model
from .data_pipeline import get_dataloaders # 為了自動產生 scaler 而導入

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_single_audio(file_path, scaler, pipeline_config):
    """
    對單一音訊檔案進行預處理，返回標準化、展平後的梅爾頻譜圖張量。
    """
    try:
        waveform, sr = torchaudio.load(file_path)
    except Exception as e:
        print(f"載入檔案錯誤 {file_path}: {e}")
        return None

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
        
    # 4. 標準化
    if scaler:
        mean = scaler['mean']
        std = scaler['std']
        # 確保兩邊都是 numpy 陣列再進行運算
        standardized_spec_np = (log_mel_spec.squeeze(0).numpy() - mean.numpy()) / std.numpy()
        input_tensor = torch.from_numpy(standardized_spec_np).float().unsqueeze(0)
    else:
        input_tensor = log_mel_spec
    
    # 為 MLP 模型展平特徵
    return torch.flatten(input_tensor).unsqueeze(0)


def predict(experiment_name, output_path=None):
    """
    使用訓練好的 MLP 模型對測試集進行預測。
    """
    print(f"使用設備: {DEVICE}")

    # 1. 根據實驗名稱建構模型和設定檔路徑
    model_name = f"mlp_{experiment_name}.pth"
    model_path = os.path.join(cfg.MODEL_SAVE_DIR, model_name)
    config_path = model_path.replace('.pth', '.json')

    if not os.path.exists(model_path) or not os.path.exists(config_path):
        print(f"錯誤：找不到模型 '{model_name}' 或其設定檔。")
        return
        
    print(f"正在載入設定檔: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 2. 載入模型
    print(f"正在載入模型: {model_path}")
    model = get_model(config['model'])
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()

    # 3. 準備 Scaler
    pipeline_name = next(iter(cfg.DATA_PIPELINES))
    scaler_path = os.path.join(cfg.PROCESSED_DATA_DIR, f'scaler_{pipeline_name}.pt')
    
    try:
        scaler = torch.load(scaler_path)
        print("標準化器 (Scaler) 已載入。")
    except FileNotFoundError:
        print(f"警告：找不到標準化器檔案 '{scaler_path}'。將嘗試自動產生...")
        try:
            get_dataloaders(config, force_recalc_scaler=True)
            print("標準化器檔案產生成功，重新載入...")
            scaler = torch.load(scaler_path)
        except Exception as e:
            print(f"自動產生 Scaler 時發生嚴重錯誤: {e}")
            return

    # 4. 準備提交檔案範本和測試資料路徑
    submission_template_path = os.path.join(cfg.RAW_DATA_DIR, 'submission.csv')
    submission_df = pd.read_csv(submission_template_path)
    test_data_dir = os.path.join(cfg.DATA_DIR, 'voice_dataset', 'test_set')
    
    # 5. 執行預測
    print("開始預測...")
    predictions = []
    with torch.no_grad():
        for wav_filename in tqdm(submission_df['wav'], desc="預測進度"):
            file_path = os.path.join(test_data_dir, wav_filename)
            
            if not os.path.exists(file_path):
                print(f"警告: 找不到檔案 {file_path}，將使用預設標籤 0")
                predictions.append(0)
                continue
                
            input_tensor = preprocess_single_audio(file_path, scaler, config['data'])
            if input_tensor is None:
                predictions.append(0)
                continue
                
            input_tensor = input_tensor.to(DEVICE)
            
            output = model(input_tensor)
            _, predicted_label = torch.max(output, 1)
            predictions.append(predicted_label.item())

    # 6. 建立並儲存提交檔案
    submission_df['label'] = predictions
    
    if output_path is None:
        submission_dir = os.path.join(cfg.MAIN_PROJECT_ROOT, "submissions")
        os.makedirs(submission_dir, exist_ok=True)
        output_path = os.path.join(submission_dir, f"submission_General_ML_{experiment_name}.csv")
        print(f"未指定輸出路徑，將自動儲存至: {output_path}")

    submission_df.to_csv(output_path, index=False)
    print(f"\n預測完成！結果已儲存至: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用訓練好的 MLP 模型對測試集進行預測。")
    parser.add_argument(
        "--experiment", 
        type=str, 
        required=True,
        help="要用於預測的實驗名稱 (定義於 mlp_config.py)。",
        choices=list(cfg.EXPERIMENTS.keys())
    )
    parser.add_argument(
        "--output_csv", 
        type=str, 
        default=None, 
        help="輸出的預測結果 CSV 檔案路徑。如果未提供，將自動生成檔名。"
    )
    args = parser.parse_args()
    
    predict(args.experiment, args.output_csv) 