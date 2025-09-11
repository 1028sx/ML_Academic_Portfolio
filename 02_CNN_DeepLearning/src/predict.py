import torch
import pandas as pd
import numpy as np
import os
import torchaudio
from tqdm import tqdm
import argparse
import json

# 導入重構後的核心模組和新的設定中心
from models import get_model
import cnn_config as cfg
from utils import AudioDataset # 我們需要它來進行單一樣本的預處理
from data_pipeline import get_dataloaders # 為了自動產生 scaler 而導入

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_single_audio(file_path, scaler, pipeline_config):
    """
    對單一音訊檔案進行預處理，返回標準化後的梅爾頻譜圖張量。
    這個函式借鑒了 AudioDataset 的邏輯，但專為單一樣本預測設計。
    """
    try:
        waveform, sr = torchaudio.load(file_path)
    except Exception as e:
        print(f"載入檔案錯誤 {file_path}: {e}")
        return None

    # 重採樣
    if sr != cfg.AUDIO['sample_rate']:
        resampler = torchaudio.transforms.Resample(sr, cfg.AUDIO['sample_rate'])
        waveform = resampler(waveform)

    # VAD 處理
    if pipeline_config.get('use_vad', False):
        # [REFACTOR] 永久性切換至 torchaudio.functional.vad
        waveform = torchaudio.functional.vad(waveform, sample_rate=cfg.AUDIO['sample_rate'])
        if waveform.shape[1] == 0:
            print(f"警告：VAD 處理後檔案 {file_path} 長度為 0。")
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
    
    # 返回符合模型輸入的 4D 張量
    return standardized_spec_tensor.unsqueeze(0).unsqueeze(0)

def predict(experiment_name, output_path=None):
    """
    主預測流程。
    現在它會自動從模型旁邊的 .json 設定檔中讀取所有必要的設定。
    """
    print(f"使用設備: {DEVICE}")

    # 1. 根據實驗名稱建構模型路徑
    model_save_name = f"{experiment_name}.pth"
    model_path = os.path.join(cfg.MODEL_SAVE_DIR, model_save_name)
    
    if not os.path.exists(model_path):
        print(f"錯誤：在 '{cfg.MODEL_SAVE_DIR}' 中找不到模型檔案 '{model_save_name}'。")
        return

    # 2. 載入模型和關聯的設定檔
    print(f"正在載入模型: {model_path}")
    
    config_path = model_path.replace('.pth', '.json')
    if not os.path.exists(config_path):
        print(f"錯誤：找不到模型 '{model_path}' 對應的設定檔 '{config_path}'。")
        print("請確保與模型權重 (.pth) 同目錄下有一個同名的 .json 設定檔。")
        return
        
    with open(config_path, 'r') as f:
        exp_config = json.load(f)
    print(f"成功載入實驗 '{experiment_name}' 的設定檔。")

    model_blueprint = exp_config['model_blueprint']
    data_pipeline_config = exp_config['data_pipeline']
    
    # 新增：找出管道名稱以載入正確的標準化器
    pipeline_name = "default" # 備用名稱
    for name, config_dict in cfg.DATA_PIPELINES.items():
        if config_dict == data_pipeline_config:
            pipeline_name = name
            break
    print(f"根據設定檔，找到對應的資料管道名稱: {pipeline_name}")
    
    # 準備 data_dims
    data_dims = {
        'input_h': cfg.AUDIO['n_mels'],
        'input_w': data_pipeline_config['target_len'],
        'output_dim': cfg.AUDIO['num_classes']
    }

    # 合併模型藍圖和資料維度，以符合 get_model 的單一參數要求
    full_model_config = {**model_blueprint, **data_dims}

    model = get_model(full_model_config)
    if model is None:
        print(f"無法載入模型，請檢查 models.py")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # 2. 載入標準化器 (Scaler)，如果不存在則自動產生
    scaler_path = os.path.join(cfg.PROCESSED_DATA_DIR, f'scaler_{pipeline_name}.pt')
    print(f"正在檢查標準化器: {scaler_path}")
    try:
        # torch.load 有 weights_only 參數來減少安全風險
        # 我們應該明確設定它。由於我們正在載入張量字典，
        # 非模型的 state_dict，False 是適當的，但明確指定是好的
        scaler = torch.load(scaler_path, weights_only=False)
        print("標準化器已存在，直接載入。")
    except FileNotFoundError:
        print(f"警告：找不到標準化器檔案 '{scaler_path}'。")
        print("將嘗試自動產生...")
        try:
            # 觸發 get_dataloaders 來產生 scaler
            # 我們只需要它執行產生 scaler 的部分，所以不需要接收其回傳值
            get_dataloaders(exp_config['experiment_name'], force_recalc_scaler=True)
            print("Scaler 檔案產生成功，重新載入...")
            scaler = torch.load(scaler_path, weights_only=False)
        except Exception as e:
            print(f"自動產生 Scaler 時發生嚴重錯誤: {e}")
            print(f"請檢查 data_pipeline.py 以及實驗 '{exp_config['experiment_name']}' 的設定。")
            return
            
    # 3. 讀取提交範本
    submission_df = pd.read_csv(cfg.SUBMISSION_CSV_PATH)
    predictions = []

    print("開始預測...")
    # 4. 迭代測試檔案並進行預測
    test_data_dir = cfg.TEST_SET_DIR
    with torch.no_grad():
        for wav_filename in tqdm(submission_df['wav'], desc="預測進度"):
            file_path = os.path.join(test_data_dir, wav_filename)
            
            if not os.path.exists(file_path):
                print(f"警告: 找不到檔案 {file_path}，將使用預設標籤 0")
                predictions.append(0)
                continue
                
            # 使用新的預處理函式和從檔案載入的設定
            input_tensor = preprocess_single_audio(file_path, scaler, data_pipeline_config)
            if input_tensor is None:
                predictions.append(0)
                continue
                
            input_tensor = input_tensor.to(DEVICE)
            
            output = model(input_tensor)
            _, predicted_label = torch.max(output, 1)
            predictions.append(predicted_label.item())

    # 5. 建立並儲存提交檔案
    submission_df['label'] = predictions
    
    if output_path is None:
        submission_dir = os.path.join(cfg.MAIN_PROJECT_ROOT, "submissions")
        os.makedirs(submission_dir, exist_ok=True)
        output_path = os.path.join(submission_dir, f"submission_CNN_Audio_Classification_{experiment_name}.csv")
        print(f"未指定輸出路徑，將自動儲存至: {output_path}")

    submission_df.to_csv(output_path, index=False)
    print(f"\n預測完成！結果已儲存至: {output_path}")
    
def save_experiment_config(experiment_name, model_save_path):
    """
    一個輔助函式，在訓練完成後，將該次實驗的設定儲存為 JSON 檔案，
    與模型權重放在一起。
    """
    config_save_path = model_save_path.replace('.pth', '.json')
    
    exp_config_to_save = {
        'experiment_name': experiment_name,
        'model_blueprint': cfg.EXPERIMENTS[experiment_name]['model_blueprint'],
        'data_pipeline': cfg.EXPERIMENTS[experiment_name]['data_pipeline']
    }
    
    with open(config_save_path, 'w') as f:
        json.dump(exp_config_to_save, f, indent=4)
    print(f"實驗設定已儲存至: {config_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用訓練好的模型對測試集進行預測。")
    parser.add_argument("--experiment", type=str, required=True, help="要用於預測的實驗名稱 (例如 V6_CNN_Baseline_Benchmark)。")
    parser.add_argument("--output_csv", type=str, default=None, help="輸出的預測結果 CSV 檔案路徑。如果未提供，將自動生成檔名。")
    args = parser.parse_args()
    
    predict(args.experiment, args.output_csv) 