"""
MLP 基礎模組的設定中心。

此檔案作為展示基礎機器學習概念的 MLP 模型的所有設定的單一來源。
學術作品集版本 - 針對研究展示進行優化。
"""
import os

# ==============================================================================
#  路徑和目錄設定
# ==============================================================================
# --- 根目錄路徑 ---
def get_mlp_project_root():
    """返回 '01_MLP_Foundation' 目錄的絕對路徑。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_main_project_root():
    """返回主專案根目錄 (ML_Academic_Portfolio) 的絕對路徑。"""
    return os.path.dirname(get_mlp_project_root())

MLP_PROJECT_ROOT = get_mlp_project_root()
MAIN_PROJECT_ROOT = get_main_project_root()

# --- 原始數據路徑（位於專案根目錄） ---
DATA_DIR = os.path.join(MAIN_PROJECT_ROOT, '04_Data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
SAMPLE_DATA_DIR = os.path.join(DATA_DIR, 'sample_data')
TRAIN_CSV_PATH = os.path.join(DATA_DIR, 'train_cleaned.csv')

# --- 輸出路徑（MLP 專案特定） ---
OUTPUT_DIR = os.path.join(MLP_PROJECT_ROOT, 'output')
MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, 'models')
PLOT_SAVE_DIR = os.path.join(OUTPUT_DIR, 'plots')
LOG_SAVE_DIR = os.path.join(OUTPUT_DIR, 'logs')

# ==============================================================================
#  音頻處理參數
# ==============================================================================
AUDIO = {
    'sample_rate': 16000,
    'n_fft': 1024,
    'hop_length': 160,
    'n_mels': 128,
    'num_classes': 8,
}

# ==============================================================================
#  數據處理管道
# ==============================================================================
# 對於 MLP，我們使用單一、標準的管道來將頻譜圖扁平化。
DATA_PIPELINES = {
    'P1_MLP_Standard': {
        'target_len': 215, # 頻譜圖的標準長度
        'use_vad': False,
        'augmentation_mode': None,
    }
}

# ==============================================================================
#  模型架構藍圖
# ==============================================================================
MODEL_BLUEPRINTS = {
    'V1_Baseline': {
        'model_class': 'V1_Baseline',
        'data_pipeline': DATA_PIPELINES['P1_MLP_Standard'],
    },
    'V2_Deeper': {
        'model_class': 'V2_Deeper',
        'data_pipeline': DATA_PIPELINES['P1_MLP_Standard'],
    },
    'V3_AugLeaky': {
        'model_class': 'V3_AugLeaky',
        'data_pipeline': DATA_PIPELINES['P1_MLP_Standard'],
    },
     'V4_AdamW': {
        'model_class': 'V4_AdamW',
        'data_pipeline': DATA_PIPELINES['P1_MLP_Standard'],
    },
     'V5_Pooling': {
        'model_class': 'V5_Pooling',
        'data_pipeline': DATA_PIPELINES['P1_MLP_Standard'],
    },
}

# ==============================================================================
#  實驗配置
# ==============================================================================
EXPERIMENTS = {
    'V1_Baseline': {
        'model_blueprint': MODEL_BLUEPRINTS['V1_Baseline'],
        'data_pipeline': DATA_PIPELINES['P1_MLP_Standard'],
        'optimizer': 'Adam', 'lr': 1e-3, 'wd': 1e-4,
        'epochs': 50, 'patience': 10, 'batch_size': 64,
    },
    'V2_Deeper': {
        'model_blueprint': MODEL_BLUEPRINTS['V2_Deeper'],
        'data_pipeline': DATA_PIPELINES['P1_MLP_Standard'],
        'optimizer': 'Adam', 'lr': 1e-3, 'wd': 1e-4,
        'epochs': 50, 'patience': 10, 'batch_size': 64,
    },
    'V3_AugLeaky': {
        'model_blueprint': MODEL_BLUEPRINTS['V3_AugLeaky'],
        'data_pipeline': DATA_PIPELINES['P1_MLP_Standard'],
        'optimizer': 'Adam', 'lr': 1e-3, 'wd': 1e-4,
        'epochs': 50, 'patience': 10, 'batch_size': 64,
    },
    'V4_AdamW': {
        'model_blueprint': MODEL_BLUEPRINTS['V4_AdamW'],
        'data_pipeline': DATA_PIPELINES['P1_MLP_Standard'],
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-3,
        'epochs': 50, 'patience': 10, 'batch_size': 64,
    },
    'V5_Pooling': {
        'model_blueprint': MODEL_BLUEPRINTS['V5_Pooling'],
        'data_pipeline': DATA_PIPELINES['P1_MLP_Standard'],
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-3,
        'epochs': 50, 'patience': 10, 'batch_size': 64,
    },
}

def get_config(experiment_name):
    """
    透過合併實驗的特定設定與其模型藍圖和數據管道來獲取給定實驗的最終配置。
    """
    if experiment_name not in EXPERIMENTS:
        raise ValueError(f"Experiment '{experiment_name}' not found in configuration.")

    exp_config = EXPERIMENTS[experiment_name]
    model_blueprint = exp_config['model_blueprint']
    data_pipeline = exp_config['data_pipeline']

    # 從基本實驗設定開始
    final_config = {
        'experiment_name': experiment_name,
        'train': {
            'optimizer': exp_config.get('optimizer', 'Adam'),
            'lr': exp_config.get('lr', 1e-3),
            'wd': exp_config.get('wd', 0),
            'epochs': exp_config.get('epochs', 50),
            'patience': exp_config.get('patience', 10),
            'batch_size': exp_config.get('batch_size', 64),
            'scheduler': exp_config.get('scheduler'),
            'gradient_clip_val': exp_config.get('gradient_clip_val'),
        },
        'model': {
            'model_class': model_blueprint['model_class'],
        },
        'data': {
            **data_pipeline,
            'train_csv_path': TRAIN_CSV_PATH,
            'processed_data_dir': PROCESSED_DATA_DIR, # 確保路徑被傳遞
        },
        'audio': {**AUDIO},

        # 為方便起見增加輸出路徑
        'log_save_dir': LOG_SAVE_DIR,
        'model_save_dir': MODEL_SAVE_DIR,
        'plot_save_dir': PLOT_SAVE_DIR,
    }
    
    # 將輸入維度加入模型配置，這些是從資料管道推導出的
    # 對於 MLP，輸入是扁平化的。我們提供所有變體以增加靈活性。
    final_config['model']['input_dim'] = AUDIO['n_mels'] * data_pipeline['target_len']
    final_config['model']['input_h'] = AUDIO['n_mels']
    final_config['model']['input_w'] = data_pipeline['target_len']
    final_config['model']['output_dim'] = AUDIO['num_classes']

    return final_config 