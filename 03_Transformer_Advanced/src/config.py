"""
進階 Transformer 模組的設定中心。

此檔案集中管理資料處理、模型架構和實驗的設定，
確保一致性並簡化新實驗的建立。
學術作品集版本 - 展示最先進的注意力機制。
"""
import os
import torch

# ==============================================================================
#  路徑和目錄設定
# ==============================================================================
def get_transformer_project_root():
    """返回 '03_Transformer_Advanced' 目錄的絕對路徑。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_main_project_root():
    """返回主專案根目錄 (ML_Academic_Portfolio) 的絕對路徑。"""
    return os.path.dirname(get_transformer_project_root())

TRANSFORMER_PROJECT_ROOT = get_transformer_project_root()
MAIN_PROJECT_ROOT = get_main_project_root()

# --- 來源資料路徑 ---
DATA_DIR = os.path.join(MAIN_PROJECT_ROOT, '04_Data', 'voice_dataset', 'train_set')
SAMPLE_DATA_DIR = os.path.join(MAIN_PROJECT_ROOT, '04_Data', 'sample_data')
TRAIN_CSV_PATH = os.path.join(MAIN_PROJECT_ROOT, '04_Data', 'train_cleaned.csv')

# --- 輸出路徑 ---
OUTPUT_DIR = os.path.join(TRANSFORMER_PROJECT_ROOT, 'output')
MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, 'models')
PLOT_SAVE_DIR = os.path.join(OUTPUT_DIR, 'plots')
LOG_SAVE_DIR = os.path.join(OUTPUT_DIR, 'logs')

# ==============================================================================
#  全域設定
# ==============================================================================
AUDIO_SETTINGS = {
    'sample_rate': 16000,
    'n_fft': 1024,
    'hop_length': 256,
    'n_mels': 128,
    'duration': 5, # 秒
    'target_len': int((16000 * 5) / 256) + 1, # 313
    'num_classes': 8
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
#  資料處理管道
# ==============================================================================
DATA_PIPELINES = {
    'P1_Baseline': {
        'augmentation_config': None,
    },
    'P2_SpecAug_Aggressive': {
        'augmentation_config': {
            'type': 'spec_augment',
            'freq_mask_param': 27,
            'time_mask_param': 70,
            'num_freq_masks': 2,
            'num_time_masks': 2
        }
    },
    'P3_SpecAug_Gentle': {
        'augmentation_config': {
            'type': 'spec_augment',
            'freq_mask_param': 10,
            'time_mask_param': 30,
            'num_freq_masks': 1,
            'num_time_masks': 1
        }
    },
    
    # --- SpecMix 增強管道 (2024年最新技術) ---
    'P4_SpecMix': {
        'augmentation_config': {
            'type': 'specmix',
            'use_specmix': True,
            'use_mixspeech': False,
            'use_noise': True,
            'use_time_shift': True,
            'specmix_prob': 0.5,
            'noise_factor': 0.002,
            'shift_limit': 0.1
        }
    },
    'P5_MixSpeech': {
        'augmentation_config': {
            'type': 'mixspeech',
            'use_specmix': False,
            'use_mixspeech': True,
            'use_noise': True,
            'use_time_shift': True,
            'specmix_prob': 0.5,
            'noise_factor': 0.002,
            'shift_limit': 0.1
        }
    },
    'P6_Combined_SpecMix': {
        'augmentation_config': {
            'type': 'specmix_combined',
            'use_specmix': True,
            'use_mixspeech': False,
            'use_noise': True,
            'use_time_shift': True,
            'specmix_prob': 0.7,  # Higher probability for combined strategy
            'noise_factor': 0.003,
            'shift_limit': 0.15
        }
    }
}

# ==============================================================================
#  模型架構藍圖
# ==============================================================================
MODEL_BLUEPRINTS = {
    'AudioTransformer_v1': {
        'model_class': 'AudioTransformer',
        'd_model': AUDIO_SETTINGS['n_mels'],
        'nhead': 8,
        'num_encoder_layers': 6,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'classification_mode': 'mean' # 'mean' or 'cls'
    },
    'AudioTransformer_v2_deeper': {
        'model_class': 'AudioTransformer',
        'd_model': AUDIO_SETTINGS['n_mels'],
        'nhead': 8,
        'num_encoder_layers': 12, # 從 6 增加到 12
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'classification_mode': 'mean'
    },
    'AudioTransformer_v3_wider': {
        'model_class': 'AudioTransformer',
        'd_model': AUDIO_SETTINGS['n_mels'],
        'nhead': 8,
        'num_encoder_layers': 6,
        'dim_feedforward': 4096, # 從 2048 增加到 4096
        'dropout': 0.1,
        'classification_mode': 'mean'
    },
    'AudioTransformer_v4_wider_regularized': {
        'model_class': 'AudioTransformer',
        'd_model': AUDIO_SETTINGS['n_mels'],
        'nhead': 8,
        'num_encoder_layers': 6,
        'dim_feedforward': 4096,
        'dropout': 0.25, # 增加 dropout 以進行正則化
        'classification_mode': 'mean'
    },
    'AudioTransformer_v5_balanced_regularization': {
        'model_class': 'AudioTransformer',
        'd_model': AUDIO_SETTINGS['n_mels'],
        'nhead': 8,
        'num_encoder_layers': 6,
        'dim_feedforward': 4096,
        'dropout': 0.2, # 平衡的 dropout
        'classification_mode': 'mean'
    },
    
    # --- SpecMix 優化架構 (基於"寬勝於深"發現) ---
    'AudioTransformer_v6_specmix_wider': {
        'model_class': 'AudioTransformer',
        'd_model': AUDIO_SETTINGS['n_mels'],
        'nhead': 8,
        'num_encoder_layers': 6,
        'dim_feedforward': 6144,  # 進一步增寬：4096 -> 6144
        'dropout': 0.3,  # 配合SpecMix的正則化
        'classification_mode': 'mean'
    },
    'AudioTransformer_v7_ultra_wide': {
        'model_class': 'AudioTransformer',
        'd_model': AUDIO_SETTINGS['n_mels'],
        'nhead': 8,
        'num_encoder_layers': 6,
        'dim_feedforward': 8192,  # 極寬架構測試
        'dropout': 0.4,  # 更強正則化匹配更寬架構
        'classification_mode': 'mean'
    },
    'AudioTransformer_v8_specmix_regularized': {
        'model_class': 'AudioTransformer',
        'd_model': AUDIO_SETTINGS['n_mels'],
        'nhead': 8,
        'num_encoder_layers': 6,
        'dim_feedforward': 4096,
        'dropout': 0.5,  # 重度正則化配合SpecMix
        'classification_mode': 'mean'
    }
}


# ==============================================================================
#  實驗配置
# ==============================================================================
EXPERIMENTS = {
    'V1_Transformer_Baseline': {
        'model_blueprint': MODEL_BLUEPRINTS['AudioTransformer_v1'],
        'data_pipeline': DATA_PIPELINES['P1_Baseline'],
        'epochs': 100, 'patience': 15, 'batch_size': 32,
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-5,
        'scheduler': None
    },
    'V2_Transformer_SpecAugment': {
        'model_blueprint': MODEL_BLUEPRINTS['AudioTransformer_v1'],
        'data_pipeline': DATA_PIPELINES['P2_SpecAug_Aggressive'],
        'epochs': 150, 'patience': 20, 'batch_size': 32,
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-5,
        'scheduler': None
    },
    'V3_SpecAugment_Gentle': {
        'model_blueprint': MODEL_BLUEPRINTS['AudioTransformer_v1'],
        'data_pipeline': DATA_PIPELINES['P3_SpecAug_Gentle'],
        'epochs': 150, 'patience': 20, 'batch_size': 32,
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-5,
        'scheduler': None
    },
    'V4_Transformer_ClsToken': {
        'model_blueprint': {
            **MODEL_BLUEPRINTS['AudioTransformer_v1'], # 繼承基本設定
            'classification_mode': 'cls' # 覆寫分類模式
        },
        'data_pipeline': DATA_PIPELINES['P1_Baseline'], # 從無增強開始
        'epochs': 100, 'patience': 15, 'batch_size': 32,
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-5,
        'scheduler': None
    },
    'V5_Transformer_OneCycleLR': {
        'model_blueprint': MODEL_BLUEPRINTS['AudioTransformer_v1'],
        'data_pipeline': DATA_PIPELINES['P3_SpecAug_Gentle'],
        'epochs': 150, 'patience': 25, 'batch_size': 32,
        'optimizer': 'AdamW', 'lr': None, 'wd': 1e-5, # 明確顯示學習率由排程器管理
        'scheduler': {
            'name': 'OneCycleLR',
            'max_lr': 1e-3
        }
    },
    'V6_Transformer_OneCycleLR_LowerLR': {
        'model_blueprint': MODEL_BLUEPRINTS['AudioTransformer_v1'],
        'data_pipeline': DATA_PIPELINES['P3_SpecAug_Gentle'],
        'epochs': 150, 'patience': 25, 'batch_size': 32,
        'optimizer': 'AdamW', 'lr': None, 'wd': 1e-5,
        'scheduler': {
            'name': 'OneCycleLR',
            'max_lr': 1e-4 # 降低學習率
        }
    },
    'V7_LabelSmoothing': {
        'model_blueprint': MODEL_BLUEPRINTS['AudioTransformer_v1'],
        'data_pipeline': DATA_PIPELINES['P3_SpecAug_Gentle'],
        'epochs': 150, 'patience': 20, 'batch_size': 32,
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-5,
        'scheduler': None,
        'label_smoothing': 0.1 # 增加標籤平滑
    },
    'V8_Deeper_Transformer': {
        'model_blueprint': MODEL_BLUEPRINTS['AudioTransformer_v2_deeper'],
        'data_pipeline': DATA_PIPELINES['P3_SpecAug_Gentle'],
        'epochs': 200, 'patience': 25, 'batch_size': 32, # 為更深的模型增加週期數/耐心度
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-5,
        'scheduler': None
    },
    'V9_Wider_Transformer': {
        'model_blueprint': MODEL_BLUEPRINTS['AudioTransformer_v3_wider'],
        'data_pipeline': DATA_PIPELINES['P3_SpecAug_Gentle'],
        'epochs': 200, 'patience': 25, 'batch_size': 32,
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-5,
        'scheduler': None
    },
    'V10_Wider_LabelSmoothing': {
        'model_blueprint': MODEL_BLUEPRINTS['AudioTransformer_v3_wider'],
        'data_pipeline': DATA_PIPELINES['P3_SpecAug_Gentle'],
        'epochs': 200, 'patience': 25, 'batch_size': 32,
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-5,
        'scheduler': None,
        'label_smoothing': 0.1 # 在更寬的模型上重新引入標籤平滑
    },
    'V11_Regularized_Wider_Transformer': {
        'model_blueprint': MODEL_BLUEPRINTS['AudioTransformer_v4_wider_regularized'],
        'data_pipeline': DATA_PIPELINES['P3_SpecAug_Gentle'],
        'epochs': 200, 'patience': 30, # 稍微增加耐心度
        'batch_size': 32,
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-2, # 顯著增加權重衰減
        'scheduler': None
    },
    'V12_Balanced_Regularization': {
        'model_blueprint': MODEL_BLUEPRINTS['AudioTransformer_v5_balanced_regularization'],
        'data_pipeline': DATA_PIPELINES['P3_SpecAug_Gentle'],
        'epochs': 200, 'patience': 30,
        'batch_size': 32,
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-4, # 平衡的權重衰減（V9 的 10 倍，但比 V11 少 100 倍）
        'scheduler': None
    },
    'V13_BalancedReg_OneCycleLR': {
        'model_blueprint': MODEL_BLUEPRINTS['AudioTransformer_v5_balanced_regularization'],
        'data_pipeline': DATA_PIPELINES['P3_SpecAug_Gentle'],
        'epochs': 200, 'patience': 30,
        'batch_size': 32,
        'optimizer': 'AdamW', 'lr': None, 'wd': 1e-4, # 學習率由排程器管理
        'scheduler': {
            'name': 'OneCycleLR',
            'max_lr': 5e-4 # 更高的峰值學習率以逃脫局部最小值
        }
    },
    
    # ==============================================================================
    #  V14-V18: SpecMix 系列實驗 (2024年最新技術)
    # ==============================================================================
    'V14_SpecMix_Wider_Transformer': {
        'description': "SpecMix + Wider架構 - 最佳組合測試",
        'model_blueprint': MODEL_BLUEPRINTS['AudioTransformer_v6_specmix_wider'],
        'data_pipeline': DATA_PIPELINES['P4_SpecMix'],
        'epochs': 150, 'patience': 25, 'batch_size': 32,
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-2,  # Heavy regularization
        'scheduler': None,
        'label_smoothing': 0.1
    },
    
    'V15_Ultra_Wide_SpecMix': {
        'description': "極寬架構 + SpecMix - 測試寬度極限",
        'model_blueprint': MODEL_BLUEPRINTS['AudioTransformer_v7_ultra_wide'],
        'data_pipeline': DATA_PIPELINES['P4_SpecMix'],
        'epochs': 200, 'patience': 30, 'batch_size': 24,  # Smaller batch for memory
        'optimizer': 'AdamW', 'lr': 8e-5, 'wd': 2e-2,  # Very heavy regularization
        'scheduler': {'type': 'CosineAnnealingLR', 'params': {'T_max': 200}},
        'label_smoothing': 0.15
    },
    
    'V16_Combined_SpecMix_Heavy_Reg': {
        'description': "組合SpecMix + 重度正則化",
        'model_blueprint': MODEL_BLUEPRINTS['AudioTransformer_v8_specmix_regularized'],
        'data_pipeline': DATA_PIPELINES['P6_Combined_SpecMix'],
        'epochs': 150, 'patience': 25, 'batch_size': 32,
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 3e-2,
        'scheduler': None,
        'label_smoothing': 0.1
    },
    
    'V17_MixSpeech_Alternative': {
        'description': "MixSpeech替代方案 - 語音專用優化",
        'model_blueprint': MODEL_BLUEPRINTS['AudioTransformer_v6_specmix_wider'],
        'data_pipeline': DATA_PIPELINES['P5_MixSpeech'],
        'epochs': 150, 'patience': 25, 'batch_size': 32,
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-2,
        'scheduler': None,
        'label_smoothing': 0.1
    },
    
    # --- 快速驗證實驗 ---
    'Debug_SpecMix_Transformer_Quick': {
        'description': "2-epoch SpecMix驗證測試",
        'model_blueprint': MODEL_BLUEPRINTS['AudioTransformer_v6_specmix_wider'],
        'data_pipeline': DATA_PIPELINES['P4_SpecMix'],
        'epochs': 2, 'patience': 10, 'batch_size': 16,  # Small batch for quick test
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-2,
        'scheduler': None
    }
}

# ==============================================================================
#  工具函式
# ==============================================================================
def get_config(experiment_name: str) -> dict:
    """
    檢索並結合給定實驗的配置。
    """
    if experiment_name not in EXPERIMENTS:
        raise ValueError(f"在 transformer_config.py 中找不到名為 '{experiment_name}' 的實驗。")
    
    exp_conf = EXPERIMENTS[experiment_name]
    
    # 將所有部分結合成單一配置字典供訓練器使用
    config = {
        'experiment_name': experiment_name,
        'audio_settings': AUDIO_SETTINGS,
        'device': DEVICE,
        **exp_conf,
    }
    return config

print(f"--- Transformer 設定檔載入 ---")
print(f"定義了 {len(EXPERIMENTS)} 個實驗。")
print(f"使用設備: {DEVICE}")
print(f"-----------------------------") 