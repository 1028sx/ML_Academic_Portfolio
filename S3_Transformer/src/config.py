import os
import torch

# 路徑和目錄設定
def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_main_project_root():
    return os.path.dirname(get_project_root())

PROJECT_ROOT = get_project_root()
MAIN_PROJECT_ROOT = get_main_project_root()

# 來源路徑
DATA_DIR = os.path.join(MAIN_PROJECT_ROOT, 'data', 'voice_dataset', 'train_set')
SAMPLE_DATA_DIR = os.path.join(MAIN_PROJECT_ROOT, 'data', 'sample_data')
TRAIN_CSV_PATH = os.path.join(MAIN_PROJECT_ROOT, 'data', 'train_cleaned.csv')

# 輸出路徑
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, 'models')
PLOT_SAVE_DIR = os.path.join(OUTPUT_DIR, 'plots')
LOG_SAVE_DIR = os.path.join(OUTPUT_DIR, 'logs')

# 全域設定
AUDIO = {
    'sample_rate': 16000,
    'n_fft': 1024,
    'hop_length': 256,
    'n_mels': 128,
    'duration': 5, # 秒
    'target_len': int((16000 * 5) / 256) + 1, # 313
    'num_classes': 8
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 資料處理管道
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

}

# 模型架構
MODEL_BLUEPRINTS = {
    'AudioTransformer_v1': {
        'model_class': 'AudioTransformer',
        'd_model': AUDIO['n_mels'],
        'nhead': 8,
        'num_encoder_layers': 6,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'classification_mode': 'mean',
        'num_classes': AUDIO['num_classes'],
        'max_len': AUDIO['target_len']
    },
    'AudioTransformer_v2_deeper': {
        'model_class': 'AudioTransformer',
        'd_model': AUDIO['n_mels'],
        'nhead': 8,
        'num_encoder_layers': 12, # 從 6 增加到 12
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'classification_mode': 'mean',
        'num_classes': AUDIO['num_classes'],
        'max_len': AUDIO['target_len']
    },
    'AudioTransformer_v3_wider': {
        'model_class': 'AudioTransformer',
        'd_model': AUDIO['n_mels'],
        'nhead': 8,
        'num_encoder_layers': 6,
        'dim_feedforward': 4096, # 從 2048 增加到 4096
        'dropout': 0.1,
        'classification_mode': 'mean',
        'num_classes': AUDIO['num_classes'],
        'max_len': AUDIO['target_len']
    },
    'AudioTransformer_v4_wider_regularized': {
        'model_class': 'AudioTransformer',
        'd_model': AUDIO['n_mels'],
        'nhead': 8,
        'num_encoder_layers': 6,
        'dim_feedforward': 4096,
        'dropout': 0.25, # 增加 dropout 以進行正則化
        'classification_mode': 'mean',
        'num_classes': AUDIO['num_classes'],
        'max_len': AUDIO['target_len']
    },
    'AudioTransformer_v5_balanced_regularization': {
        'model_class': 'AudioTransformer',
        'd_model': AUDIO['n_mels'],
        'nhead': 8,
        'num_encoder_layers': 6,
        'dim_feedforward': 4096,
        'dropout': 0.2, # 平衡的 dropout
        'classification_mode': 'mean',
        'num_classes': AUDIO['num_classes'],
        'max_len': AUDIO['target_len']
    },

    'AudioTransformer_v7_ultra_wide': {
        'model_class': 'AudioTransformer',
        'd_model': AUDIO['n_mels'],
        'nhead': 8,
        'num_encoder_layers': 6,
        'dim_feedforward': 8192,  # 極寬架構測試
        'dropout': 0.4,  # 更強正則化匹配更寬架構
        'classification_mode': 'mean',
        'num_classes': AUDIO['num_classes'],
        'max_len': AUDIO['target_len']
    },
}


# 實驗配置
EXPERIMENTS = {
    'V1_Transformer_Baseline': {
        'description': "Transformer Encoder基礎架構",
        'model_blueprint': MODEL_BLUEPRINTS['AudioTransformer_v1'],
        'data_pipeline': DATA_PIPELINES['P1_Baseline'],
        'epochs': 100, 'patience': 15, 'batch_size': 32,
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-5,
        'scheduler': None
    },
    'V2_Transformer_SpecAugment': {
        'description': "V1+激進SpecAugment",
        'model_blueprint': MODEL_BLUEPRINTS['AudioTransformer_v1'],
        'data_pipeline': DATA_PIPELINES['P2_SpecAug_Aggressive'],
        'epochs': 150, 'patience': 20, 'batch_size': 32,
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-5,
        'scheduler': None
    },
    'V3_SpecAugment_Gentle': {
        'description': "V1+溫和SpecAugment",
        'model_blueprint': MODEL_BLUEPRINTS['AudioTransformer_v1'],
        'data_pipeline': DATA_PIPELINES['P3_SpecAug_Gentle'],
        'epochs': 150, 'patience': 20, 'batch_size': 32,
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-5,
        'scheduler': None
    },
    'V4_Transformer_ClsToken': {
        'description': "V1+CLS token分類",
        'model_blueprint': {
            **MODEL_BLUEPRINTS['AudioTransformer_v1'],
            'classification_mode': 'cls' # 覆寫分類模式
        },
        'data_pipeline': DATA_PIPELINES['P1_Baseline'],
        'epochs': 100, 'patience': 15, 'batch_size': 32,
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-5,
        'scheduler': None
    },
    'V5_Transformer_OneCycleLR': {
        'description': "V3+OneCycleLR學習率策略",
        'model_blueprint': MODEL_BLUEPRINTS['AudioTransformer_v1'],
        'data_pipeline': DATA_PIPELINES['P3_SpecAug_Gentle'],
        'epochs': 150, 'patience': 25, 'batch_size': 32,
        'optimizer': 'AdamW', 'lr': None, 'wd': 1e-5,
        'scheduler': {
            'name': 'OneCycleLR',
            'max_lr': 1e-3
        }
    },
    'V6_Transformer_OneCycleLR_LowerLR': {
        'description': "V5+降低學習率",
        'model_blueprint': MODEL_BLUEPRINTS['AudioTransformer_v1'],
        'data_pipeline': DATA_PIPELINES['P3_SpecAug_Gentle'],
        'epochs': 150, 'patience': 25, 'batch_size': 32,
        'optimizer': 'AdamW', 'lr': None, 'wd': 1e-5,
        'scheduler': {
            'name': 'OneCycleLR',
            'max_lr': 1e-4
        }
    },
    'V7_LabelSmoothing': {
        'description': "V3+標籤平滑正規化",
        'model_blueprint': MODEL_BLUEPRINTS['AudioTransformer_v1'],
        'data_pipeline': DATA_PIPELINES['P3_SpecAug_Gentle'],
        'epochs': 150, 'patience': 20, 'batch_size': 32,
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-5,
        'scheduler': None,
        'label_smoothing': 0.1
    },
    'V8_Deeper_Transformer': {
        'description': "更深的12層Transformer",
        'model_blueprint': MODEL_BLUEPRINTS['AudioTransformer_v2_deeper'],
        'data_pipeline': DATA_PIPELINES['P3_SpecAug_Gentle'],
        'epochs': 200, 'patience': 25, 'batch_size': 32,
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-5,
        'scheduler': None
    },
    'V9_Wider_Transformer': {
        'description': "更寬的4096維 feedforward",
        'model_blueprint': MODEL_BLUEPRINTS['AudioTransformer_v3_wider'],
        'data_pipeline': DATA_PIPELINES['P3_SpecAug_Gentle'],
        'epochs': 200, 'patience': 25, 'batch_size': 32,
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-5,
        'scheduler': None
    },
    'V10_Wider_LabelSmoothing': {
        'description': "V9+標籤平滑",
        'model_blueprint': MODEL_BLUEPRINTS['AudioTransformer_v3_wider'],
        'data_pipeline': DATA_PIPELINES['P3_SpecAug_Gentle'],
        'epochs': 200, 'patience': 25, 'batch_size': 32,
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-5,
        'scheduler': None,
        'label_smoothing': 0.1
    },
    'V11_Regularized_Wider_Transformer': {
        'description': "V9+重度正規化",
        'model_blueprint': MODEL_BLUEPRINTS['AudioTransformer_v4_wider_regularized'],
        'data_pipeline': DATA_PIPELINES['P3_SpecAug_Gentle'],
        'epochs': 200, 'patience': 30,
        'batch_size': 32,
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-2,
        'scheduler': None
    },
    'V12_Balanced_Regularization': {
        'description': "平衡正規化策略",
        'model_blueprint': MODEL_BLUEPRINTS['AudioTransformer_v5_balanced_regularization'],
        'data_pipeline': DATA_PIPELINES['P3_SpecAug_Gentle'],
        'epochs': 200, 'patience': 30,
        'batch_size': 32,
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-4,
        'scheduler': None
    },
    'V13_BalancedReg_OneCycleLR': {
        'description': "V12+OneCycleLR策略",
        'model_blueprint': MODEL_BLUEPRINTS['AudioTransformer_v5_balanced_regularization'],
        'data_pipeline': DATA_PIPELINES['P3_SpecAug_Gentle'],
        'epochs': 200, 'patience': 30,
        'batch_size': 32,
        'optimizer': 'AdamW', 'lr': None, 'wd': 1e-4,
        'scheduler': {
            'name': 'OneCycleLR',
            'max_lr': 5e-4
        }
    },


}

# 工具函式
def get_config(experiment_name: str) -> dict:
    """檢索並給定配置。"""
    exp_conf = EXPERIMENTS[experiment_name]

    config = {
        'experiment_name': experiment_name,
        'audio': AUDIO,
        'device': DEVICE,
        **exp_conf,
    }
    return config

