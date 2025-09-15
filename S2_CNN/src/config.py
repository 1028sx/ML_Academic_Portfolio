import os

# 路徑和目錄設定
def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_main_project_root():
    return os.path.dirname(get_project_root())

PROJECT_ROOT = get_project_root()
MAIN_PROJECT_ROOT = get_main_project_root()

# 原始數據
DATA_DIR = os.path.join(MAIN_PROJECT_ROOT, 'data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
VOICE_DATA_DIR = os.path.join(DATA_DIR, 'voice_dataset')
SAMPLE_DATA_DIR = os.path.join(DATA_DIR, 'sample_data')
TRAIN_SET_DIR = os.path.join(VOICE_DATA_DIR, 'train_set')
TEST_SET_DIR = os.path.join(VOICE_DATA_DIR, 'test_set')
TRAIN_CSV_PATH = os.path.join(DATA_DIR, 'raw', 'train.csv')
SUBMISSION_CSV_PATH = os.path.join(DATA_DIR, 'raw', 'submission.csv')

# 輸出路徑
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
MODEL_SAVE_DIR = os.path.join(OUTPUT_DIR, 'models')
PLOT_SAVE_DIR = os.path.join(OUTPUT_DIR, 'plots')
LOG_SAVE_DIR = os.path.join(OUTPUT_DIR, 'logs')


# 音頻處理參數
AUDIO = {
    'sample_rate': 16000,
    'n_fft': 1024,
    'hop_length': 160,
    'n_mels': 128,
    'num_classes': 8,
}

# 數據處理管道
DATA_PIPELINES = {
    # 基礎管道（無增強）
    'P1_Standard_215': {
        'target_len': 215,
        'use_vad': False,
        'vad_config': None,
        'augmentation_mode': None,
    },
    'P2_VAD_400': {
        'target_len': 400,
        'use_vad': True,
        'vad_config': { "top_db": 25, "frame_length": 1024, "hop_length": 256 },
        'augmentation_mode': None,
    },

    # 帶增強的管道
    'P3_Standard_SpecAug_215': {
        'target_len': 215,
        'use_vad': False,
        'vad_config': None,
        'augmentation_mode': 'spec_augment',
    },
    'P4_VAD_SpecAug_400': {
        'target_len': 400,
        'use_vad': True,
        'vad_config': { "top_db": 25, "frame_length": 1024, "hop_length": 256 },
        'augmentation_mode': 'spec_augment',
    },
    'P5_Standard_GentleAudio_215': {
        'target_len': 215,
        'use_vad': False,
        'vad_config': None,
        'augmentation_mode': 'gentle_audio',
    },
    'P6_VAD_GentleAudio_400': {
        'target_len': 400,
        'use_vad': True,
        'vad_config': { "top_db": 25, "frame_length": 1024, "hop_length": 256 },
        'augmentation_mode': 'gentle_audio',
    },
    'P7_VAD_GentleAudio_215': {
        'target_len': 215,
        'use_vad': True,
        'vad_config': { "top_db": 25, "frame_length": 1024, "hop_length": 256 },
        'augmentation_mode': 'gentle_audio',
    },
    'P8_VAD_Gentle_215': {
        'target_len': 215,
        'use_vad': True,
        'vad_config': { "top_db": 30, "frame_length": 2048, "hop_length": 512 },
        'augmentation_mode': None,
    },

}

# 模型架構藍圖
MODEL_BLUEPRINTS = {
    'V6_CNN_Baseline': {
        'model_class': 'V6_CNN_Baseline',
        'data_pipeline': DATA_PIPELINES['P1_Standard_215'],
        'flattened_dim': None,
    },
    'V7_CNN_Advanced': {
        'model_class': 'V7_CNN_Advanced',  # 加入BatchNorm和進階正則化
        'data_pipeline': DATA_PIPELINES['P1_Standard_215'],
        'flattened_dim': None,
        'dropout_p': 0.4,  # 新增dropout正則化
    },
    'V11_DeeperCNN': {
        'model_class': 'V11_DeeperCNN',  # 加深網絡層數
        'data_pipeline': DATA_PIPELINES['P1_Standard_215'],
        'flattened_dim': None,
        'dropout_p': 0.4,
    },
    'V12_OptimizedCNN': {
        'model_class': 'V12_OptimizedCNN',  # 針對VAD優化架構
        'data_pipeline': DATA_PIPELINES['P2_VAD_400'],  # 改用VAD預處理
        'flattened_dim': 51200,  # VAD後的固定維度
        'dropout_p': 0.4,
    },
}


# 實驗配置
'''配置格式說明：
1. 舊版扁平格式 (V16-V24，早期實驗):'optimizer' 為字串，'lr' 和 'wd' 在同層級
2. 新版嵌套格式 (V25+，後期實驗):'optimizer' 為字典，包含所有參數
train_cnn.py 中的 get_optimizer() 函式保持兩種格式的向後兼容性
'''


EXPERIMENTS = {
    'V16_Baseline_Restored': {
        'description': "V6架構基準",
        'model_blueprint': MODEL_BLUEPRINTS['V6_CNN_Baseline'],
        'data_pipeline': DATA_PIPELINES['P1_Standard_215'],
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-3,  # 舊版扁平格式
        'epochs': 100, 'patience': 15, 'batch_size': 64,
        'scheduler': None,
    },

    # 正則化
    'V17_Regularization_Upgrade': {
        'description': "正規化對抗過擬合",
        'model_blueprint': MODEL_BLUEPRINTS['V7_CNN_Advanced'],
        'data_pipeline': DATA_PIPELINES['P1_Standard_215'],
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-3,
        'epochs': 100, 'patience': 15, 'batch_size': 64,
        'scheduler': None,
    },

    # 進階策略
    'V18_AdamW_CosineLR': {
        'description': "V17+餘弦學習率排程",
        'model_blueprint': MODEL_BLUEPRINTS['V7_CNN_Advanced'],
        'data_pipeline': DATA_PIPELINES['P1_Standard_215'],
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-3,
        'epochs': 150, 'patience': 25, 'batch_size': 64,
        'scheduler': {'type': 'CosineAnnealingLR', 'params': {'T_max': 150}},
        'gradient_clip_val': 1.0,
    },

    # SpecAugment
    'V19_V17_Model_with_SpecAug': {
        'description': "V17+SpecAugment增強",
        'model_blueprint': MODEL_BLUEPRINTS['V7_CNN_Advanced'],
        'data_pipeline': DATA_PIPELINES['P3_Standard_SpecAug_215'],
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-3,
        'epochs': 150, 'patience': 20, 'batch_size': 64,
        'scheduler': None,
    },

    # VAD
    'V20_V17_Model_with_VAD': {
        'description': "V17+VAD語音檢測",
        'model_blueprint': MODEL_BLUEPRINTS['V7_CNN_Advanced'],
        'data_pipeline': DATA_PIPELINES['P2_VAD_400'],
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-3,
        'epochs': 150, 'patience': 20, 'batch_size': 64,
        'scheduler': None,
    },

    # 舊模型
    'V6_CNN_Baseline_Benchmark': {
        'description': "CNN 基準",
        'model_blueprint': MODEL_BLUEPRINTS['V6_CNN_Baseline'],
        'data_pipeline': DATA_PIPELINES['P1_Standard_215'],
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-3,
        'epochs': 100, 'patience': 15, 'batch_size': 64,
        'scheduler': None,
    },
    'V7_CNN_Advanced_Benchmark': {
        'description': "進階 CNN 基準",
        'model_blueprint': MODEL_BLUEPRINTS['V7_CNN_Advanced'],
        'data_pipeline': DATA_PIPELINES['P1_Standard_215'],
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-3,
        'epochs': 150, 'patience': 20, 'batch_size': 64,
        'scheduler': None,
    },
     'V11_DeeperCNN_Benchmark': {
        'description': "深化架構",
        'model_blueprint': MODEL_BLUEPRINTS['V11_DeeperCNN'],
        'data_pipeline': DATA_PIPELINES['P1_Standard_215'],
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-3,
        'epochs': 150, 'patience': 25, 'batch_size': 64,
        'scheduler': None,
    },

    # 進階實驗
    'V9_CNN_Scheduled': {
        'description': "V7+SpecAugment+學習率排程",
        'model_blueprint': MODEL_BLUEPRINTS['V7_CNN_Advanced'],
        'data_pipeline': DATA_PIPELINES['P3_Standard_SpecAug_215'],
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-3,
        'epochs': 150, 'patience': 20, 'batch_size': 64,
        'scheduler': {'type': 'CosineAnnealingLR', 'params': {'T_max': 150}},
    },
    'V10_CNN_Gentle_Aug': {
        'description': "V7+溫和音訊增強",
        'model_blueprint': MODEL_BLUEPRINTS['V7_CNN_Advanced'],
        'data_pipeline': DATA_PIPELINES['P5_Standard_GentleAudio_215'],
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-3,
        'epochs': 150, 'patience': 20, 'batch_size': 64,
        'scheduler': {'type': 'CosineAnnealingLR', 'params': {'T_max': 150}},
    },
    'V12_Full_Optimization': {
        'description': "V12超參數全面優化",
        'model_blueprint': MODEL_BLUEPRINTS['V12_OptimizedCNN'],
        'data_pipeline': DATA_PIPELINES['P4_VAD_SpecAug_400'],
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-3,
        'epochs': 200, 'patience': 30, 'batch_size': 64,
        'scheduler': {'type': 'CosineAnnealingLR', 'params': {'T_max': 200}},
        'gradient_clip_val': 1.0,
    },
     'V15_Advanced_VAD': {
        'description': "V12+進階VAD",
        'model_blueprint': MODEL_BLUEPRINTS['V12_OptimizedCNN'],
        'data_pipeline': DATA_PIPELINES['P4_VAD_SpecAug_400'],
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-3,
        'epochs': 200, 'patience': 30, 'batch_size': 64,
        'scheduler': {'type': 'CosineAnnealingLR', 'params': {'T_max': 200}},
        'gradient_clip_val': 1.0,
    },

    'V21a_V7_GentleAug_Scheduled': {
        'description': "V7+溫和增強+學習率排程",
        'model_blueprint': MODEL_BLUEPRINTS['V7_CNN_Advanced'],
        'data_pipeline': DATA_PIPELINES['P5_Standard_GentleAudio_215'],
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-3,
        'epochs': 150, 'patience': 25, 'batch_size': 64,
        'scheduler': {'type': 'CosineAnnealingLR', 'params': {'T_max': 150}},
        'gradient_clip_val': 1.0,
    },

    # SOTA & VAD
    'V22_V21a_with_VAD': {
        'description': "V21a+VAD數據清洗",
        'model_blueprint': MODEL_BLUEPRINTS['V7_CNN_Advanced'],
        'data_pipeline': DATA_PIPELINES['P6_VAD_GentleAudio_400'],
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-3,
        'epochs': 200, 'patience': 30, 'batch_size': 64,
        'scheduler': {'type': 'CosineAnnealingLR', 'params': {'T_max': 200}},
        'gradient_clip_val': 1.0,
    },

    # 驗證 VAD 修復
    'V23_VAD_Fix': {
        'description': "V7+VAD修正版本",
        'model_blueprint': MODEL_BLUEPRINTS['V7_CNN_Advanced'],
        'data_pipeline': DATA_PIPELINES['P7_VAD_GentleAudio_215'],
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-3,
        'epochs': 200, 'patience': 30, 'batch_size': 64,
        'scheduler': {'type': 'CosineAnnealingLR', 'params': {'T_max': 200}},
        'gradient_clip_val': 1.0,
    },

    # V24：雙重優勢實驗
    'V24_V21a_with_VAD_Fix': {
        'description': "V21a+修正VAD",
        'model_blueprint': MODEL_BLUEPRINTS['V7_CNN_Advanced'],
        'data_pipeline': DATA_PIPELINES['P7_VAD_GentleAudio_215'],
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-3,
        'epochs': 150, 'patience': 25, 'batch_size': 64,
        'scheduler': {'type': 'CosineAnnealingLR', 'params': {'T_max': 150}},
        'gradient_clip_val': 1.0,
    },

    # V25 過擬合修復嘗試
    'V25_Conservative_Approach': {
        'description': "回歸V21a配置但增加正則化",
        'model_blueprint': MODEL_BLUEPRINTS['V7_CNN_Advanced'],
        'data_pipeline': DATA_PIPELINES['P5_Standard_GentleAudio_215'],
        'optimizer': {'type': 'AdamW', 'lr': 8e-5, 'wd': 1e-2},
        'epochs': 120, 'patience': 20, 'batch_size': 64,
        'scheduler': {'type': 'CosineAnnealingLR', 'params': {'T_max': 120}},
        'gradient_clip_val': 0.5,
        'dropout_rate': None,
    },

    'V25_Gentle_VAD': {
        'description': "溫和VAD且不搭配增強",
        'model_blueprint': MODEL_BLUEPRINTS['V7_CNN_Advanced'],
        'data_pipeline': DATA_PIPELINES['P8_VAD_Gentle_215'],
        'optimizer': {'type': 'AdamW', 'lr': 8e-5, 'wd': 1e-2},
        'epochs': 120, 'patience': 20, 'batch_size': 64,
        'scheduler': {'type': 'CosineAnnealingLR', 'params': {'T_max': 120}},
        'gradient_clip_val': 0.5,
        'dropout_rate': None,
    },

    # SOTA及其變體(已調整超參數)
    'V25_Heavy_Regularization_Tuned': {
        'description': "Optuna調優重度正規化SOTA",
        'model_blueprint': MODEL_BLUEPRINTS['V7_CNN_Advanced'],
        'data_pipeline': DATA_PIPELINES['P1_Standard_215'],
        'optimizer': 'AdamW',
        'lr': 0.0001896061389230623,
        'wd': 0.00996023191543026,
        'epochs': 200, 'patience': 30, 'batch_size': 64,
        'scheduler': {'type': 'CosineAnnealingLR', 'params': {'T_max': 200}},
        'model_params': {
            'dropout_p': 0.4355431698292854
        }
    },

    'V25_Stage1_Pretrain': {
        'description': "分階段訓練第一步預訓練",
        'model_blueprint': MODEL_BLUEPRINTS['V7_CNN_Advanced'],
        'data_pipeline': DATA_PIPELINES['P1_Standard_215'],
        'optimizer': {'type': 'AdamW', 'lr': 1e-4, 'wd': 1e-3},
        'epochs': 80, 'patience': 15, 'batch_size': 64,
        'scheduler': None,
    },

    'V25_Stage2_Finetune': {
        'description': "分階段訓練第二步微調",
        'load_checkpoint_from': 'V25_Stage1_Pretrain.pth',  # 載入權重
        'model_blueprint': MODEL_BLUEPRINTS['V7_CNN_Advanced'],
        'data_pipeline': DATA_PIPELINES['P5_Standard_GentleAudio_215'],
        'optimizer': {'type': 'AdamW', 'lr': 5e-5, 'wd': 2e-3},
        'epochs': 40, 'patience': 10, 'batch_size': 64,
        'scheduler': None,
    },

    # 混合與進階正則化
    'V26_VAD_HeavyReg': {
        'description': "VAD+極重正規化實驗",
        'model_blueprint': MODEL_BLUEPRINTS['V7_CNN_Advanced'],
        'data_pipeline': DATA_PIPELINES['P8_VAD_Gentle_215'],
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 5e-2,
        'epochs': 200, 'patience': 40, 'batch_size': 64,
        'scheduler': None,
        'dropout_rate': 0.7,
    },

    'V27_VCReg_Experiment': {
        'description': "VCReg正規化技術測試",
        'model_blueprint': MODEL_BLUEPRINTS['V7_CNN_Advanced'],
        'data_pipeline': DATA_PIPELINES['P1_Standard_215'],
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 5e-3,
        'epochs': 200, 'patience': 40, 'batch_size': 64,
        'scheduler': None,
        'dropout_rate': 0.5,
        'regularizers': [
            {'type': 'vcreg', 'lambda': 25.0, 'mu': 25.0}
        ]
    },

}

def get_config(experiment_name):
    """獲取配置"""
    config = EXPERIMENTS[experiment_name]
    model_blueprint = config['model_blueprint']
    data_pipeline = config['data_pipeline']

    # 實驗設定
    final_config = {
        'experiment_name': experiment_name,
        'train': {
            'optimizer': config.get('optimizer', 'AdamW'),
            'lr': config.get('lr', 1e-4),
            'wd': config.get('wd', 1e-3),
            'epochs': config.get('epochs', 100),
            'patience': config.get('patience', 15),
            'batch_size': config.get('batch_size', 64),
            'scheduler': config.get('scheduler'),
            'gradient_clip_val': config.get('gradient_clip_val'),
        },
        'model': {
            'model_class': model_blueprint['model_class'],
            'dropout_p': model_blueprint.get('dropout_p', 0.4),
        },
        'data': {
            **data_pipeline,
            'train_csv_path': TRAIN_CSV_PATH,
            'voice_data_dir': VOICE_DATA_DIR,
        },
        'audio': {**AUDIO},

        # 輸出路徑
        'log_save_dir': LOG_SAVE_DIR,
        'model_save_dir': MODEL_SAVE_DIR,
        'plot_save_dir': PLOT_SAVE_DIR,
    }

    # 動態新增輸入維度到模型配置
    final_config['model']['input_h'] = AUDIO['n_mels']
    final_config['model']['input_w'] = data_pipeline['target_len']
    final_config['model']['output_dim'] = AUDIO['num_classes']
    final_config['model']['flattened_dim'] = model_blueprint.get('flattened_dim')

    return final_config