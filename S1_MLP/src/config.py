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
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
SAMPLE_DATA_DIR = os.path.join(DATA_DIR, 'sample_data')
TRAIN_CSV_PATH = os.path.join(DATA_DIR, 'train_cleaned.csv')

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
    'P1_MLP_Standard': {
        'target_len': 215,
        'use_vad': False,
        'augmentation_mode': None,
    }
}

# 模型架構
MODEL_BLUEPRINTS = {
    'V1_Baseline': {
        'model_class': 'V1_Baseline',
        'data_pipeline': DATA_PIPELINES['P1_MLP_Standard'],
    },
    'V2_Deeper': {
        'model_class': 'V2_Deeper',  # 增加神經元數量 128→256
        'data_pipeline': DATA_PIPELINES['P1_MLP_Standard'],
    },
    'V3_AugLeaky': {
        'model_class': 'V3_AugLeaky',  # 擴寬至512層+LeakyReLU激活
        'data_pipeline': DATA_PIPELINES['P1_MLP_Standard'],
    },
    'V4_AdamW': {
        'model_class': 'V4_AdamW',  # 同V3架構，改用AdamW優化器
        'data_pipeline': DATA_PIPELINES['P1_MLP_Standard'],
    },
    'V5_Pooling': {
        'model_class': 'V5_Pooling',  # V3架構+全局平均池化
        'data_pipeline': DATA_PIPELINES['P1_MLP_Standard'],
    },
}

# 模型配置
EXPERIMENTS = {
    'V1_Baseline': {
        'description': "基礎架構",
        'model_blueprint': MODEL_BLUEPRINTS['V1_Baseline'],
        'data_pipeline': DATA_PIPELINES['P1_MLP_Standard'],
        'optimizer': 'Adam', 'lr': 1e-3, 'wd': 1e-4,
        'epochs': 50, 'patience': 10, 'batch_size': 64,
    },
    'V2_Deeper': {
        'description': "加深層數",
        'model_blueprint': MODEL_BLUEPRINTS['V2_Deeper'],
        'data_pipeline': DATA_PIPELINES['P1_MLP_Standard'],
        'optimizer': 'Adam', 'lr': 1e-3, 'wd': 1e-4,
        'epochs': 50, 'patience': 10, 'batch_size': 64,
    },
    'V3_AugLeaky': {
        'description': "擴寬架構+LeakyReLU激活",
        'model_blueprint': MODEL_BLUEPRINTS['V3_AugLeaky'],
        'data_pipeline': DATA_PIPELINES['P1_MLP_Standard'],
        'optimizer': 'Adam', 'lr': 1e-3, 'wd': 1e-4,
        'epochs': 50, 'patience': 10, 'batch_size': 64,
    },
    'V4_AdamW': {
        'description': "V3+AdamW優化",
        'model_blueprint': MODEL_BLUEPRINTS['V4_AdamW'],
        'data_pipeline': DATA_PIPELINES['P1_MLP_Standard'],
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-3,
        'epochs': 50, 'patience': 10, 'batch_size': 64,
    },
    'V5_Pooling': {
        'description': "V3+池化",
        'model_blueprint': MODEL_BLUEPRINTS['V5_Pooling'],
        'data_pipeline': DATA_PIPELINES['P1_MLP_Standard'],
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-3,
        'epochs': 50, 'patience': 10, 'batch_size': 64,
    },
}

def get_config(experiment_name):
    exp_config = EXPERIMENTS[experiment_name]
    model_blueprint = exp_config['model_blueprint']
    data_pipeline = exp_config['data_pipeline']

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
            'processed_data_dir': PROCESSED_DATA_DIR,
        },
        'audio': {**AUDIO},

        # 輸出路徑
        'log_save_dir': LOG_SAVE_DIR,
        'model_save_dir': MODEL_SAVE_DIR,
        'plot_save_dir': PLOT_SAVE_DIR,
    }

    final_config['model']['input_dim'] = AUDIO['n_mels'] * data_pipeline['target_len']
    final_config['model']['input_h'] = AUDIO['n_mels']
    final_config['model']['input_w'] = data_pipeline['target_len']
    final_config['model']['output_dim'] = AUDIO['num_classes']

    return final_config