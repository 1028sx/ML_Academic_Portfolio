"""
CNN 深度學習模組的設定中心。

此檔案作為所有設定的單一來源，包括資料處理參數、模型架構和實驗定義。
學術作品集版本 - 展示進階 CNN 架構的音訊分類。
"""
import os

# ==============================================================================
#  路徑和目錄設定
# ==============================================================================
# --- 根目錄路徑 ---
def get_cnn_project_root():
    """返回 '02_CNN_DeepLearning' 目錄的絕對路徑。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_main_project_root():
    """返回主專案根目錄 (ML_Academic_Portfolio) 的絕對路徑。"""
    return os.path.dirname(get_cnn_project_root())

CNN_PROJECT_ROOT = get_cnn_project_root()
MAIN_PROJECT_ROOT = get_main_project_root()

# --- 原始數據路徑（位於專案根目錄） ---
DATA_DIR = os.path.join(MAIN_PROJECT_ROOT, '04_Data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
VOICE_DATA_DIR = os.path.join(DATA_DIR, 'voice_dataset')
SAMPLE_DATA_DIR = os.path.join(DATA_DIR, 'sample_data')
TRAIN_SET_DIR = os.path.join(VOICE_DATA_DIR, 'train_set')
TEST_SET_DIR = os.path.join(VOICE_DATA_DIR, 'test_set')
TRAIN_CSV_PATH = os.path.join(DATA_DIR, 'raw', 'train.csv')
SUBMISSION_CSV_PATH = os.path.join(DATA_DIR, 'raw', 'submission.csv')

# --- 輸出路徑（CNN 專案特定） ---
OUTPUT_DIR = os.path.join(CNN_PROJECT_ROOT, 'output')
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
# 定義可重複使用的資料處理策略。
# 每個管道指定音訊在輸入模型前應如何轉換。
# 這讓我們可以輕鬆地混合匹配不同的資料策略與模型。
#
# - 'target_len': 頻譜圖的最終寬度。對模型輸入形狀至關重要。
# - 'use_vad': 是否應用語音活動檢測來裁剪靜音。
# - 'vad_config': VAD 啟用時的參數。
# - 'augmentation_mode': 要應用的資料增強類型。
# ------------------------------------------------------------------------------
DATA_PIPELINES = {
    # --- 基礎管道（無增強） ---
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

    # --- 帶增強的管道 ---
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
    # 新增：為 V23 測試 VAD 修復，使用更相容的目標長度
    'P7_VAD_GentleAudio_215': {
        'target_len': 215, # 配合原始 V7 模型的預期輸入
        'use_vad': True,
        'vad_config': { "top_db": 25, "frame_length": 1024, "hop_length": 256 },
        'augmentation_mode': 'gentle_audio',
    },
    # 新增：P8 用於 V25_Gentle_VAD 實驗
    'P8_VAD_Gentle_215': {
        'target_len': 215,
        'use_vad': True,
        'vad_config': { "top_db": 30, "frame_length": 2048, "hop_length": 512 },
        'augmentation_mode': None, # 無增強，只有溫和的 VAD
    },
    
    # --- SpecMix 增強管道 (2024年最新技術) ---
    'P9_SpecMix_215': {
        'target_len': 215,
        'use_vad': False,
        'vad_config': None,
        'augmentation_mode': 'specmix',
        'specmix_config': {
            'use_specmix': True,
            'use_mixspeech': False,
            'use_noise': True,
            'use_time_shift': True,
            'specmix_prob': 0.5,
            'noise_factor': 0.002,
            'shift_limit': 0.1
        }
    },
    'P10_MixSpeech_215': {
        'target_len': 215,
        'use_vad': False,
        'vad_config': None,
        'augmentation_mode': 'mixspeech',
        'specmix_config': {
            'use_specmix': False,
            'use_mixspeech': True,
            'use_noise': True,
            'use_time_shift': True,
            'specmix_prob': 0.5,
            'noise_factor': 0.002,
            'shift_limit': 0.1
        }
    },
    'P11_Combined_SpecMix_VAD_215': {
        'target_len': 215,
        'use_vad': True,
        'vad_config': { "top_db": 25, "frame_length": 1024, "hop_length": 256 },
        'augmentation_mode': 'specmix_combined',
        'specmix_config': {
            'use_specmix': True,
            'use_mixspeech': False,
            'use_noise': True,
            'use_time_shift': True,
            'specmix_prob': 0.7,  # Higher probability for combined strategy
            'noise_factor': 0.003,
            'shift_limit': 0.15
        }
    },
}

# ==============================================================================
#  模型架構藍圖
# ==============================================================================
# 定義每個模型架構的藍圖。這將模型的內在屬性與訓練超參數分離。
#
# - 'model_class': models.py 中模型類別的名稱。
# - 'data_pipeline': 這個模型設計的*預設*資料管道。
#   這有助於確定正確的輸入維度 (input_w)。
# - 'flattened_dim': 對於 CNN，這可以預先計算以加速
#   模型初始化。如為 None，將動態計算。
# ------------------------------------------------------------------------------
MODEL_BLUEPRINTS = {
    'V6_CNN_Baseline': {
        'model_class': 'V6_CNN_Baseline',
        'data_pipeline': DATA_PIPELINES['P1_Standard_215'],
        'flattened_dim': None, # 讓模型自動計算
    },
    'V7_CNN_Advanced': {
        'model_class': 'V7_CNN_Advanced',
        'data_pipeline': DATA_PIPELINES['P1_Standard_215'],
        'flattened_dim': None, # 讓模型自動計算
        'dropout_p': 0.4, # V7 為基礎模型的預設 dropout 率，可被 Optuna 覆寫
    },
    'V11_DeeperCNN': {
        'model_class': 'V11_DeeperCNN',
        'data_pipeline': DATA_PIPELINES['P1_Standard_215'],
        'flattened_dim': None, # 讓模型自動計算
        'dropout_p': 0.4, # 預設 dropout 率
    },
    'V12_OptimizedCNN': {
        'model_class': 'V12_OptimizedCNN',
        'data_pipeline': DATA_PIPELINES['P2_VAD_400'], # 這個模型期望 VAD 處理後的資料
        'flattened_dim': 51200, # 為加速而預先計算
        'dropout_p': 0.4, # 預設 dropout 率
    },
}


# ==============================================================================
#  實驗配置
# ==============================================================================
# 定義具體實驗，結合模型、資料管道和訓練超參數。
# 這是定義新測試的主要字典。
# ------------------------------------------------------------------------------
EXPERIMENTS = {
    # --- 在修復數據上的新基線 ---
    'V16_Baseline_Restored': {
        'model_blueprint': MODEL_BLUEPRINTS['V6_CNN_Baseline'],
        'data_pipeline': DATA_PIPELINES['P1_Standard_215'],
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-3,
        'epochs': 100, 'patience': 15, 'batch_size': 64,
        'scheduler': None,
    },

    # --- 路徑 1：升級模型正則化 ---
    'V17_Regularization_Upgrade': {
        'model_blueprint': MODEL_BLUEPRINTS['V7_CNN_Advanced'], # 使用 V7 架構
        'data_pipeline': DATA_PIPELINES['P1_Standard_215'],    # 使用與 V16 相同的乾淨資料
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-3,           # 與 V16 相同的訓練參數
        'epochs': 100, 'patience': 15, 'batch_size': 64,
        'scheduler': None,
    },

    # --- 路徑 1.1：在 V17 基礎上增加進階訓練策略 ---
    'V18_AdamW_CosineLR': {
        'model_blueprint': MODEL_BLUEPRINTS['V7_CNN_Advanced'], # 與 V17 相同的模型
        'data_pipeline': DATA_PIPELINES['P1_Standard_215'],    # 與 V17 相同的資料
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-3,
        'epochs': 150, 'patience': 25, 'batch_size': 64,        # 為排程器增加週期數
        'scheduler': {'type': 'CosineAnnealingLR', 'params': {'T_max': 150}}, # 增加 V12 的排程器
        'gradient_clip_val': 1.0,                               # 增加 V12 的梯度裁剪
    },

    # --- 路徑 1.2：謹慎地將 SpecAugment 應用於最佳模型 (V17) ---
    'V19_V17_Model_with_SpecAug': {
        'model_blueprint': MODEL_BLUEPRINTS['V7_CNN_Advanced'],   # 與 V17 相同的穩定模型
        'data_pipeline': DATA_PIPELINES['P3_Standard_SpecAug_215'], # 使用 SpecAugment 但不使用 VAD
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-3,             # 來自 V17 的穩定訓練參數
        'epochs': 150, 'patience': 20, 'batch_size': 64,          # 為增強增加更多週期
        'scheduler': None,
    },

    # --- 路徑 2：將 VAD 應用於最佳模型 (V17) ---
    'V20_V17_Model_with_VAD': {
        'model_blueprint': MODEL_BLUEPRINTS['V7_CNN_Advanced'],   # 與 V17 相同的穩定模型
        'data_pipeline': DATA_PIPELINES['P2_VAD_400'],           # 使用僅 VAD 的管道
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-3,             # 來自 V17 的穩定訓練參數
        'epochs': 150, 'patience': 20, 'batch_size': 64,          # 為 VAD 資料增加更多週期
        'scheduler': None,
    },

    # --- 重新訓練舊模型以獲得準確基準 ---
    'V6_CNN_Baseline_Benchmark': {
        'model_blueprint': MODEL_BLUEPRINTS['V6_CNN_Baseline'],
        'data_pipeline': DATA_PIPELINES['P1_Standard_215'],
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-3,
        'epochs': 100, 'patience': 15, 'batch_size': 64,
        'scheduler': None,
    },
    'V7_CNN_Advanced_Benchmark': {
        'model_blueprint': MODEL_BLUEPRINTS['V7_CNN_Advanced'],
        'data_pipeline': DATA_PIPELINES['P1_Standard_215'],
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-3,
        'epochs': 150, 'patience': 20, 'batch_size': 64,
        'scheduler': None,
    },
     'V11_DeeperCNN_Benchmark': {
        'model_blueprint': MODEL_BLUEPRINTS['V11_DeeperCNN'],
        'data_pipeline': DATA_PIPELINES['P1_Standard_215'],
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-3,
        'epochs': 150, 'patience': 25, 'batch_size': 64,
        'scheduler': None,
    },

    # --- 進階實驗 ---
    'V9_CNN_Scheduled': {
        'model_blueprint': MODEL_BLUEPRINTS['V7_CNN_Advanced'],
        'data_pipeline': DATA_PIPELINES['P3_Standard_SpecAug_215'],
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-3,
        'epochs': 150, 'patience': 20, 'batch_size': 64,
        'scheduler': {'type': 'CosineAnnealingLR', 'params': {'T_max': 150}},
    },
    'V10_CNN_Gentle_Aug': {
        'model_blueprint': MODEL_BLUEPRINTS['V7_CNN_Advanced'],
        'data_pipeline': DATA_PIPELINES['P5_Standard_GentleAudio_215'],
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-3,
        'epochs': 150, 'patience': 20, 'batch_size': 64,
        'scheduler': {'type': 'CosineAnnealingLR', 'params': {'T_max': 150}},
    },
    'V12_Full_Optimization': {
        'model_blueprint': MODEL_BLUEPRINTS['V12_OptimizedCNN'],
        'data_pipeline': DATA_PIPELINES['P4_VAD_SpecAug_400'],
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-3,
        'epochs': 200, 'patience': 30, 'batch_size': 64,
        'scheduler': {'type': 'CosineAnnealingLR', 'params': {'T_max': 200}},
        'gradient_clip_val': 1.0,
    },
     'V15_Advanced_VAD': {
        'model_blueprint': MODEL_BLUEPRINTS['V12_OptimizedCNN'],
        'data_pipeline': DATA_PIPELINES['P4_VAD_SpecAug_400'], # 使用 VAD + SpecAugment
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-3,
        'epochs': 200, 'patience': 30, 'batch_size': 64,
        'scheduler': {'type': 'CosineAnnealingLR', 'params': {'T_max': 200}},
        'gradient_clip_val': 1.0,
    },

    # ==============================================================================
    #  V21 開發路徑：使用驗證過的組件復製 SOTA
    # ==============================================================================
    'V21a_V7_GentleAug_Scheduled': {
        'model_blueprint': MODEL_BLUEPRINTS['V7_CNN_Advanced'],           # 從最穩定的模型開始
        'data_pipeline': DATA_PIPELINES['P5_Standard_GentleAudio_215'],  # 重新應用 V10 驗證過的溫和音訊增強
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-3,                     # 使用穩定的訓練參數
        'epochs': 150, 'patience': 25, 'batch_size': 64,                  # 為增強和排程器增加週期
        'scheduler': {'type': 'CosineAnnealingLR', 'params': {'T_max': 150}}, # 增加排程器以穩定收斂
        'gradient_clip_val': 1.0,                                        # 增加梯度裁剪以穩定
    },

    # --- 路徑 B：結合 SOTA 模型與 VAD ---
    'V22_V21a_with_VAD': {
        'model_blueprint': MODEL_BLUEPRINTS['V7_CNN_Advanced'],      # 與 V21a 相同的穩定模型
        'data_pipeline': DATA_PIPELINES['P6_VAD_GentleAudio_400'], # 新增：結合 VAD 和溫和增強
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-3,                # 與 V21a 相同的訓練參數
        'epochs': 200, 'patience': 30, 'batch_size': 64,             # 為更複雜的資料增加週期
        'scheduler': {'type': 'CosineAnnealingLR', 'params': {'T_max': 200}},
        'gradient_clip_val': 1.0,
    },

    # ==============================================================================
    #  V23 開發路徑：驗證 VAD 修復
    # ==============================================================================
    'V23_VAD_Fix': {
        'model_blueprint': MODEL_BLUEPRINTS['V7_CNN_Advanced'],      # 基於穩定的 V7 模型
        'data_pipeline': DATA_PIPELINES['P7_VAD_GentleAudio_215'], # 使用新管道來修復 VAD 並更正目標長度
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-3,                # 與 V21a 相同的訓練參數
        'epochs': 200, 'patience': 30, 'batch_size': 64,             # 複雜數據需要更多訓練週期
        'scheduler': {'type': 'CosineAnnealingLR', 'params': {'T_max': 200}},
        'gradient_clip_val': 1.0,
        'description': "Verify the fix for VAD-processed audio. This experiment combines the SOTA model (V7) with VAD and gentle augmentation, using a target_len of 215 which is more compatible with the V7 architecture and VAD's output."
    },

    # --- V24：「雙重優勢」實驗 ---
    # 目標：結合最佳數據管道（V23 的修正）和最佳訓練策略（V21a）
    'V24_V21a_with_VAD_Fix': {
        'model_blueprint': MODEL_BLUEPRINTS['V7_CNN_Advanced'],           # V21a 的穩定模型
        'data_pipeline': DATA_PIPELINES['P7_VAD_GentleAudio_215'],  # V23 的修正 VAD 管道
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-3,                     # V21a 的訓練參數
        'epochs': 150, 'patience': 25, 'batch_size': 64,                  # V21a 的訓練參數
        'scheduler': {'type': 'CosineAnnealingLR', 'params': {'T_max': 150}}, # V21a 的調度器
        'gradient_clip_val': 1.0,                                        # V21a 的梯度裁剪
    },

    # ==============================================================================
    #  V25 過擬合修復嘗試
    # ==============================================================================
    'V25_Conservative_Approach': {
        'description': "回歸V21a配置但增加正則化，作為修正過擬合的第一步",
        'model_blueprint': MODEL_BLUEPRINTS['V7_CNN_Advanced'],
        'data_pipeline': DATA_PIPELINES['P5_Standard_GentleAudio_215'], # 回歸V21a的簡單配置
        'optimizer': {'type': 'AdamW', 'lr': 8e-5, 'wd': 1e-2},      # 降低學習率, 增加L2正則化
        'epochs': 120, 'patience': 20, 'batch_size': 64,             # 更保守的訓練
        'scheduler': {'type': 'CosineAnnealingLR', 'params': {'T_max': 120}},
        'gradient_clip_val': 0.5,                                   # 更嚴格的梯度裁剪
        'dropout_rate': None, # 使用模型預設值
    },
    
    'V25_Gentle_VAD': {
        'description': "如果仍要使用VAD，採用更溫和的參數且不搭配增強",
        'model_blueprint': MODEL_BLUEPRINTS['V7_CNN_Advanced'],
        'data_pipeline': DATA_PIPELINES['P8_VAD_Gentle_215'],
        'optimizer': {'type': 'AdamW', 'lr': 8e-5, 'wd': 1e-2},      # 與Conservative一致
        'epochs': 120, 'patience': 20, 'batch_size': 64,
        'scheduler': {'type': 'CosineAnnealingLR', 'params': {'T_max': 120}},
        'gradient_clip_val': 0.5,
        'dropout_rate': None,
    },
    
    # --- 最終 SOTA 模型及其變體 ---
    'V25_Heavy_Regularization_Tuned': {
        'model_blueprint': MODEL_BLUEPRINTS['V7_CNN_Advanced'],
        'data_pipeline': DATA_PIPELINES['P1_Standard_215'],
        'optimizer': 'AdamW', 
        'lr': 0.0001896061389230623,         # 已調優
        'wd': 0.00996023191543026,          # 已調優
        'epochs': 200, 'patience': 30, 'batch_size': 64,
        'scheduler': {'type': 'CosineAnnealingLR', 'params': {'T_max': 200}},
        'model_params': {
            'dropout_p': 0.4355431698292854   # 已調優
        }
    },

    'V25_Stage1_Pretrain': {
        'description': "分階段訓練第一步：在無增強的基礎數據上預訓練",
        'model_blueprint': MODEL_BLUEPRINTS['V7_CNN_Advanced'],
        'data_pipeline': DATA_PIPELINES['P1_Standard_215'], # 無增強的基礎數據
        'optimizer': {'type': 'AdamW', 'lr': 1e-4, 'wd': 1e-3},
        'epochs': 80, 'patience': 15, 'batch_size': 64,
        'scheduler': None,
    },

    'V25_Stage2_Finetune': {
        'description': "分階段訓練第二步：載入Stage1模型，在溫和增強數據上微調",
        'load_checkpoint_from': 'V25_Stage1_Pretrain.pth', # 提示腳本需要載入權重
        'model_blueprint': MODEL_BLUEPRINTS['V7_CNN_Advanced'],
        'data_pipeline': DATA_PIPELINES['P5_Standard_GentleAudio_215'], # 在溫和增強上微調
        'optimizer': {'type': 'AdamW', 'lr': 5e-5, 'wd': 2e-3}, # 更低學習率、更高正則化
        'epochs': 40, 'patience': 10, 'batch_size': 64,
        'scheduler': None,
    },

    # ==============================================================================
    #  V26/V27 開發路徑：SOTA 組合和進階正則化
    # ==============================================================================
    'V26_VAD_HeavyReg': {
        'model_blueprint': MODEL_BLUEPRINTS['V7_CNN_Advanced'],
        'data_pipeline': DATA_PIPELINES['P8_VAD_Gentle_215'],
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 5e-2,
        'epochs': 200, 'patience': 40, 'batch_size': 64,
        'scheduler': None,
        'dropout_rate': 0.7,
    },

    'V27_VCReg_Experiment': {
        'model_blueprint': MODEL_BLUEPRINTS['V7_CNN_Advanced'],
        'data_pipeline': DATA_PIPELINES['P1_Standard_215'], # 使用清理過的數據
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 5e-3,             # 較溫和的權重衰減
        'epochs': 200, 'patience': 40, 'batch_size': 64,
        'scheduler': None,
        'dropout_rate': 0.5,                                      # 標準 dropout 率
        'regularizers': [
            {'type': 'vcreg', 'lambda': 25.0, 'mu': 25.0} # 標準 VCReg 數值
        ]
    },

    # ==============================================================================
    #  V28-V30: SpecMix 實驗系列 (2024年最新技術)
    # ==============================================================================
    'V28_SpecMix_Test': {
        'description': "Initial SpecMix implementation test - should outperform SpecAugment by 4-6%",
        'model_blueprint': MODEL_BLUEPRINTS['V7_CNN_Advanced'],
        'data_pipeline': DATA_PIPELINES['P9_SpecMix_215'],
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-2,
        'epochs': 100, 'patience': 20, 'batch_size': 64,
        'scheduler': None,
        'dropout_rate': 0.5,  # Increased regularization for SpecMix
    },
    
    'V29_Combined_Augmentation': {
        'description': "Combined augmentation strategy with SpecMix + noise + time shift",
        'model_blueprint': MODEL_BLUEPRINTS['V7_CNN_Advanced'],
        'data_pipeline': DATA_PIPELINES['P11_Combined_SpecMix_VAD_215'],
        'optimizer': 'AdamW', 'lr': 8e-5, 'wd': 3e-2,  # Heavy regularization
        'epochs': 150, 'patience': 25, 'batch_size': 64,
        'scheduler': {'type': 'CosineAnnealingLR', 'params': {'T_max': 150}},
        'dropout_rate': 0.6,  # Even more dropout for combined strategy
        'label_smoothing': 0.1,  # Add label smoothing
    },
    
    'V30_MixSpeech_Alternative': {
        'description': "Test MixSpeech as alternative to SpecMix for speech-specific optimization",
        'model_blueprint': MODEL_BLUEPRINTS['V7_CNN_Advanced'],
        'data_pipeline': DATA_PIPELINES['P10_MixSpeech_215'],
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 2e-2,
        'epochs': 100, 'patience': 20, 'batch_size': 64,
        'scheduler': None,
        'dropout_rate': 0.5,
    },

    # --- 除錯區段 ---
    # 此部分用於臨時的一次性除錯實驗
    # 不應用於最終模型評估
    'Debug_SpecMix_Quick': {
        'description': "Quick 2-epoch test to verify SpecMix implementation",
        'model_blueprint': MODEL_BLUEPRINTS['V7_CNN_Advanced'],
        'data_pipeline': DATA_PIPELINES['P9_SpecMix_215'],
        'optimizer': 'AdamW', 'lr': 1e-4, 'wd': 1e-2,
        'epochs': 2,  # Only 2 epochs for quick validation
        'patience': 10, 'batch_size': 32,  # Smaller batch for faster testing
        'scheduler': None,
        'dropout_rate': 0.4,
    },
    
    'Debug_Augmentation_Fix': {
        'description': "A single-epoch test based on V25_Heavy_Regularization to verify the fix for audio augmentation length issues.",
        'model_blueprint': MODEL_BLUEPRINTS['V7_CNN_Advanced'],
        'data_pipeline': DATA_PIPELINES['P5_Standard_GentleAudio_215'],
        'optimizer': {'type': 'AdamW', 'lr': 1e-4, 'wd': 5e-3},
        'epochs': 1, # 只運行一個週期以進行快速驗證
        'patience': 1,
        'batch_size': 64,
        'scheduler': None,
        'gradient_clip_val': 0.5,
        'dropout_rate': 0.4,
    }
}

def get_config(experiment_name):
    """
    透過合併實驗的特定設定與其模型藍圖和數據管道來獲取給定實驗的最終配置。

    此函數作為訪問任何配置的單一入口點，確保所有必要部分都能一致地組合。
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
            'optimizer': exp_config.get('optimizer', 'AdamW'),
            'lr': exp_config.get('lr', 1e-4),
            'wd': exp_config.get('wd', 1e-3),
            'epochs': exp_config.get('epochs', 100),
            'patience': exp_config.get('patience', 15),
            'batch_size': exp_config.get('batch_size', 64),
            'scheduler': exp_config.get('scheduler'),
            'gradient_clip_val': exp_config.get('gradient_clip_val'),
        },
        'model': {
            'model_class': model_blueprint['model_class'],
            # 允許 Optuna 覆寫預設的 dropout_p
            'dropout_p': model_blueprint.get('dropout_p', 0.4), 
        },
        'data': {
            **data_pipeline,
            'train_csv_path': TRAIN_CSV_PATH,
            'voice_data_dir': VOICE_DATA_DIR,
        },
        'audio': {**AUDIO},

        # 新增輸出路徑用於記錄、儲存模型和繪圖
        'log_save_dir': LOG_SAVE_DIR,
        'model_save_dir': MODEL_SAVE_DIR,
        'plot_save_dir': PLOT_SAVE_DIR,
    }
    
    # 根據數據管道動態新增輸入維度到模型配置
    # 這使模型對數據處理的變化具有健壯性
    final_config['model']['input_h'] = AUDIO['n_mels']
    final_config['model']['input_w'] = data_pipeline['target_len']
    final_config['model']['output_dim'] = AUDIO['num_classes']
    final_config['model']['flattened_dim'] = model_blueprint.get('flattened_dim')

    return final_config 