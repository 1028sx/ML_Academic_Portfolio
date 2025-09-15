# ML Academic Portfolio - 中文語音分類學習專案

## 專案簡介

這是一個以深度學習實作語音八分類任務的學術專案，通過 MLP → CNN → Transformer 的順序展現從基礎到進階的學習軌跡。

### 議題背景
該議題完整描述如下（來自 AIdea 平台，[Link](https://aidea-web.tw/topic/2a695119-99c4-4c2c-b509-a4731bd05a2e)）：

讓機器理解人類語音所表達的訊息，一直以來是業界、學界共同努力的方向。語音辨識（speech recognition）技術擁有數十年的研究歷史，在 AI 人工智慧興起後又掀起另一波新浪潮，中文語音辨識是否能有新的突破，快來參加本議題跟大家說你就是中文語音資料的專家。

### 資料集說明
本議題資料集節選自科技部推出的「AI 語音數據資料集」，內容包含：
- 中國四大文學名著：紅樓夢、三國演義、西遊記、水滸傳
- 警察廣播電台的路況報導
- 教育廣播電台的新聞時事

共有 1,751 個音訊檔案分為八個類別，參加者須透過音檔辨識為八大分類的那一種。

## 壹、實際成果

### 一、MLP 基線模型
- **V3_AugLeaky**：達成 58% 準確率

### 二、CNN 最佳模型
- **V25_Heavy_Regularization_Tuned**：達成 75.27% 準確率

### 三、Transformer 最佳模型
- **V9_Wider_Transformer**：達成 85.48% 準確率（Kaggle 0.6866 分）

## 貳、核心發現

### 一、MLP 階段實驗發現

1. **V5_Pooling vs V1_Baseline 實驗**
   - 全域最大池化策略相比直接展平成效不彰（56% vs 55%）
   
2. **MLP 在八分類任務整體成效不彰**
   - 因為將頻譜圖直接展平會丟失垂直特徵

### 二、CNN 階段實驗發現

1. **V6_CNN_Baseline vs MLP 對比**
   - CNN 基線 57.0% 超越 MLP 最佳模型（56%）
   - 足見卷積架構之優勢
   
2. **V7_CNN_Advanced 正規化實驗**
   - 引入 BatchNorm 和 Dropout 後提升至 64.5%
   
3. **V25_Conservative vs V25_Heavy 正則化範圍實驗**
   - 極端保守（lr=5e-5, wd=5e-3）導致欠擬合 69.4%
   - 重度正則化達成 74.19%

### 三、Transformer 階段實驗發現

1. **V1_Baseline vs CNN 架構跨越**
   - Transformer 基線 84.41% 大幅超越 CNN 最佳 75.27%
   - 足見 Transformer 更上一層樓
   
2. **V2_SpecAugment 過度增強實驗**
   - 激進參數導致 73.66% 大幅下降
   - V3 溫和參數回升至 77.96%
   - 過度增強反而破壞其注意力機制
   
3. **V8_Deeper vs V9_Wider 架構對比**
   - 12 層深度模型 84.95% vs 6 層寬度模型（dim_feedforward=4096）85.48%
   - 寬度勝過深度
   
4. **V9 vs V10 標籤平滑對比**
   - V9 原始 85.48% vs V10 標籤平滑 82.26%
   - 高容量模型過度正規化反而產生反效果

## 參、快速導覽

### 專案結構

```
S1_MLP/
└── src/
    ├── config.py           # 實驗配置
    ├── data_pipeline.py    # 數據處理流程
    ├── models.py          # 模型定義
    ├── train_mlp.py       # 訓練腳本
    ├── predict.py         # 預測腳本
    └── utils.py           # 工具函數

S2_CNN/
└── src/
    ├── config.py
    ├── data_pipeline.py
    ├── models.py
    ├── train_cnn.py
    ├── predict.py
    ├── tune_hyperparams.py  # 超參數調優
    └── utils.py

S3_Transformer/
└── src/
    ├── config.py
    ├── data_pipeline.py
    ├── models.py
    ├── train_transformer.py
    └── predict.py

shared_utils/              # 共享工具模組
├── audio_processing.py   # 音頻處理工具
└── early_stopping.py     # early_stopping 實現

experiment_records/        # 實驗記錄文檔
├── S1_MLP實驗記錄.md
├── S2_CNN實驗記錄.md
└── S3_Transformer實驗記錄.md
```
