ML Academic Portfolio - 中文語音分類學習專案

這是一個以深度學習實作語音八分類任務的學術專案，通過MLP→CNN→Transformer的順序展現從基礎到進階的學習軌跡。
該議題完整描述如下：(來自AIdea平台，Link：https://aidea-web.tw/topic/2a695119-99c4-4c2c-b509-a4731bd05a2e)
讓機器理解人類語音所表達的訊息，一直以來是業界、學界共同努力的方向。語音辨識（speech recognition）技術擁有數十年的研究歷史，在AI人工智慧興起後又掀起另一波新浪潮，中文語音辨識是否能有新的突破，快來參加本議題跟大家說你就是中文語音資料的專家。

本議題資料集節選自科技部推出的「AI語音數據資料集」，內容包含中國四大文學名著，紅樓夢、三國演義、西遊記、水滸傳，以及警察廣播電台的路況報導，和教育廣播電台的新聞時事等。共有1,751個音訊檔案分為八個類別，參加者須透過音檔辨識為八大分類的那一種。

壹、實際成果：
    一、MLP基線模型: V3_AugLeaky達成58%準確率
    二、CNN最佳模型: V25_Heavy_Regularization_Tuned達成75.27%準確率
    三、Transformer最佳模型: V9_Wider_Transformer達成85.48%準確率 (Kaggle 0.6866分)

貳、核心發現：
    一、MLP階段實驗發現：
        (一)V5_Pooling vs V1_Baseline實驗：全域最大池化策略相比直接展平成效不彰(56% vs 55%)。
        (二)MLP在八分類任務整體成效不彰，因為將頻譜圖直接展平會丟失垂直特徵。
    二、CNN階段實驗發現：
        (一)V6_CNN_Baseline vs MLP對比：CNN基線57.0%超越MLP最佳模型(56%)，足見卷積架構之優勢。
        (二)V7_CNN_Advanced正規化實驗：引入BatchNorm和Dropout後提升至64.5%。
        (三)V25_Conservative vs V25_Heavy正則化範圍實驗：極端保守(lr=5e-5, wd=5e-3)導致欠擬合69.4%，重度正則化達成74.19%。
    三、Transformer階段實驗發現：
        (一)V1_Baseline vs CNN架構跨越：Transformer基線84.41%大幅超越CNN最佳75.27%，足見Transformer更上一層樓。
        (二)V2_SpecAugment過度增強實驗：激進參數導致73.66%大幅下降，V3溫和參數回升至77.96%，過度增強反而破壞其注意力機制。
        (三)V8_Deeper vs V9_Wider架構對比：12層深度模型84.95% vs 6層寬度模型(dim_feedforward=4096)85.48%，寬度勝過深度。
        (四)V9 vs V10標籤平滑對比：V9原始85.48% vs V10標籤平滑82.26%，高容量模型過度正規化反而產生反效果。


參、快速導覽

S1_MLP/
└── src/
    ├── config.py       # 實驗配置
    ├── data_pipeline.py # 數據處理流程
    ├── models.py       # 模型定義
    ├── train_mlp.py    # 訓練腳本
    ├── predict.py      # 預測腳本
    └── utils.py        # 工具函數

S2_CNN/
└── src/
    ├── config.py
    ├── data_pipeline.py
    ├── models.py
    ├── train_cnn.py
    ├── predict.py 
    ├── tune_hyperparams.py # 超參數調優
    └── utils.py

S3_Transformer/
└── src/
    ├── config.py
    ├── data_pipeline.py
    ├── models.py
    ├── train_transformer.py
    └── predict.py

shared_utils/           # 共享工具模組
├── audio_processing.py # 音頻處理工具
└── early_stopping.py   # early_stopping實現

experiment_records/
├── S1_MLP實驗記錄.md
├── S2_CNN實驗記錄.md
└── S3_Transformer實驗記錄.md
