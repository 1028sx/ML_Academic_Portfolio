# ML Academic Portfolio - 中文語音分類學習專案

這是一個記錄深度學習學習過程的學術專案，通過MLP→CNN→Transformer的漸進式技術探索，展現從基礎到進階的完整學習軌跡。

## 專案特色

### 🎯 學習導向
- 重視**學習過程**勝過完美結果
- 記錄**實驗失敗**和反思心得
- 展現**技術理解**的真實深度

### 📊 實際成果
- **MLP基線**: ~50% 準確率
- **CNN進階**: 74.19% 準確率  
- **Transformer最佳**: 85.48% 準確率 (Kaggle 0.6866分)

### 🔬 核心發現
- **寬勝於深**: Transformer寬度比深度更重要
- **失敗分析**: SpecAugment、OneCycleLR等技術的適用性思考
- **任務特性**: 中文語音分類的技術選擇考量

## 快速導覽

### 📁 核心代碼模組
```
01_MLP_Foundation/     # 經典機器學習基礎
02_CNN_DeepLearning/   # 卷積神經網路探索  
03_Transformer_Advanced/ # 注意力機制學習
04_Data/               # 數據管理
```

### 📋 學習記錄文檔
```
07_Documentation/
├── learning_process/        # 真實學習歷程
│   ├── Real_Model_Analysis.md
│   ├── Reflection_and_Future_Directions.md
│   └── experimental_failures.md
├── academic_preparation/    # 學術準備材料
│   ├── Interview_Preparation_Guide.md  
│   └── field_connections.md
└── technical_guides/       # 技術學習指導
    ├── seven_day_plan.md
    └── DAY2_COMPLETE_GUIDE.md
```

### 🎨 展示材料
```
06_Academic_Presentation/
├── academic_presentation_final.html  # 互動式技術展示
└── 展示說明.md                      # 使用說明
```

## 使用方式

### 快速開始
1. **查看專案概覽**: 閱讀本 README.md
2. **理解核心原則**: 閱讀 [CLAUDE.md](CLAUDE.md)
3. **瀏覽技術展示**: 開啟 `06_Academic_Presentation/academic_presentation_final.html`

### 深入了解
1. **學習歷程**: 查看 `07_Documentation/learning_process/` 
2. **技術細節**: 瀏覽各代碼模組
3. **面試準備**: 參考 `07_Documentation/academic_preparation/`

### 環境設置
```bash
# 創建虛擬環境
python -m venv venv_academic

# 啟動環境 (Windows)
venv_academic\Scripts\activate

# 安裝依賴
pip install -r requirements.txt
```

## 學術價值

### 研究所申請適用性
- ✅ 系統性學習方法展示
- ✅ 技術深度理解證明  
- ✅ 實驗方法論應用
- ✅ 誠實的自我評估

### 面試展示重點
1. **技術演進邏輯**: MLP→CNN→Transformer的選擇原因
2. **實驗失敗反思**: 展現問題解決能力
3. **理論實踐結合**: 技術選擇的思考過程
4. **未來學習規劃**: 與目標領域的連接分析

## 學術領域連接

基於2024年研究現狀分析：
- **實驗語音學**: 最直接相關 ⭐⭐⭐⭐⭐
- **語料庫語言學**: 良好連接 ⭐⭐⭐⭐
- **教育科技**: 應用潛力大 ⭐⭐⭐⭐
- **計算語言學**: 需要基礎補強 ⭐⭐⭐

詳細分析見: [field_connections.md](07_Documentation/academic_preparation/field_connections.md)

## 核心原則

1. **理解優先**: 所有內容不超出實際理解水平
2. **可解釋性**: 每個技術點都能向教授清楚說明  
3. **學習過程導向**: 重視journey勝過destination
4. **誠實學術連接**: 基於真實研究現狀評估未來發展
5. **專業溝通**: 避免誇大，保持學術誠信

## 聯絡資訊

**專案性質**: 個人學習專案  
**時間跨度**: 2024年深度學習探索  
**技術棧**: Python, PyTorch, Transformers, Jupyter  
**文檔更新**: 持續維護

---

**致謝**: 感謝所有在學習過程中提供幫助的資源和社群。這個專案記錄了從零開始學習深度學習的真實歷程，希望能對其他學習者有所啟發。