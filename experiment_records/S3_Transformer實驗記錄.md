Transformer階段實驗記錄

本記錄詳細追蹤Transformer模型從基線建立、資料增強探索、架構優化到正規化平衡的完整演進過程。每個實驗版本探索注意力機制在音頻分類任務上的潛力，最終達成專案SOTA效能。

V1_Transformer_Baseline
    建立標準Transformer基準模型，採用6層編碼器、8頭注意力、2048維前饋網路。關鍵修復：將頻譜圖轉為分貝尺度解決nan loss問題。
    驗證準確率：84.41%

V2_SpecAugment_Aggressive
    在V1基礎上引入激進SpecAugment參數。頻率遮罩27、時間遮罩70，各2次。過度增強導致關鍵聲學特徵被抹除，效能大幅下降。
    驗證準確率：84.41% -> 73.66%

V3_SpecAugment_Gentle
    採用溫和SpecAugment參數修正V2失敗。頻率遮罩10、時間遮罩30，各1次。證明適度增強有效，但仍未超越基準。
    驗證準確率：77.96%

V4_CLS_Token
    實驗CLS token分類策略替代平均池化。在輸入序列前加入可學習CLS token用於分類。出現過擬合跡象，效能下降。
    驗證準確率：80.11%

V5_OneCycleLR_Aggressive
    引入OneCycleLR學習率排程器，max_lr=1e-3。學習率過於激進導致訓練不穩定，模型無法收斂至良好解。
    驗證準確率：74.73%

V6_OneCycleLR_Conservative
    降低OneCycleLR最大學習率至1e-4。學習率問題得到改善，效能回升，但仍未超越固定學習率基準。
    驗證準確率：80.65%

V7_Label_Smoothing
    在V3基礎上引入標籤平滑0.1。成功壓制訓練集準確率防止過度自信，但驗證效能未提升，暗示瓶頸在模型容量而非過擬合。
    驗證準確率：82.80%

V8_Deeper_Transformer
    增加模型深度至12層編碼器，測試深度對效能影響。更深模型訓練不穩定且過擬合風險增加，未能超越6層基準。
    驗證準確率：84.95%

V9_Wider_Transformer
    增加前饋網路寬度至4096維。加寬策略成功突破效能瓶頸，創下新紀錄，但伴隨過擬合跡象。
    驗證準確率：85.48%

V10_Wider_Label_Smoothing
    在V9寬模型基礎上加入標籤平滑。過度正規化限制高容量模型學習能力，效能顯著下降，證明正規化需適度。
    驗證準確率：82.26%

V11_Strong_Regularization
    在V9基礎上大幅增加dropout(0.25)和權重衰減(1e-2)。強力正規化導致嚴重欠擬合，為最佳正規化強度劃定上界。
    驗證準確率：76.88%

V12_Balanced_Regularization
    採用中等強度正規化，dropout=0.2、weight_decay=1e-4。成功找到正規化平衡點，訓練穩定但未達V9峰值效能。
    驗證準確率：82.80%

V13_Balanced_OneCycleLR
    在V12平衡正規化基礎上引入溫和OneCycleLR。動態學習率增加不穩定性未帶來提升，最終儲存模型效能退步。
    驗證準確率：80.11%