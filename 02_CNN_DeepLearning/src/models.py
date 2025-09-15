"""
卷積神經網路（CNN）模型定義檔案。

本檔案包含不同複雜度的 CNN 架構，從基礎的雙層卷積到包含正規化和高級損失函式的深層網路。
所有模型都針對音訊頻譜圖分類任務進行了優化。
"""
import torch
import torch.nn as nn
import inspect

class V6_CNN_Baseline(nn.Module):
    """基礎 CNN 模型：雙層卷積網路。"""
    def __init__(self, model_cfg):
        super().__init__()
        input_h = model_cfg['input_h']
        input_w = model_cfg['input_w']
        num_classes = model_cfg['output_dim'] # 標準化命名

        # 特徵提取層 (卷積層 + 池化層)
        self.features = nn.Sequential(
            # 第一個卷積層
            # 輸入: (N, 1, 128, 215)
            # 輸出: (N, 16, 128, 215)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 第一個池化層
            # 輸入: (N, 16, 128, 215)
            # 輸出: (N, 16, 64, 107)
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第二個卷積層
            # 輸入: (N, 16, 64, 107)
            # 輸出: (N, 32, 64, 107)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 第二個池化層
            # 輸入: (N, 32, 64, 107)
            # 輸出: (N, 32, 32, 53)
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 為了讓模型更具彈性，我們動態計算展平後的維度
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_h, input_w)
            flattened_dim = self.features(dummy_input).view(-1).shape[0]

        # 分類層 (全連接層)
        self.classifier = nn.Sequential(
            nn.Linear(flattened_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # 1. 通過特徵提取層
        x = self.features(x)
        # 2. 展平特徵圖
        x = x.view(x.size(0), -1)
        # 3. 通過分類層
        x = self.classifier(x)
        return x

class V7_CNN_Advanced(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        input_h = model_cfg['input_h']
        input_w = model_cfg['input_w']
        num_classes = model_cfg['output_dim']
        dropout_p = model_cfg.get('dropout_p', 0.5) # 使用 .get 以保持向下相容性

        # 特徵提取層 (加入批次正規化)
        self.features = nn.Sequential(
            # 區塊 1
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 區塊 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_h, input_w)
            flattened_dim = self.features(dummy_input).view(-1).shape[0]

        # 分類層 (加入 Dropout)
        self.classifier = nn.Sequential(
            nn.Linear(flattened_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, return_features=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        
        # 為了 VCReg，我們需要分類層的第一個線性變換後的特徵
        features = self.classifier[0](x)
        
        # 繼續正常的分類流程
        x = self.classifier[1](features) # ReLU 激活
        x = self.classifier[2](x)      # Dropout 層
        logits = self.classifier[3](x) # 最終線性層
        
        if return_features:
            return logits, features
        return logits

class V11_DeeperCNN(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        input_h = model_cfg['input_h']
        input_w = model_cfg['input_w']
        num_classes = model_cfg['output_dim']
        dropout_p = model_cfg.get('dropout_p', 0.5)
        flattened_dim = model_cfg.get('flattened_dim')

        # 特徵提取層 (3個區塊)
        self.features = nn.Sequential(
            # 區塊 1
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 區塊 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 區塊 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 如果沒有提供固定的 flattened_dim，才動態計算
        if flattened_dim is None:
            with torch.no_grad():
                dummy_input = torch.zeros(1, 1, input_h, input_w)
                self.flattened_dim = self.features(dummy_input).view(-1).shape[0]
        else:
            self.flattened_dim = flattened_dim

        # 分類層 (加寬並加入 Dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, return_features=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        # 同樣為 VCReg 提取特徵
        features = self.classifier[0](x)
        
        x = self.classifier[1](features) # ReLU 激活
        x = self.classifier[2](x)      # Dropout 層
        logits = self.classifier[3](x) # 最終線性層

        if return_features:
            return logits, features
        return logits

class V12_OptimizedCNN(V11_DeeperCNN):
    pass

# ==============================================================================
#  模型工廠
# ==============================================================================
# 全域的模型註冊表
_MODEL_REGISTRY = {
    'V6_CNN_Baseline': V6_CNN_Baseline,
    'V7_CNN_Advanced': V7_CNN_Advanced,
    'V11_DeeperCNN': V11_DeeperCNN,
    'V12_OptimizedCNN': V12_OptimizedCNN,
}

def get_model(model_config: dict):
    """
    模型工廠函式 (已重構)。
    根據給定的完整模型設定物件，實例化並返回對應的模型。

    Args:
        model_config (dict): 一個包含所有模型初始化所需參數的字典。
                             (例如: 'model_class', 'input_h', 'output_dim', 'dropout_p' 等)

    Returns:
        torch.nn.Module: 實例化的模型物件。
    """
    model_class_name = model_config.get('model_class')
    if not model_class_name or model_class_name not in _MODEL_REGISTRY:
        raise ValueError(f"無效或未指定模型類別: {model_class_name}")

    model_class = _MODEL_REGISTRY[model_class_name]
    
    # 檢查 __init__ 是否只接收 'model_cfg'
    sig = inspect.signature(model_class.__init__)
    if 'model_cfg' in sig.parameters and len(sig.parameters) == 2: # (self, model_cfg)
        # 對於新式模型 (如 V7)，直接傳遞完整的設定字典
        # Note: 經過我們的修改，V6 現在也屬於新式模型了
        return model_class(model_config)
    else:
        # 對於舊式模型，智能地匹配參數
        required_args = {p.name for p in sig.parameters.values() if p.name != 'self'}
        init_args = {key: value for key, value in model_config.items() if key in required_args}
        
        # 檢查是否有遺漏的必要參數
        missing_args = required_args - set(init_args.keys())
        if missing_args:
            raise ValueError(f"模型 '{model_class_name}' 初始化缺少必要參數: {missing_args}")

        return model_class(**init_args) 