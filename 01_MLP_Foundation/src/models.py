"""
多層感知器（MLP）模型定義檔案。

本檔案包含不同複雜度的 MLP 架構，從基礎的三層網路到包含正規化的深層網路。
所有模型都針對音訊分類任務進行了優化。
"""
import torch
import torch.nn as nn
import inspect

class V1_Baseline(nn.Module):
    """基礎 MLP 模型：3層全連接網路。"""
    def __init__(self, model_cfg):
        super().__init__()
        input_dim = model_cfg['input_dim']
        output_dim = model_cfg['output_dim']
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)

class V2_Deeper(nn.Module):
    """更深的 MLP 模型：增加神經元數量以提升學習能力。"""
    def __init__(self, model_cfg):
        super().__init__()
        input_dim = model_cfg['input_dim']
        output_dim = model_cfg['output_dim']
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)

class V3_AugLeaky(nn.Module):
    """使用 LeakyReLU 激活函數的深層 MLP 模型。"""
    def __init__(self, model_cfg):
        super().__init__()
        input_dim = model_cfg['input_dim']
        output_dim = model_cfg['output_dim']
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_dim)
        )
        
    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)
        
class V4_AdamW(nn.Module):
    """針對 AdamW 優化器調整的 MLP 模型。"""
    def __init__(self, model_cfg):
        super().__init__()
        input_dim = model_cfg['input_dim']
        output_dim = model_cfg['output_dim']
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)
        
class V5_Pooling(nn.Module):
    """包含 Dropout 正規化的 MLP 模型，支援池化策略。"""
    def __init__(self, model_cfg):
        super().__init__()
        input_h = model_cfg['input_h']
        input_w = model_cfg['input_w']
        output_dim = model_cfg['output_dim']
        dropout_p_1 = model_cfg.get('dropout_p_1', 0.5)
        dropout_p_2 = model_cfg.get('dropout_p_2', 0.3)

        self.flattened_dim = input_h * input_w
        self.model = nn.Sequential(
            nn.Linear(self.flattened_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p_1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p_2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        x = x.view(-1, self.flattened_dim)
        return self.model(x)

# ==============================================================================
#  模型工廠
# ==============================================================================
# 全域的模型註冊表
_MODEL_REGISTRY = {
    'V1_Baseline': V1_Baseline,
    'V2_Deeper': V2_Deeper,
    'V3_AugLeaky': V3_AugLeaky,
    'V4_AdamW': V4_AdamW,
    'V5_Pooling': V5_Pooling,
}

def get_model(model_config: dict):
    """
    模型工廠函式 (已重構)。
    根據給定的完整模型設定物件，實例化並返回對應的模型。

    Args:
        model_config (dict): 一個包含所有模型初始化所需參數的字典。
                             (例如: 'model_class', 'input_dim', 'output_dim' 等)

    Returns:
        torch.nn.Module: 實例化的模型物件。
    """
    model_class_name = model_config.get('model_class')
    if not model_class_name or model_class_name not in _MODEL_REGISTRY:
        raise ValueError(f"無效或未指定模型類別: {model_class_name}")

    model_class = _MODEL_REGISTRY[model_class_name]

    # 所有模型都已更新為接收單一的 model_cfg，因此直接傳遞即可。
    return model_class(model_config) 