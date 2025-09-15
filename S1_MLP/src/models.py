import torch
import torch.nn as nn
import inspect

class V1_Baseline(nn.Module):
    # 3層全連接
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
    #增加神經元
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
    # LeakyReLU 激活函數
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
    # AdamW 優化
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
    # Dropout
    def __init__(self, model_cfg):
        super().__init__()
        input_h = model_cfg['input_h']
        input_w = model_cfg['input_w']
        output_dim = model_cfg['output_dim']
        dropout_p_1 = model_cfg['dropout_p_1']
        dropout_p_2 = model_cfg['dropout_p_2']

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

# 模型註冊表
_MODEL_REGISTRY = {
    'V1_Baseline': V1_Baseline,
    'V2_Deeper': V2_Deeper,
    'V3_AugLeaky': V3_AugLeaky,
    'V4_AdamW': V4_AdamW,
    'V5_Pooling': V5_Pooling,
}

def get_model(model_config: dict):
    model_class_name = model_config['model_class']
    model_class = _MODEL_REGISTRY[model_class_name]
    return model_class(model_config)