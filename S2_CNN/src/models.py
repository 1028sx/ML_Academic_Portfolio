import torch
import torch.nn as nn

class V6_CNN_Baseline(nn.Module):
    # 基礎 CNN 模型：雙層卷積網路
    def __init__(self, model_cfg):
        super().__init__()
        input_h = model_cfg['input_h']
        input_w = model_cfg['input_w']
        num_classes = model_cfg['output_dim']
        self.features = nn.Sequential(
            # 卷積層 1
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 池化層 1
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 卷積層 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # 池化層 2
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_h, input_w)
            flattened_dim = self.features(dummy_input).view(-1).shape[0]

        # 全連接層
        self.classifier = nn.Sequential(
            nn.Linear(flattened_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # 特徵提取
        x = self.features(x)
        # 展平
        x = x.view(x.size(0), -1)
        # 分類
        x = self.classifier(x)
        return x

class V7_CNN_Advanced(nn.Module):
    # 加入 BatchNorm 和 Dropout
    def __init__(self, model_cfg):
        super().__init__()
        input_h = model_cfg['input_h']
        input_w = model_cfg['input_w']
        num_classes = model_cfg['output_dim']
        dropout_p = model_cfg['dropout_p']

        # 特徵提取層
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

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
        features = self.classifier[0](x)

        # 繼續正常的分類流程
        x = self.classifier[1](features) # ReLU 激活
        x = self.classifier[2](x)      # Dropout 層
        logits = self.classifier[3](x) # 最終線性層

        if return_features:
            return logits, features
        return logits

class V11_DeeperCNN(nn.Module):
    # 更深 CNN 模型：3個卷積塊
    def __init__(self, model_cfg):
        super().__init__()
        input_h = model_cfg['input_h']
        input_w = model_cfg['input_w']
        num_classes = model_cfg['output_dim']
        dropout_p = model_cfg['dropout_p']
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

        # 沒有提供固定的 flattened_dim才動態計算
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

        features = self.classifier[0](x)

        x = self.classifier[1](features) # ReLU 激活
        x = self.classifier[2](x)      # Dropout 層
        logits = self.classifier[3](x) # 最終線性層

        if return_features:
            return logits, features
        return logits

class V12_OptimizedCNN(V11_DeeperCNN):
    # 繼承 V11_DeeperCNN
    pass

# 模型工廠
_MODEL_REGISTRY = {
    'V6_CNN_Baseline': V6_CNN_Baseline,
    'V7_CNN_Advanced': V7_CNN_Advanced,
    'V11_DeeperCNN': V11_DeeperCNN,
    'V12_OptimizedCNN': V12_OptimizedCNN,
}

def get_model(model_config: dict):
    """模型工廠函式"""
    model_class_name = model_config['model_class']
    model_class = _MODEL_REGISTRY[model_class_name]
    return model_class(model_config)