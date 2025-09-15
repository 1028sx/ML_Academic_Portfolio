import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """位置編碼計算"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 產生一個 (max_len, d_model) 的位置編碼矩陣
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term) # 偶數維度使用 sin
        pe[:, 1::2] = torch.cos(position * div_term) # 奇數維度使用 cos
        self.register_buffer('pe', pe.unsqueeze(0))
        self.pe: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """將位置編碼加到輸入張量"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class AudioTransformer(nn.Module):
    """接收梅爾頻譜圖並利用 Transformer Encoder 學習序列中的時間依賴性。 """
    def __init__(self, model_cfg):
        super().__init__()
        d_model = model_cfg['d_model']
        nhead = model_cfg['nhead']
        num_encoder_layers = model_cfg['num_encoder_layers']
        dim_feedforward = model_cfg['dim_feedforward']
        num_classes = model_cfg['num_classes']
        dropout = model_cfg['dropout']
        max_len = model_cfg['max_len']
        classification_mode = model_cfg.get('classification_mode', 'mean')

        self.d_model = d_model
        self.classification_mode = classification_mode

        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True # 讓輸入維度為 (N, S, E)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        if self.classification_mode == 'cls':
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.final_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.classifier.bias.data.zero_()
        self.classifier.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """ 回傳Tensor, shape [batch_size, num_classes] """
        # 維度重塑與調整
        src = src.squeeze(1).permute(0, 2, 1)

        # 應用縮放與位置編碼
        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        # 根據分類模式處理輸入
        if self.classification_mode == 'cls':
            # 將 cls_token 擴展到與 batch_size 相同
            cls_tokens = self.cls_token.expand(src.size(0), -1, -1)
            # 拼接到序列的最前面
            src = torch.cat((cls_tokens, src), dim=1)

        # 通過 Transformer Encoder
        output = self.transformer_encoder(src)

        # 根據分類模式選擇輸出並進行分類
        if self.classification_mode == 'cls':
            # 只取 [CLS] token 的輸出 (在序列的第一個位置)
            output = output[:, 0]
        else: # 對序列維度取平均
            output = output.mean(dim=1)

        output = self.final_dropout(output)
        output = self.classifier(output)
        return output

def get_model(model_config: dict, audio_settings: dict | None = None) -> nn.Module:
    """ 工廠函式"""
    return AudioTransformer(model_config)