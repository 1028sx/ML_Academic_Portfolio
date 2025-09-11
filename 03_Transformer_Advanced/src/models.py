import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    位置編碼 (Positional Encoding)

    Transformer 模型本身無法感知序列中 token 的順序。為了引入順序資訊，
    我們在輸入嵌入中加入「位置編碼」。這個編碼不是學習出來的，而是使用
    不同頻率的 sin 和 cos 函數直接計算得出。

    數學公式：
    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    其中 pos 是 token 在序列中的位置，i 是 embedding 維度中的索引。
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 產生一個 (max_len, d_model) 的位置編碼矩陣
        position = torch.arange(max_len).unsqueeze(1)
        # 計算分母中的除法項，即 10000^(2i / d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term) # 偶數維度使用 sin
        pe[:, 1::2] = torch.cos(position * div_term) # 奇數維度使用 cos
        
        # 將 pe 的 shape 變為 (1, max_len, d_model) 以符合批次輸入
        # register_buffer 將 pe 註冊為模型的一部分，它會被儲存和載入，但不會被視為模型參數（即不參與梯度更新）
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        將位置編碼加到輸入張量上。
        Args:
            x: 輸入張量，shape 為 [batch_size, seq_len, d_model]
        """
        # x.size(1) 是輸入序列的實際長度
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class AudioTransformer(nn.Module):
    """
    用於音訊分類的 Transformer 模型。

    這個模型接收梅爾頻譜圖作為輸入，將其視為一個序列，
    並利用 Transformer Encoder 學習序列中的時間依賴性。

    它支援兩種分類策略:
    - 'mean': 對 Encoder 的所有輸出時間步取平均值，然後進行分類。
    - 'cls': 在序列開頭加入一個可學習的 [CLS] token，只使用該 token 的最終輸出進行分類。
    """
    def __init__(self, d_model: int, nhead: int, num_encoder_layers: int, 
                 dim_feedforward: int, num_classes: int, dropout: float,
                 max_len: int, classification_mode: str = 'mean'):
        super().__init__()
        self.d_model = d_model
        self.classification_mode = classification_mode
        
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True # <<< 非常重要！讓輸入維度為 (N, S, E)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        
        # 根據分類模式，決定是否需要 [CLS] token
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
        """
        Args:
            src: Tensor, 來自 DataLoader 的原始輸入，
                 shape 為 [batch_size, 1, n_mels, seq_len]
        
        Returns:
            Tensor, shape [batch_size, num_classes]
        """
        # 1. 維度重塑與調整
        src = src.squeeze(1).permute(0, 2, 1)

        # 2. 應用縮放與位置編碼 (在 CLS token 加入前)
        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # 3. 根據分類模式處理輸入
        if self.classification_mode == 'cls':
            # 將 cls_token 擴展到與 batch_size 相同
            cls_tokens = self.cls_token.expand(src.size(0), -1, -1)
            # 將 cls_token 拼接到序列的最前面
            src = torch.cat((cls_tokens, src), dim=1)

        # 4. 通過 Transformer Encoder
        output = self.transformer_encoder(src)
        
        # 5. 根據分類模式選擇輸出並進行分類
        if self.classification_mode == 'cls':
            # 只取 [CLS] token 的輸出 (在序列的第一個位置)
            output = output[:, 0]
        else: # 'mean' mode
            # 對序列維度取平均
            output = output.mean(dim=1) 
        
        output = self.final_dropout(output)
        output = self.classifier(output)
        return output

def get_model(model_config: dict, audio_settings: dict) -> nn.Module:
    """
    工廠函式，根據組合後的設定字典建立模型。
    """
    model_class_name = model_config.get('model_class', 'AudioTransformer')
    
    if model_class_name == 'AudioTransformer':
        return AudioTransformer(
            d_model=model_config['d_model'],
            nhead=model_config['nhead'],
            num_encoder_layers=model_config['num_encoder_layers'],
            dim_feedforward=model_config['dim_feedforward'],
            dropout=model_config['dropout'],
            classification_mode=model_config['classification_mode'],
            num_classes=audio_settings['num_classes'],
            max_len=audio_settings['target_len']
        )
    else:
        raise ValueError(f"未知的模型類別: {model_class_name}") 