import numpy as np
import torch
import logging

# 為此工具模組設定日誌記錄器
logger = logging.getLogger(__name__)

class EarlyStopping:
    """
    如果驗證損失在給定耐心後沒有改善，則提前停止訓練。
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth'):
        """
        參數:
            patience (int): 驗證損失上次改善後等待多長時間。
                            預設: 7
            verbose (bool): 如果為 True，為每次驗證損失改善打印訊息。
                            預設: False
            delta (float): 監控量的最小變化才算改善。
                           預設: 0
            path (str): 檢查點要保存到的路徑。
                        預設: 'checkpoint.pth'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model to {self.path} ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss 