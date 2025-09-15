# Early Stopping
import numpy as np
import torch
import logging

# 設定 logger
logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth'):
        """初始化 Early Stopping"""
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        """使用當前驗證損失呼叫 early stopping"""
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info(f'Early stopping triggered after {self.patience} epochs of no improvement')
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """當驗證損失下降時儲存模型"""
        if self.verbose:
            message = f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model to {self.path}'
            logger.info(message)
            print(message)

        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

    def reset(self):
        """重置 early stopping 狀態。"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

        if self.verbose:
            logger.info('EarlyStopping state has been reset')