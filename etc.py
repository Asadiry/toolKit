import os
import torch
import numpy as np

def save_model_weights(model, save_root, save_name):
    save_path = os.path.join(save_root, save_name)
    torch.save(model.state_dict(), save_path)

def load_model_weights(model, file_root, file_name):
    file_path = os.path.join(file_root, file_name)
    model_weights = torch.load(file_path)
    model.load_state_dict(model_weights)

class EarlyStopping:
    """Early stops the training if test loss doesn't improve after a given patience."""
    def __init__(self, save_path, patience=7, verbose=False, delta=0):
        """ Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.delta = delta

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.test_loss_min = np.Inf

    def __call__(self, test_loss, model):
        score = -test_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(test_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(test_loss, model)
            self.counter = 0

    def save_checkpoint(self, test_loss, model):
        '''Saves model when test loss decrease.'''
        if self.verbose:
            print(f'Test loss decreased ({self.test_loss_min:.6f} --> {test_loss:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, "checkpoints.pth")
        torch.save(model.state_dict(), path)
        self.test_loss_min = test_loss

