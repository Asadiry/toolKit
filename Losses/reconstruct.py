import torch
import torch.nn as nn


class mse_loss(nn.Module):
    def __init__(self):
        super(mse_loss, self).__init__()
        self.loss_func = nn.MSELoss()
    
    def forward(self, x):
        old_vaules = x["old_values"]
        new_values = x["new_values"]
        loss = self.loss_func(old_vaules, new_values)
        return loss

class ce_loss(nn.Module):
    def __init__(self):
        super(ce_loss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x):
        labels = x["labels"]
        predicted = x["predicted"]
        loss = self.loss_func(predicted, labels)
        return loss

class recon_mix_cls(nn.Module):
    def __init__(self, alpha):
        super(recon_mix_cls, self).__init__()
        self.alpha = alpha
        self.recon_func = mse_loss()
        self.cls_func = ce_loss()

    def forward(self, x):
        loss = self.recon_func(x) + self.alpha * self.cls_func(x)
        return loss