import torch
import torch.nn as nn

class mae_loss(nn.Module):
    def __init__(self, reduction="mean", compare_keys=None):   
        super(mae_loss, self).__init__()
        self.loss_func = nn.L1Loss(reduction=reduction)
        self.compare_keys = compare_keys

    def forward(self, x):
        value_x = x[self.compare_keys[0]]
        value_y = x[self.compare_keys[1]]
        loss = self.loss_func(value_x, value_y)
        return loss

class mse_loss(nn.Module):
    def __init__(self, reduction="mean"):
        super(mse_loss, self).__init__()
        self.loss_func = nn.MSELoss(reduction=reduction)
    
    def forward(self, x):
        old_values = x["old_value"]
        new_values = x["new_value"]
        loss = self.loss_func(old_values, new_values)
        return loss

class ce_loss(nn.Module):
    def __init__(self, reduction="mean"):
        super(ce_loss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, x):
        labels = x["label"]
        predicted = x["class_probs"]
        loss = self.loss_func(predicted, labels.long())
        return loss

class recon_mix_cls(nn.Module):
    def __init__(self, alpha):
        super(recon_mix_cls, self).__init__()
        self.alpha = alpha
        self.recon_func = mse_loss()
        self.cls_func = ce_loss()

    def forward(self, x):
        loss = self.recon_func(x) +  self.alpha * self.cls_func(x)
        return loss

class recon_mix_reg(nn.Module):
    def __init__(self, alpha):
        super(recon_mix_cls, self).__init__()
        self.alpha = alpha
        self.recon_func = mse_loss()
        self.reg_func = mse_loss()

    def forward(self, x):
        loss = self.recon_func(x) +  self.alpha * self.reg_func(x)
        return loss