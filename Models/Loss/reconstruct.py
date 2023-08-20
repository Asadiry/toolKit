import torch
import torch.nn as nn


def build_loss(**params):
    if params["loss_name"] == "recon_mix_cls":
        alpha = params["alpha"]
        loss = recon_mix_cls(alpha)
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