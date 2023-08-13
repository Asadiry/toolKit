import os
import torch

def save_model_weights(model, save_root, save_name):
    save_path = os.path.join(save_root, save_name)
    torch.save(model.state_dict(), save_path)

def load_model_weights(model, file_root, file_name):
    file_path = os.path.join(file_root, file_name)
    model_weights = torch.load(file_path)
    model.load_state_dict(model_weights)