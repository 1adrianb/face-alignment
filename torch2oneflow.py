import torch
from torch.utils.model_zoo import load_url

import oneflow

from pathlib import Path

class t2o:
    def __init__(self,url_or_path,device="cpu"):
        self.url_or_path = url_or_path
        self.device = device
        self.model_weights = self.load()

    def load(self):
        if Path(self.url_or_path).is_symlink():
            model_weights = load_url(self.url_or_path)
        elif Path(self.url_or_path).is_file():
            model_weights = torch.load(self.url_or_path)
        else:
            print(f'{self.url_or_path} is wrong')
            model_weights = None
        return model_weights

    def convert(self):
        return_model_dict = self.model_weights.copy()
        for key, value in return_model_dict.items():
            val = value.detach().cpu().numpy()
            return_model_dict[key] = val
            print("key:", key, "value.shape", val.shape)
        return return_model_dict

