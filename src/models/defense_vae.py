

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image



class Defence_VAE(nn.Module):
    def __init__(self, vae, model):
        super(Defence_VAE, self).__init__()
        self.vae = vae
        self.model = model

    def forward(self, x):
        x = self.vae(x)
        x = self.model(x)
        return x