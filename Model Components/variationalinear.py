# Enhanced Multi-Agent Bayesian Disease Prediction Framework
import sys
import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix, brier_score_loss
from sklearn.calibration import calibration_curve
import math
from scipy.stats import entropy
from sklearn.metrics import average_precision_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
# Add CXR-CLIP model path
module_path = "/home/dawood/lab2_rotaion/cxr-clip/cxrclip/model/"
if module_path not in sys.path:
    sys.path.append(module_path)

from image_classification import CXRClassification

class VariationalLinear(nn.Module):
    """
    FIXED VERSION - Replace your existing VariationalLinear with this
    """
    def __init__(self, in_features, out_features, prior_std=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # BETTER initialization (smaller variance)
        self.weight_mean = nn.Parameter(torch.randn(out_features, in_features) * np.sqrt(1.0 / in_features))
        self.weight_logvar = nn.Parameter(torch.ones(out_features, in_features) * -8.0)  # Much smaller
        
        self.bias_mean = nn.Parameter(torch.zeros(out_features))
        self.bias_logvar = nn.Parameter(torch.ones(out_features) * -8.0)  # Much smaller
        
        self.register_buffer('prior_std', torch.tensor(prior_std))
        
    def forward(self, x, sample_posterior=True):
        device = x.device
        epoch = getattr(self, '_current_epoch', 0)
        
        # More aggressive stochasticity
        if self.training and sample_posterior and epoch:
            # Proper variance sampling
            weight_std = torch.exp(0.5 * self.weight_logvar)
            weight = self.weight_mean + weight_std * torch.randn_like(self.weight_mean)
            
            bias_std = torch.exp(0.5 * self.bias_logvar)
            bias = self.bias_mean + bias_std * torch.randn_like(self.bias_mean)
        else:
            weight = self.weight_mean
            bias = self.bias_mean
        
        output = F.linear(x, weight, bias)
        kl_div = self.compute_kl_divergence()
        
        return output, kl_div