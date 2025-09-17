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


class SimpleCalibration(nn.Module):
    """Simple Calibration with Temperature and Platt Scaling"""
    def __init__(self, num_diseases=14):
        super().__init__()
        
        # Platt scaling
        self.platt_scale = nn.Parameter(torch.ones(num_diseases))
        self.platt_bias = nn.Parameter(torch.zeros(num_diseases))
        
        # Temperature scaling
        self.temperature = nn.Parameter(torch.ones(num_diseases) * 1.5)
        
    def forward(self, logits, method='temperature'):
        if method == 'platt':
            return logits * self.platt_scale + self.platt_bias
        elif method == 'temperature':
            return logits / (self.temperature + 1e-8)
        else:  # combine both
            temp_scaled = logits / (self.temperature + 1e-8)
            return temp_scaled * self.platt_scale + self.platt_bias
