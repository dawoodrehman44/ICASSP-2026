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


class BayesianDiseaseClassificationAgent(nn.Module):
    """Simplified Disease Classification with Uncertainty Quantification"""
    def __init__(self, input_dim, num_diseases=14, num_mc_samples=10):
        super().__init__()
        self.num_diseases = num_diseases
        self.num_mc_samples = num_mc_samples
        
        # Simple deterministic classifier
        hidden_dim = input_dim // 2
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_diseases)
        )
        
        # Keep only epistemic and aleatoric uncertainty networks
        self.epistemic_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_diseases),
            nn.Softplus()
        )
        
        self.aleatoric_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_diseases),
            nn.Softplus()
        )
        
    def forward(self, features, return_distribution=False):
        class_logits = self.classifier(features)
        
        # Use dropout at inference for proper epistemic uncertainty
        if not self.training:
            self.eval()
            with torch.no_grad():
                # Enable dropout for MC sampling
                for module in self.classifier.modules():
                    if isinstance(module, nn.Dropout):
                        module.train()
                
                mc_predictions = []
                for _ in range(20):  # More samples
                    mc_predictions.append(self.classifier(features))
                
                mc_predictions = torch.stack(mc_predictions)
                epistemic_uncertainty = torch.var(torch.sigmoid(mc_predictions), dim=0)
        else:
            epistemic_uncertainty = self.epistemic_net(features)
        
        # Aleatoric uncertainty
        aleatoric_uncertainty = self.aleatoric_net(features)
        
        return {
            'logits': class_logits,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'total_uncertainty': epistemic_uncertainty + aleatoric_uncertainty
        }