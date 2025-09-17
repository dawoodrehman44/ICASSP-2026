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

from variationalinear import VariationalLinear
from consistency_validation import EnhancedBayesianConsistencyAgent
from bayesianencoder import HierarchicalBayesianEncoder
from classification_network import BayesianDiseaseClassificationAgent
from calibration import SimpleCalibration


class EnhancedBayesianFramework(nn.Module):
    """Simplified Framework without redundant components"""
    def __init__(self, input_dim, num_diseases=14):
        super().__init__()
        
        # Keep Bayesian encoder for feature extraction
        self.bayesian_encoder = HierarchicalBayesianEncoder(input_dim, num_hierarchy_levels=3)
        final_dim = input_dim // 8  # 2^3
        
        # Simplified agents
        self.classification_agent = BayesianDiseaseClassificationAgent(final_dim, num_diseases)
        self.consistency_agent = EnhancedBayesianConsistencyAgent(final_dim, num_diseases)
        self.calibration = SimpleCalibration(num_diseases)
        
    def forward(self, features, return_all_outputs=False):
        # Encode features
        encoded = self.bayesian_encoder(features)
        final_features = encoded['aggregated_features']
        
        # Get predictions and uncertainties
        class_output = self.classification_agent(final_features)
        
        # Check consistency
        consistency_output = self.consistency_agent(
            final_features,
            class_output['logits'],
            class_output
        )
        
        # Calibrate
        calibrated_logits = self.calibration(class_output['logits'], method='temperature')
        
        # Collect KL from encoder only
        kl_divergences = encoded['kl_divergences']
        
        outputs = {
            'disease_logits': calibrated_logits,
            'raw_logits': class_output['logits'],
            'class_uncertainties': {
                'epistemic_uncertainty': class_output['epistemic_uncertainty'],
                'aleatoric_uncertainty': class_output['aleatoric_uncertainty'],
                'total_uncertainty': class_output['total_uncertainty']
            },
            'consistency_score': consistency_output['consistency_score'],
            'feature_consistency': consistency_output['feature_consistency'],
            'uncertainty_consistency': consistency_output['uncertainty_consistency'],
            'kl_divergences': kl_divergences
        }
        
        return outputs