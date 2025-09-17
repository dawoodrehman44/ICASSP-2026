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

from variationalinear import VariationalLinear

class HierarchicalBayesianEncoder(nn.Module):
    """Enhanced Hierarchical Bayesian Encoder with Multi-Scale Feature Extraction"""
    def __init__(self, input_dim, num_hierarchy_levels=3, dropout_rate=0.1):
        super().__init__()
        self.hierarchy_levels = num_hierarchy_levels
        
        # Multi-scale Bayesian layers
        dims = [input_dim // (2**i) for i in range(num_hierarchy_levels + 1)]
        
        self.bayesian_layers = nn.ModuleList([
            VariationalLinear(dims[i], dims[i+1])
            for i in range(num_hierarchy_levels)
        ])
        
        # Residual connections for gradient flow
        self.residual_projections = nn.ModuleList([
            nn.Linear(dims[i], dims[i+1]) if dims[i] != dims[i+1] else nn.Identity()
            for i in range(num_hierarchy_levels)
        ])
        
        # Feature aggregation with attention
        self.attention_weights = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dims[i+1], 1),
                nn.Sigmoid()
            ) for i in range(num_hierarchy_levels)
        ])
        
        self.activation = nn.GELU()  # Better than ReLU for Bayesian networks
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.ModuleList([
            nn.LayerNorm(dims[i+1]) for i in range(num_hierarchy_levels)
        ])
        
    def forward(self, x):
        hierarchical_features = []
        hierarchical_kl = []
        attention_scores = []
        
        current_features = x
        for i, (layer, residual, attention, norm) in enumerate(
            zip(self.bayesian_layers, self.residual_projections, 
                self.attention_weights, self.layer_norm)):
            
            # Bayesian transformation
            # features, kl_div = layer(current_features)
            sample_flag = self.training
            features, kl_div = layer(current_features, sample_posterior=sample_flag)
            features = self.activation(features)
            
            # Residual connection
            residual_features = residual(current_features)
            features = features + residual_features
            
            # Layer normalization
            features = norm(features)
            features = self.dropout(features)
            
            # Attention-based feature importance
            att_score = attention(features)
            attention_scores.append(att_score)
            
            hierarchical_features.append(features)
            hierarchical_kl.append(kl_div)
            
            current_features = features
            
        # Weighted feature aggregation
        aggregated_features = torch.zeros_like(hierarchical_features[-1])
        total_attention = sum(attention_scores)
        
        for feat, att in zip(hierarchical_features, attention_scores):
            if feat.shape == aggregated_features.shape:
                aggregated_features += feat * (att / (total_attention + 1e-8))
            
        return {
            'features': hierarchical_features,
            'kl_divergences': hierarchical_kl,
            'final_features': current_features,
            'aggregated_features': aggregated_features,
            'attention_scores': attention_scores
        }