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



### The improved and new_one
class EnhancedMultiObjectiveLoss(nn.Module):
    """
    Updated loss function with better weight balance
    """
    def __init__(self, num_diseases=14):
        super().__init__()
        self.num_diseases = num_diseases
        
        # IMPROVED weights based on your plateau analysis
        self.classification_weight = 1.0
        self.uncertainty_weight = 0.1 #0.05       
        self.calibration_weight = 0.1 #0.05      
        self.consistency_weight = 0.02     
        self.kl_weight = 1e-3              # Slightly increased from 1e-4
        
        # Focal loss parameters for hard examples
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0

    def _compute_ece_loss(self, probs, labels, n_bins=10):
        """Compute Expected Calibration Error as a differentiable loss"""
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
        ece = torch.tensor(0.0, device=probs.device)
        
        for i in range(n_bins):
            mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
            if i == n_bins - 1:
                mask = (probs >= bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
            
            if mask.sum() > 10:  # Only compute if sufficient samples
                bin_accuracy = (probs[mask].round() == labels[mask]).float().mean()
                bin_confidence = probs[mask].mean()
                bin_weight = mask.sum().float() / probs.numel()
                ece += bin_weight * torch.abs(bin_accuracy - bin_confidence)
        
        return ece
        
    def forward(self, outputs, targets, epoch=0):
        losses = {}
        device = outputs['disease_logits'].device
        
        class_logits = outputs['disease_logits']
        disease_labels = targets['diseases'].float()
        
        # Progressive weight scheduling (ramps up over first 20 epochs)
        epoch_factor = min(1.0, epoch / 20)
        
        # 1. CLASSIFICATION LOSS - Always active
        probs = torch.sigmoid(class_logits)
        
        # Base BCE loss
        base_ce_loss = F.binary_cross_entropy_with_logits(
            class_logits, disease_labels, reduction='mean'
        )
        
        # Add focal loss after initial training for hard examples
        if epoch > 10:
            ce_loss_unreduced = F.binary_cross_entropy_with_logits(
                class_logits, disease_labels, reduction='none'
            )
            p_t = probs * disease_labels + (1 - probs) * (1 - disease_labels)
            focal_weight = self.focal_alpha * (1 - p_t) ** self.focal_gamma
            focal_loss = (focal_weight * ce_loss_unreduced).mean()
            
            # Blend focal with BCE
            classification_loss = 0.7 * base_ce_loss + 0.3 * focal_loss
        else:
            classification_loss = base_ce_loss
        
        # Label smoothing with ramping
        if epoch > 5:
            smoothing_factor = min(0.1, epoch * 0.01)  # Gradually increase smoothing
            smooth_labels = disease_labels * (1 - smoothing_factor) + smoothing_factor / 2
            smooth_loss = F.binary_cross_entropy_with_logits(class_logits, smooth_labels)
            classification_loss = 0.9 * classification_loss + 0.1 * smooth_loss
        
        losses['classification'] = classification_loss * self.classification_weight
        
        # 2. UNCERTAINTY LOSS - Active early but ramped up
        if 'class_uncertainties' in outputs:
            uncertainty = outputs['class_uncertainties']['total_uncertainty']
            epistemic = outputs['class_uncertainties']['epistemic_uncertainty']
            aleatoric = outputs['class_uncertainties']['aleatoric_uncertainty']
            
            # Compute prediction errors for correlation
            with torch.no_grad():
                pred_errors = torch.abs(probs - disease_labels)
            
            # Uncertainty should correlate with errors
            uncertainty_corr_loss = -torch.mean(uncertainty * pred_errors)
            
            # Also add confidence-based targets
            confidence = torch.abs(probs - 0.5) * 2
            target_uncertainty = 0.2 * (1 - confidence) + 0.1
            uncertainty_mse = F.mse_loss(uncertainty, target_uncertainty.detach())
            
            # Balance epistemic vs aleatoric (epistemic should be higher early in training)
            epistemic_weight = max(0.3, 0.7 - epoch * 0.01)  # Decreases over time
            aleatoric_weight = 1 - epistemic_weight
            
            epistemic_target = target_uncertainty * epistemic_weight
            aleatoric_target = target_uncertainty * aleatoric_weight
            balance_loss = (F.mse_loss(epistemic, epistemic_target.detach()) + 
                        F.mse_loss(aleatoric, aleatoric_target.detach()))
            
            #Combine with Emphasis on correlation
            total_uncertainty_loss = (
                0.5 * uncertainty_corr_loss +  # Main objective - maximize correlation
                0.3 * uncertainty_mse +         # Secondary - reasonable magnitudes
                0.2 * balance_loss              # Tertiary - balance components
            )
            
            # Ramp up uncertainty weight
            uncertainty_weight = self.uncertainty_weight * epoch_factor
            losses['uncertainty'] = total_uncertainty_loss * uncertainty_weight
        else:
            losses['uncertainty'] = torch.tensor(0.0, device=device)
        
        # 3. CALIBRATION LOSS - Active early with ramping
        if epoch > 5:  # Start much earlier
            # ECE loss
            calibration_loss = self._compute_ece_loss(probs, disease_labels)
            
            # Brier score
            brier_loss = torch.mean((probs - disease_labels) ** 2)
            
            # Confidence penalty - penalize overconfident wrong predictions
            confidence_penalty = torch.mean(
                torch.where(
                    (probs > 0.8) & (disease_labels < 0.5),
                    probs - 0.8,
                    torch.zeros_like(probs)
                ) + torch.where(
                    (probs < 0.2) & (disease_labels > 0.5),
                    0.2 - probs,
                    torch.zeros_like(probs)
                )
            )
            
            total_calibration_loss = (calibration_loss + 
                                    0.2 * brier_loss + 
                                    0.1 * confidence_penalty)
            
            # Ramp up calibration weight
            calibration_weight = self.calibration_weight * min(1.0, (epoch - 5) / 15)
            losses['calibration'] = total_calibration_loss * calibration_weight
        else:
            losses['calibration'] = torch.tensor(0.0, device=device)
        
        # 4. CONSISTENCY LOSS - Active with ramping
        if 'consistency_score' in outputs:
            # Dynamic target based on epoch (start lower, increase over time)
            consistency_target = min(0.9, 0.5 + epoch * 0.005)
            consistency_target = torch.ones_like(outputs['consistency_score']) * consistency_target
            
            consistency_loss = F.mse_loss(outputs['consistency_score'], consistency_target.detach())
            
            # Ramp up consistency weight
            consistency_weight = self.consistency_weight * epoch_factor
            losses['consistency'] = consistency_loss * consistency_weight
        else:
            losses['consistency'] = torch.tensor(0.0, device=device)
        
        # 5. KL DIVERGENCE - ALWAYS ACTIVE (this is crucial!)
        if 'kl_divergences' in outputs:
            if isinstance(outputs['kl_divergences'], list):
                total_kl = sum(outputs['kl_divergences'])
            else:
                total_kl = outputs['kl_divergences']
            
            # Scale KL by number of samples to normalize
            batch_size = class_logits.size(0)
            normalized_kl = total_kl / batch_size
            
            # Use annealing to gradually increase KL weight
            kl_annealing = min(1.0, epoch / 30)  # Full weight by epoch 30
            
            # MUCH higher weight than before (was 1e-8, now 1e-4 * annealing)
            losses['kl_divergence'] = normalized_kl * (self.kl_weight * kl_annealing)
        else:
            losses['kl_divergence'] = torch.tensor(0.0, device=device)
        
        # 6. Add diversity loss to prevent mode collapse
        if epoch > 10 and 'disease_logits' in outputs:
            # Encourage diverse predictions across the batch
            pred_mean = torch.mean(probs, dim=0)
            pred_std = torch.std(probs, dim=0)
            
            # We want reasonable variance in predictions (not all same)
            diversity_loss = -torch.mean(pred_std)  # Maximize std
            
            # But also want balanced predictions (not all positive or negative)
            balance_loss = torch.mean((pred_mean - 0.5) ** 2)
            
            losses['diversity'] = (diversity_loss * 0.01 + balance_loss * 0.01) * epoch_factor
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return total_loss, {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}