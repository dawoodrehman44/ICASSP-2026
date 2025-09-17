class EnhancedMultiAgentBayesianModel(nn.Module):
    """Enhanced Multi-Agent Model focused on Disease Prediction and Consistency"""
    def __init__(self, base_encoder, num_classes=14, hidden_dim=512, dropout_rate=0.3):
        super().__init__()
        self.encoder = base_encoder
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Enhanced Bayesian framework
        self.bayesian_framework = EnhancedBayesianFramework(
            input_dim=hidden_dim,
            num_diseases=num_classes
        )
        
        # Loss function
        self.loss_function = EnhancedMultiObjectiveLoss(num_classes)
        
        # Feature projection with attention
        self.feature_projection = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Additional feature extraction layers
        self.feature_enhancer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, batch, device, mc_dropout=False, n_mc=10, return_uncertainty_decomposition=False):
        # Extract features using base encoder
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    
        images = batch["images"]  # Extract the images tensor
        encoder_output = self.encoder(batch, device)
        
        # Get base features
        base_features = encoder_output["cls_pred"]  # [batch_size, num_classes]
        
        # Project and enhance features
        projected_features = self.feature_projection(base_features)
        enhanced_features = self.feature_enhancer(projected_features)
        
        if mc_dropout:
            # Monte Carlo Dropout for uncertainty estimation
            self.train()
            mc_outputs = []
            
            for _ in range(n_mc):
                output = self.bayesian_framework(enhanced_features, return_all_outputs=False)
                mc_outputs.append(torch.sigmoid(output['disease_logits']).detach())
            
            mc_preds = torch.stack(mc_outputs, dim=0)
            
            # Calculate different uncertainty metrics
            mean_pred = mc_preds.mean(dim=0)
            epistemic_uncertainty = mc_preds.var(dim=0)
            
            # Predictive entropy
            predictive_entropy = -torch.sum(mean_pred * torch.log(mean_pred + 1e-8), dim=-1)
            
            # Mutual information
            expected_entropy = -torch.mean(
                torch.sum(mc_preds * torch.log(mc_preds + 1e-8), dim=-1),
                dim=0
            )
            mutual_info = predictive_entropy - expected_entropy
            
            self.eval()
            
            output = {
                'disease_logits': torch.logit(mean_pred + 1e-8),
                'epistemic_uncertainty': epistemic_uncertainty,
                'predictive_entropy': predictive_entropy,
                'mutual_information': mutual_info,
                'mc_samples': mc_preds
            }
            
            return output
            
        else:
            # Standard Bayesian forward pass
            outputs = self.bayesian_framework(enhanced_features, return_all_outputs=return_uncertainty_decomposition)
            
            if return_uncertainty_decomposition:
                # Add uncertainty decomposition analysis
                self._add_uncertainty_analysis(outputs)
            
            return outputs
    
    def _add_uncertainty_analysis(self, outputs):
        """Add detailed uncertainty analysis to outputs"""
        epistemic = outputs['class_uncertainties']['epistemic_uncertainty']
        aleatoric = outputs['class_uncertainties']['aleatoric_uncertainty']
        
        # Uncertainty ratios
        total_unc = epistemic + aleatoric + 1e-8
        outputs['uncertainty_ratios'] = {
            'epistemic_ratio': epistemic / total_unc,
            'aleatoric_ratio': aleatoric / total_unc
        }
        
        # Uncertainty statistics
        outputs['uncertainty_stats'] = {
            'epistemic_mean': epistemic.mean(dim=-1),
            'epistemic_std': epistemic.std(dim=-1),
            'aleatoric_mean': aleatoric.mean(dim=-1),
            'aleatoric_std': aleatoric.std(dim=-1),
            'total_mean': total_unc.mean(dim=-1),
            'total_std': total_unc.std(dim=-1)
        }
    
    # def compute_loss(self, outputs, disease_labels, epoch=0):
    #     targets = {'diseases': disease_labels}
    #     return self.loss_function(outputs, targets, epoch=epoch)

    def compute_loss(self, outputs, disease_labels, epoch=0):
        device = disease_labels.device  # Get correct device
        targets = {'diseases': disease_labels}

        # 🔍 Move loss function to device if not already
        self.loss_function = self.loss_function.to(device)

        # 🔍 Also move outputs to device (if needed)
        outputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in outputs.items()}

        # 🔍 Move targets to device (defensive programming)
        targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}

        return self.loss_function(outputs, targets, epoch=epoch)