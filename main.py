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

### Main Training

if __name__ == "__main__":
    # Configuration
    model_config = {
        "load_backbone_weights": "checkpoints/cxrclip_mc/r50_mc.pt",
        "freeze_backbone_weights": False,  # Allow fine-tuning
        "projection_dim": 512,  # ResNet50 uses 512-dim features
        "image_encoder": {
            "name": "resnet",
            "resnet_type": "resnet50",
            "pretrained": True,
            "source": "cxr_clip"
        },
        "classifier": {
            "config": {
                "name": "linear",
                "n_class": 14
            }
        }
    }

    # Training configuration
    resume_epoch = 0
    epochs = 200
    batch_size = 32
    learning_rate = 1e-3
    weight_decay = 1e-5
    gradient_accumulation_steps = 4  # or even 8
    effective_batch_size = batch_size * gradient_accumulation_steps
    checkpoint_dir = ""
    
    # Data paths
    train_csv = ""
    valid_csv = ""
    image_root = ""
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # ================================
    # DATA LOADING
    # ================================
    
    def categorize_age(age):
        if age <= 30:
            return 'AGE_GROUP_AGE_0_30'
        elif age <= 50:
            return 'AGE_GROUP_AGE_31_50'
        elif age <= 70:
            return 'AGE_GROUP_AGE_51_70'
        else:
            return 'AGE_GROUP_AGE_71_plus'
    
    # Load data
    train_df = pd.read_csv(train_csv)
    valid_df = pd.read_csv(valid_csv)
    
    # Process age groups
    for df in [train_df, valid_df]:
        df['AGE_GROUP'] = df['Age'].apply(categorize_age)
        df = pd.get_dummies(df, columns=['AGE_GROUP'])
    
    age_group_cols = ['AGE_GROUP_AGE_0_30', 'AGE_GROUP_AGE_31_50', 
                      'AGE_GROUP_AGE_51_70', 'AGE_GROUP_AGE_71_plus']
    for col in age_group_cols:
        for df in [train_df, valid_df]:
            if col not in df.columns:
                df[col] = 0
    
    # Data augmentation and normalization
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),  # Increased
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),  # Increased
        transforms.ColorJitter(brightness=0.3, contrast=0.3),  # Increased
        transforms.RandomResizedCrop(320, scale=(0.8, 1.0)),  # Added
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    valid_transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    class CheXpertDataset(Dataset):
        def __init__(self, dataframe, transform=None, image_root=None):
            self.dataframe = dataframe
            self.transform = transform
            self.image_root = image_root
            self.label_cols = [
                'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 
                'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 
                'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 
                'Pleural Other', 'Fracture', 'Support Devices'
            ]
            
            # Pre-validate dataset to identify problematic indices
            self.valid_indices = []
            self._validate_dataset()
        
        def _validate_dataset(self):
            """Pre-validate all images to identify valid indices"""
            print("Validating dataset images...")
            for idx in range(len(self.dataframe)):
                item = self.dataframe.iloc[idx]
                img_path = os.path.join(self.image_root, 
                                    item['Path'].replace("CheXpert-v1.0/train/", ""))
                
                if os.path.exists(img_path):
                    try:
                        # Quick validation - just try to open without loading
                        with Image.open(img_path) as img:
                            img.verify()  # Verify it's a valid image
                        self.valid_indices.append(idx)
                    except (FileNotFoundError, IOError, UnidentifiedImageError) as e:
                        logger.warning(f"Invalid image at index {idx}: {img_path} - {e}")
                else:
                    logger.warning(f"Image not found at index {idx}: {img_path}")
            
            print(f"Dataset validation complete: {len(self.valid_indices)}/{len(self.dataframe)} valid images")
            
            if len(self.valid_indices) == 0:
                raise ValueError("No valid images found in dataset!")
        
        def __len__(self):
            return len(self.valid_indices)
        
        def __getitem__(self, idx):
            # Use only valid indices
            actual_idx = self.valid_indices[idx]
            item = self.dataframe.iloc[actual_idx]  # ✅ FIXED: iloc for position-based indexing

            img_path = os.path.join(
                self.image_root, item['Path'].replace("CheXpert-v1.0/train/", "")
            )
            
            try:
                # Load image
                image = Image.open(img_path).convert("RGB")
                
                # Apply transforms
                if self.transform:
                    image = self.transform(image)
                
                # Get labels
                label = item[self.label_cols].values.astype(np.float32)
                
                # Convert to tensors if needed
                if not isinstance(image, torch.Tensor):
                    image = torch.as_tensor(image)
                
                label = torch.as_tensor(label)
                
                # Return dummy text (empty) since report generation removed
                dummy_text = torch.zeros(1)
                
                return image, label, dummy_text
                
            except Exception as e:
                print(f"Unexpected error loading image at index {actual_idx}: {img_path} - {e}")
                return self._get_fallback_item()
        
        def _get_fallback_item(self):
            """Create a fallback item when image loading fails unexpectedly"""
            # Create a black image as fallback
            if hasattr(self.transform, 'transforms'):
                # Try to infer expected image size from transforms
                for t in self.transform.transforms:
                    if hasattr(t, 'size'):
                        if isinstance(t.size, (list, tuple)):
                            height, width = t.size
                        else:
                            height = width = t.size
                        break
                else:
                    height, width = 224, 224  # Default size
            else:
                height, width = 224, 224
            
            # Create fallback image
            fallback_image = torch.zeros(3, height, width)  # RGB image
            fallback_label = torch.zeros(len(self.label_cols))  # All negative labels
            dummy_text = torch.zeros(1)
            
            logger.warning("Using fallback item due to image loading failure")
            return fallback_image, fallback_label, dummy_text


    def collate_fn(batch):
        valid_batch = [
            item for item in batch
            if item is not None and item[0] is not None and item[1] is not None
        ]
        
        if len(valid_batch) == 0:
            batch_size = len(batch)
            fallback_images = torch.zeros(batch_size, 3, 224, 224)
            fallback_labels = torch.zeros(batch_size, 14)
            return fallback_images, fallback_labels, [None] * batch_size

        try:
            images, labels, texts = zip(*valid_batch)
            images = torch.stack([torch.as_tensor(img) for img in images])
            labels = torch.stack([torch.as_tensor(lbl) for lbl in labels])
            texts = list(texts)
            return images, labels, texts
        except Exception:
            batch_size = len(valid_batch)
            fallback_images = torch.zeros(batch_size, 3, 224, 224)
            fallback_labels = torch.zeros(batch_size, 14)
            return fallback_images, fallback_labels, [None] * batch_size



    # Alternative simpler collate function if you prefer
    def simple_collate_fn(batch):
        """Simpler version that just filters None and uses default collation"""
        # Remove None items
        batch = [item for item in batch if item is not None]
        
        if len(batch) == 0:
            return None  # This will be caught in your training loop
        
        # Use default collation for the rest
        from torch.utils.data.dataloader import default_collate
        return default_collate(batch)

    # Create Dataset objects
    train_dataset = CheXpertDataset(train_df, transform=train_transform, image_root=image_root)
    valid_dataset = CheXpertDataset(valid_df, transform=valid_transform, image_root=image_root)

    # Use the custom collate function
    train_loader = DataLoader(
        train_dataset,                   # ✅ Use dataset, not dataframe
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        drop_last=True
    )

    valid_loader = DataLoader(
        valid_dataset,                   # ✅ Use dataset, not dataframe
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # ================================
    # MODEL INITIALIZATION
    # ================================
    
    print("Initializing Enhanced Bayesian Framework...")
    

    from image_classification import CXRClassification

    model_config["load_backbone_weights"] = "/home/dawood/lab2_rotaion/cxr-clip/cxrclip/model/r50_mc.pt"
    # Initialize base CXR-CLIP model
    base_model = CXRClassification(model_config=model_config, model_type="resnet")
    
    # Initialize enhanced model
    model = EnhancedMultiAgentBayesianModel(
        base_encoder=base_model,
        num_classes=model_config["classifier"]["config"]["n_class"],
        hidden_dim=model_config["projection_dim"] * 2,
    ).to(device)

    # Also ensure model is in correct mode
    model = model.to(device)

    # Force all submodules to device
    for module in model.modules():
        module.to(device)

    # Ensure loss function parameters are on device
    if hasattr(model, 'loss_function'):
        model.loss_function = model.loss_function.to(device)

    # Ensure all buffers and parameters are on the correct device
    def ensure_device(module):
        """Ensure all parameters and buffers are on the correct device"""
        for param in module.parameters():
            param.data = param.data.to(device)
        for buffer in module.buffers():
            buffer.data = buffer.data.to(device)

    # Apply to model
    model.apply(ensure_device)

    # Define weight initialization function
    def init_weights(module):
        """Initialize weights for better training stability"""
        if isinstance(module, VariationalLinear):
            # Initialize with smaller variance for stability
            nn.init.xavier_normal_(module.weight_mean, gain=0.5)
            nn.init.constant_(module.weight_logvar, -5.0)  # Start with very low variance
            nn.init.zeros_(module.bias_mean)
            nn.init.constant_(module.bias_logvar, -5.0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)
    
    # Apply the initialization to the model
    print("Applying custom weight initialization...")
    model.apply(init_weights)
    
    # Also ensure model is in correct mode
    model = model.to(device)
    model.train()


    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} total parameters")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': 1e-6, 'weight_decay': 1e-4},
        {'params': model.bayesian_framework.parameters(), 'lr': 5e-5, 'weight_decay': 1e-5},
        {'params': model.feature_projection.parameters(), 'lr': 5e-5, 'weight_decay': 1e-5},
        {'params': model.feature_enhancer.parameters(), 'lr': 1e-4, 'weight_decay': 1e-5}
    ])


    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=10, 
        min_lr=1e-6,
        verbose=True
    )

    # scheduler = CyclicWarmupScheduler(optimizer, base_lr=1e-6, max_lr=1e-4, 
    #                                 step_size_up=5, step_size_down=15)

    # ================================
    # RESUME FROM CHECKPOINT
    # ================================
    
    best_metrics = {'roc_auc': 0.0, 'ece': 1.0, 'consistency': 0.0}
    
    if resume_epoch > 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{resume_epoch - 1}.pt")
        if os.path.exists(checkpoint_path):
            print(f"Resuming training from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            best_metrics = checkpoint.get("best_metrics", best_metrics)
        else:
            print(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")

# ================================
# TRAINING LOOP
# ================================

print("\n" + "="*80)
print("Starting Enhanced Bayesian Framework Training")
print("="*80)
print(f"Training for {epochs} epochs")
print(f"Batch size: {batch_size}")
print(f"Learning rate: {learning_rate}")
print(f"Device: {device}")
print("="*80 + "\n")

# Training history
training_history = {
    'train_loss': [],
    'val_metrics': [],
    'learning_rates': []
}


for epoch in range(resume_epoch, epochs):
    print(f"\n{'='*80}")
    print(f"Epoch {epoch+1}/{epochs}")
    print(f"{'='*80}")
        
    # Training phase
    train_loss, train_loss_components = train_epoch(
        model, train_loader, optimizer, device, epoch, gradient_accumulation_steps
    )
       
    print(f"\nTraining Results:")
    print(f"Total Loss: {train_loss:.4f}")
    print("Loss Components:")
    for component, value in train_loss_components.items():
        if component != 'total':
            print(f"  {component}: {value:.4f}")
    
    # Validation phase
    val_metrics = validate_epoch(model, valid_loader, device, epoch)

    if val_metrics:
        print(f"\nValidation Results:")
        print(f"ROC-AUC (Macro): {val_metrics['roc_auc_macro']:.4f}")
        print(f"Average Precision: {val_metrics['average_precision']:.4f}")
        print(f"ECE: {val_metrics['ece']:.4f}")
        print(f"MCE: {val_metrics['mce']:.4f}")
        
        if 'uncertainty_error_correlation' in val_metrics:
            print(f"\nUncertainty Metrics:")
            print(f"  Uncertainty-Error Correlation: {val_metrics['uncertainty_error_correlation']:.4f}")
            
            # Only print AUUPC if it exists
            if 'auupc' in val_metrics:
                print(f"  AUUPC: {val_metrics['auupc']:.4f}")
            
            if 'epistemic_ratio_mean' in val_metrics:
                print(f"  Epistemic Ratio: {val_metrics['epistemic_ratio_mean']:.3f} ± {val_metrics['epistemic_ratio_std']:.3f}")
            
            # Print the new metrics we're tracking
            if 'epistemic_uncertainty_mean' in val_metrics:
                print(f"  Epistemic Uncertainty: {val_metrics['epistemic_uncertainty_mean']:.4f} ± {val_metrics['epistemic_uncertainty_std']:.4f}")
            
            if 'aleatoric_uncertainty_mean' in val_metrics:
                print(f"  Aleatoric Uncertainty: {val_metrics['aleatoric_uncertainty_mean']:.4f} ± {val_metrics['aleatoric_uncertainty_std']:.4f}")
        
        if 'mean_consistency' in val_metrics:
            print(f"\nConsistency Metrics:")
            print(f"  Mean Consistency: {val_metrics['mean_consistency']:.4f}")
            print(f"  Std Consistency: {val_metrics['std_consistency']:.4f}")
        
        # Update best metrics
        is_best = False
        if val_metrics['roc_auc_macro'] > best_metrics['roc_auc']:
            best_metrics['roc_auc'] = val_metrics['roc_auc_macro']
            is_best = True
            print(f"  → New best ROC-AUC!")
        
        if val_metrics['ece'] < best_metrics['ece']:
            best_metrics['ece'] = val_metrics['ece']
            is_best = True
            print(f"  → New best ECE!")
        
        if 'mean_consistency' in val_metrics and val_metrics['mean_consistency'] > best_metrics.get('consistency', 0):
            best_metrics['consistency'] = val_metrics['mean_consistency']
            is_best = True
            print(f"  → New best Consistency!")
    
    # Update learning rate
    scheduler.step(val_metrics['roc_auc_macro'])
    current_lr = optimizer.param_groups[0]['lr']
    print(f"\nLearning rate: {current_lr:.6f}")
    
    # Save checkpoint
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": train_loss,
        "train_loss_components": train_loss_components,
        "val_metrics": val_metrics,
        "best_metrics": best_metrics
    }
    
    # Save regular checkpoint
    torch.save(checkpoint, os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt"))
    
    # Save best model
    if is_best and val_metrics:
        torch.save(checkpoint, os.path.join(checkpoint_dir, "best_model.pt"))
    
    # Update training history
    training_history['train_loss'].append(train_loss)
    training_history['val_metrics'].append(val_metrics)
    training_history['learning_rates'].append(current_lr)
    
    print(f"Checkpoint saved to {checkpoint_dir}")

# ================================
# FINAL EVALUATION AND VISUALIZATION
# ================================

print("\n" + "="*80)
print("Training Complete!")
print("="*80)
print(f"Best ROC-AUC: {best_metrics['roc_auc']:.4f}")
print(f"Best ECE: {best_metrics['ece']:.4f}")
print(f"Best Consistency: {best_metrics.get('consistency', 0):.4f}")