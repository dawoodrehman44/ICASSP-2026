import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Import required modules from your training code
from image_classification import CXRClassification
# You'll need to import your model classes - adjust path as needed
# from your_model_file import EnhancedMultiAgentBayesianModel, VariationalLinear, etc.

def categorize_diseases_into_groups():
    """
    Categorize 14 CheXpert diseases into clinical groups
    """
    disease_groups = {
        'Cardio-thoracic': ['Cardiomegaly', 'Enlarged Cardiomediastinum'],
        'Lung Parenchymal': ['Lung Opacity', 'Lung Lesion', 'Consolidation', 'Pneumonia', 'Atelectasis'],
        'Pleural Disorders': ['Pleural Effusion', 'Pleural Other', 'Pneumothorax'],
        'Support Devices': ['Support Devices'],
        'Rare Pathologies': ['Fracture', 'Edema'],
        'No Finding': ['No Finding']
    }
    
    # Create mapping from disease to group
    disease_to_group = {}
    all_diseases = [
        'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 
        'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 
        'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 
        'Pleural Other', 'Fracture', 'Support Devices'
    ]
    
    for group, diseases in disease_groups.items():
        for disease in diseases:
            if disease in all_diseases:
                disease_to_group[all_diseases.index(disease)] = group
    
    return disease_groups, disease_to_group, all_diseases

class CheXpertTestDataset(Dataset):
    """Dataset for testing with CheXpert validation data"""
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
        self.valid_indices = self._validate_dataset()
    
    def _validate_dataset(self):
        """Validate dataset and return valid indices"""
        valid_indices = []
        for idx in range(len(self.dataframe)):
            item = self.dataframe.iloc[idx]
            img_path = os.path.join(self.image_root, 
                                  item['Path'].replace("CheXpert-v1.0/train/", ""))
            if os.path.exists(img_path):
                valid_indices.append(idx)
        return valid_indices
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        item = self.dataframe.iloc[actual_idx]
        
        img_path = os.path.join(
            self.image_root, item['Path'].replace("CheXpert-v1.0/train/", "")
        )
        
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            
            label = item[self.label_cols].values.astype(np.float32)
            label = torch.as_tensor(label)
            
            return image, label, img_path
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None

def collate_fn(batch):
    """Custom collate function"""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    
    images, labels, paths = zip(*batch)
    images = torch.stack(images)
    labels = torch.stack(labels)
    
    return images, labels, paths

def load_trained_model(checkpoint_path, device):
    """Load the trained model from checkpoint"""
    print(f"Loading model from {checkpoint_path}")
    
    # Model configuration
    model_config = {
        "load_backbone_weights": "/home/dawood/lab2_rotaion/cxr-clip/cxrclip/model/r50_mc.pt",
        "freeze_backbone_weights": False,
        "projection_dim": 512,
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
    
    # Initialize base model
    base_model = CXRClassification(model_config=model_config, model_type="resnet")
    
    # Initialize enhanced model
    model = EnhancedMultiAgentBayesianModel(
        base_encoder=base_model,
        num_classes=14,
        hidden_dim=512 * 2,
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print("Model loaded successfully!")
    return model

def perform_monte_carlo_inference(model, images, device, n_mc_samples=100):
    """
    Perform Monte Carlo dropout inference for uncertainty estimation
    """
    model.train()  # Enable dropout
    
    mc_predictions = []
    mc_epistemic = []
    mc_aleatoric = []
    
    with torch.no_grad():
        for _ in range(n_mc_samples):
            # Get predictions with uncertainty
            outputs = model({'images': images}, device, return_uncertainty_decomposition=True)
            
            # Store predictions
            probs = torch.sigmoid(outputs['disease_logits'])
            mc_predictions.append(probs)
            
            # Store uncertainties
            if 'class_uncertainties' in outputs:
                mc_epistemic.append(outputs['class_uncertainties']['epistemic_uncertainty'])
                mc_aleatoric.append(outputs['class_uncertainties']['aleatoric_uncertainty'])
    
    # Stack all MC samples
    mc_predictions = torch.stack(mc_predictions)  # [n_mc, batch, n_diseases]
    
    # Calculate mean prediction and predictive uncertainty
    mean_prediction = mc_predictions.mean(dim=0)
    
    # Epistemic uncertainty (uncertainty in predictions)
    epistemic_uncertainty = mc_predictions.var(dim=0)
    
    # Aleatoric uncertainty (average of predicted aleatoric)
    if mc_aleatoric:
        aleatoric_uncertainty = torch.stack(mc_aleatoric).mean(dim=0)
    else:
        # Fallback: estimate from prediction entropy
        entropy = -mean_prediction * torch.log(mean_prediction + 1e-8) - \
                  (1 - mean_prediction) * torch.log(1 - mean_prediction + 1e-8)
        aleatoric_uncertainty = entropy
    
    model.eval()  # Disable dropout
    
    return {
        'mean_prediction': mean_prediction,
        'epistemic_uncertainty': epistemic_uncertainty,
        'aleatoric_uncertainty': aleatoric_uncertainty,
        'total_uncertainty': epistemic_uncertainty + aleatoric_uncertainty,
        'mc_samples': mc_predictions
    }

def compute_group_uncertainties(epistemic_all, aleatoric_all, disease_to_group, disease_groups):
    """
    Compute uncertainties for disease groups
    """
    group_uncertainties = {}
    
    for group_name in disease_groups.keys():
        if group_name == 'No Finding':
            continue  # Skip "No Finding" group for the table
            
        # Find indices for this group
        group_indices = [idx for idx, group in disease_to_group.items() if group == group_name]
        
        if not group_indices:
            continue
        
        # Extract uncertainties for this group
        group_epistemic = epistemic_all[:, group_indices]
        group_aleatoric = aleatoric_all[:, group_indices]
        
        # Compute statistics
        epis_mean = np.mean(group_epistemic)
        alea_mean = np.mean(group_aleatoric)
        
        # Combined uncertainty for std and CI
        combined = group_epistemic + group_aleatoric
        combined_std = np.std(combined)
        
        # 95% CI using bootstrap or normal approximation
        n_samples = combined.size
        ci_95 = 1.96 * combined_std / np.sqrt(n_samples)
        
        group_uncertainties[group_name] = {
            'epistemic': epis_mean,
            'aleatoric': alea_mean,
            'std_dev': combined_std,
            'ci_95': ci_95
        }
    
    return group_uncertainties

def test_and_analyze(model, test_loader, device, n_mc_samples=1000):
    """
    Main testing function with uncertainty analysis
    """
    print(f"Starting evaluation with {n_mc_samples} MC samples...")
    
    all_epistemic = []
    all_aleatoric = []
    all_predictions = []
    all_labels = []
    
    # Get disease groupings
    disease_groups, disease_to_group, all_diseases = categorize_diseases_into_groups()
    
    # Process batches
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Processing batches")):
        if batch is None:
            continue
            
        images, labels, paths = batch
        images = images.to(device)
        labels = labels.to(device)
        
        # Perform MC inference
        mc_results = perform_monte_carlo_inference(model, images, device, n_mc_samples)
        
        # Store results
        all_epistemic.append(mc_results['epistemic_uncertainty'].cpu().numpy())
        all_aleatoric.append(mc_results['aleatoric_uncertainty'].cpu().numpy())
        all_predictions.append(mc_results['mean_prediction'].cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    
    # Concatenate all results
    all_epistemic = np.concatenate(all_epistemic, axis=0)
    all_aleatoric = np.concatenate(all_aleatoric, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    print(f"Processed {len(all_predictions)} samples")
    
    # Compute group uncertainties
    group_uncertainties = compute_group_uncertainties(
        all_epistemic, all_aleatoric, disease_to_group, disease_groups
    )
    
    # Create formatted table
    create_uncertainty_table(group_uncertainties)
    
    # Additional analysis
    compute_additional_metrics(all_epistemic, all_aleatoric, all_predictions, 
                             all_labels, all_diseases)
    
    return group_uncertainties, all_epistemic, all_aleatoric

def create_uncertainty_table(group_uncertainties):
    """
    Create and display the uncertainty analysis table
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE UNCERTAINTY ANALYSIS")
    print("="*70)
    
    # Header
    print(f"{'Disease Group':<20} {'Epis':<8} {'Alea':<8} {'Std Dev':<10} {'CI 95%':<10}")
    print("-"*70)
    
    # Define order for display
    display_order = ['Cardio-thoracic', 'Lung Parenchymal', 'Pleural Disorders', 
                    'Support Devices', 'Rare Pathologies']
    
    # Table rows
    for group_name in display_order:
        if group_name in group_uncertainties:
            stats = group_uncertainties[group_name]
            print(f"{group_name:<20} {stats['epistemic']:.3f}    {stats['aleatoric']:.3f}    "
                  f"±{stats['std_dev']:.3f}     ±{stats['ci_95']:.3f}")
    
    print("="*70)
    print("Table: Comprehensive uncertainty analysis with epistemic (Epis) and")
    print("aleatoric (Alea) estimates, standard deviation, and 95% confidence intervals.")
    print("="*70)
    
    # Also create a pandas DataFrame for export
    df_data = []
    for group_name in display_order:
        if group_name in group_uncertainties:
            stats = group_uncertainties[group_name]
            df_data.append({
                'Disease Group': group_name,
                'Epistemic': f"{stats['epistemic']:.3f}",
                'Aleatoric': f"{stats['aleatoric']:.3f}",
                'Std Dev': f"±{stats['std_dev']:.3f}",
                'CI 95%': f"±{stats['ci_95']:.3f}"
            })
    
    df = pd.DataFrame(df_data)
    return df

def compute_additional_metrics(epistemic, aleatoric, predictions, labels, disease_names):
    """
    Compute additional metrics for comprehensive analysis
    """
    print("\n" + "="*70)
    print("ADDITIONAL UNCERTAINTY METRICS")
    print("="*70)
    
    # Overall statistics
    print(f"\nOverall Statistics:")
    print(f"  Mean Epistemic Uncertainty: {np.mean(epistemic):.4f}")
    print(f"  Mean Aleatoric Uncertainty: {np.mean(aleatoric):.4f}")
    print(f"  Epistemic/Aleatoric Ratio: {np.mean(epistemic)/np.mean(aleatoric):.3f}")
    
    # Uncertainty-error correlation
    errors = np.abs(predictions - labels)
    total_uncertainty = epistemic + aleatoric
    
    correlations = []
    for i in range(predictions.shape[1]):
        if np.std(total_uncertainty[:, i]) > 0 and np.std(errors[:, i]) > 0:
            corr = np.corrcoef(total_uncertainty[:, i], errors[:, i])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
    
    print(f"\n  Uncertainty-Error Correlation: {np.mean(correlations):.4f}")
    
    # Per-disease analysis
    print(f"\nPer-Disease Uncertainty Analysis:")
    print(f"{'Disease':<25} {'Epistemic':<12} {'Aleatoric':<12} {'Total':<12}")
    print("-"*60)
    
    for i, disease in enumerate(disease_names):
        epis_mean = np.mean(epistemic[:, i])
        alea_mean = np.mean(aleatoric[:, i])
        total_mean = epis_mean + alea_mean
        print(f"{disease:<25} {epis_mean:.4f}      {alea_mean:.4f}      {total_mean:.4f}")

def main():
    """
    Main execution function
    """
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Paths
    model_epoch = 000
    checkpoint_dir = ""
    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{model_epoch}.pt")
    
    valid_csv = ""
    image_root = ""
    
    # Parameters
    batch_size = 16  # Smaller batch size for MC sampling
    n_mc_samples = 100  # Number of Monte Carlo samples
    
    # Data preparation
    print("Loading validation data...")
    valid_df = pd.read_csv(valid_csv)
    
    # Transform for validation
    valid_transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and loader
    test_dataset = CheXpertTestDataset(valid_df, transform=valid_transform, image_root=image_root)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Load model
    model = load_trained_model(checkpoint_path, device)
    
    # Perform testing and analysis
    group_uncertainties, epistemic_all, aleatoric_all = test_and_analyze(
        model, test_loader, device, n_mc_samples
    )
    
    # Save results
    results = {
        'group_uncertainties': group_uncertainties,
        'epistemic_mean': float(np.mean(epistemic_all)),
        'aleatoric_mean': float(np.mean(aleatoric_all)),
        'epistemic_std': float(np.std(epistemic_all)),
        'aleatoric_std': float(np.std(aleatoric_all))
    }
    
    # Save to JSON
    import json
    with open(os.path.join(checkpoint_dir, f'uncertainty_analysis_epoch_{model_epoch}.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save table to CSV
    df = create_uncertainty_table(group_uncertainties)
    df.to_csv(os.path.join(checkpoint_dir, f'uncertainty_table_epoch_{model_epoch}.csv'), index=False)
    
    print(f"\nResults saved to {checkpoint_dir}")
    print("Analysis complete!")

if __name__ == "__main__":
    main()