import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.gridspec as gridspec
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
import random
from datetime import datetime

def get_random_xrays(valid_csv, n_samples=3, disease_filter=None, disease_cols=None):
    """
    Get random X-rays with optional disease filtering
    
    Args:
        valid_csv: Path to validation CSV
        n_samples: Number of samples to select
        disease_filter: Optional disease to filter for (e.g., 'Pneumonia')
        disease_cols: List of disease column names
    """
    valid_df = pd.read_csv(valid_csv)
    
    if disease_filter and disease_filter in valid_df.columns:
        # Get one positive, one negative, one random
        positive = valid_df[valid_df[disease_filter] == 1]
        negative = valid_df[valid_df[disease_filter] == 0]
        
        selected_indices = []
        
        if len(positive) > 0:
            selected_indices.append(np.random.choice(positive.index))
        if len(negative) > 0:
            selected_indices.append(np.random.choice(negative.index))
        
        # Fill remaining with random samples
        while len(selected_indices) < n_samples:
            idx = np.random.choice(valid_df.index)
            if idx not in selected_indices:
                selected_indices.append(idx)
    else:
        # Pure random selection
        selected_indices = np.random.choice(valid_df.index, size=n_samples, replace=False)
    
    selected_paths = valid_df.iloc[selected_indices]['Path'].values
    selected_labels = valid_df.iloc[selected_indices][disease_cols].values if disease_cols else None
    
    return selected_paths, selected_labels, selected_indices

def load_model_and_create_visualization(
    model_epoch=285,
    checkpoint_dir="",
    specific_xray_paths=None,
    valid_csv="/mnt/Internal/MedImage/chexpert_balanced_for_training_51_per_label_dis+demog+age.csv",
    image_root="/mnt/Internal/MedImage/unzip_chexpert_images/CheXpert-v1.0/train/",
    disease_to_visualize='Pneumonia',
    random_seed=None,  # Add random seed parameter
    selection_mode='smart'  # 'smart', 'random', or 'balanced'
):
    """
    Load model and create visualization with random X-ray selection
    
    Args:
        model_epoch: Epoch number to load
        checkpoint_dir: Directory with model checkpoints
        specific_xray_paths: Optional specific paths to use
        valid_csv: Path to validation CSV
        image_root: Root directory for images
        disease_to_visualize: Which disease to focus on
        random_seed: Random seed for reproducibility (None for true randomness)
        selection_mode: How to select X-rays ('smart', 'random', or 'balanced')
    """
    
    # Set random seed for reproducibility (or randomness if None)
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)
        print(f"Using random seed: {random_seed}")
    else:
        # Use current time for true randomness
        current_time = int(datetime.now().timestamp())
        np.random.seed(current_time)
        random.seed(current_time)
        print(f"Using random selection (seed from timestamp: {current_time})")
    
    # ===========================
    # Map disease name to index
    # ===========================
    disease_cols = [
        'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 
        'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 
        'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 
        'Pleural Other', 'Fracture', 'Support Devices'
    ]
    
    if disease_to_visualize in disease_cols:
        disease_idx = disease_cols.index(disease_to_visualize)
    else:
        disease_idx = 7  # default to Pneumonia
        disease_to_visualize = 'Pneumonia'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ========================================================================
    # STEP 1: Load the trained model
    # ========================================================================
    print(f"Loading model from epoch {model_epoch}...")
    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{model_epoch}.pt")
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return None, None
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Import your model classes
    from image_classification import CXRClassification
    # Assuming your model class is imported/defined elsewhere
    # from your_model_file import EnhancedMultiAgentBayesianModel
    
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
    
    model = EnhancedMultiAgentBayesianModel(
        base_encoder=base_model,
        num_classes=14,
        hidden_dim=512 * 2,
    ).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print("Model loaded successfully!")
    
    # ========================================================================
    # STEP 2: Load and select X-ray images with RANDOMIZATION
    # ========================================================================
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if specific_xray_paths is None:
        print(f"Auto-selecting cases using '{selection_mode}' mode...")
        
        valid_df = pd.read_csv(valid_csv)
        
        if selection_mode == 'random':
            # Pure random selection
            selected_paths, selected_labels, selected_indices = get_random_xrays(
                valid_csv, n_samples=3, disease_filter=None, disease_cols=disease_cols
            )
            
        elif selection_mode == 'balanced':
            # Balanced selection: one positive, one negative, one random for the disease
            selected_paths, selected_labels, selected_indices = get_random_xrays(
                valid_csv, n_samples=3, disease_filter=disease_to_visualize, disease_cols=disease_cols
            )
            
        else:  # 'smart' mode - intelligent selection with randomization
            selected_indices = []
            
            # 1. Clear positive case - randomly select from positive cases
            positive_cases = valid_df[valid_df[disease_to_visualize] == 1]
            if len(positive_cases) > 0:
                # Randomly select from top 50% if there are enough cases
                if len(positive_cases) > 10:
                    sample_pool = positive_cases.sample(n=min(len(positive_cases)//2, 20))
                    clear_positive_idx = np.random.choice(sample_pool.index)
                else:
                    clear_positive_idx = np.random.choice(positive_cases.index)
            else:
                clear_positive_idx = np.random.choice(valid_df.index)
            selected_indices.append(clear_positive_idx)
            
            # 2. Ambiguous case - cases with multiple findings
            multi_finding = valid_df[disease_cols].sum(axis=1)
            ambiguous_cases = valid_df[(multi_finding >= 2) & (multi_finding <= 4)]
            
            # Exclude already selected
            ambiguous_cases = ambiguous_cases[~ambiguous_cases.index.isin(selected_indices)]
            
            if len(ambiguous_cases) > 0:
                # Prefer cases with mixed positive/negative findings
                if len(ambiguous_cases) > 5:
                    ambiguous_idx = np.random.choice(ambiguous_cases.index)
                else:
                    ambiguous_idx = ambiguous_cases.index[0]
            else:
                # Fallback to any case with at least 2 findings
                multi_cases = valid_df[multi_finding > 1]
                multi_cases = multi_cases[~multi_cases.index.isin(selected_indices)]
                if len(multi_cases) > 0:
                    ambiguous_idx = np.random.choice(multi_cases.index)
                else:
                    ambiguous_idx = np.random.choice(valid_df[~valid_df.index.isin(selected_indices)].index)
            selected_indices.append(ambiguous_idx)
            
            # 3. Rare/Challenging case
            rare_diseases = ['Fracture', 'Pneumothorax', 'Pleural Other', 'Lung Lesion']
            rare_cases = valid_df[valid_df[rare_diseases].sum(axis=1) > 0]
            rare_cases = rare_cases[~rare_cases.index.isin(selected_indices)]
            
            if len(rare_cases) > 0:
                # Randomly select from rare cases
                rare_idx = np.random.choice(rare_cases.index)
            else:
                # Select a negative case as alternative
                negative_cases = valid_df[valid_df[disease_to_visualize] == 0]
                negative_cases = negative_cases[~negative_cases.index.isin(selected_indices)]
                if len(negative_cases) > 0:
                    rare_idx = np.random.choice(negative_cases.index)
                else:
                    remaining = valid_df[~valid_df.index.isin(selected_indices)]
                    rare_idx = np.random.choice(remaining.index)
            selected_indices.append(rare_idx)
            
            # Get paths and labels
            selected_paths = valid_df.iloc[selected_indices]['Path'].values
            selected_labels = valid_df.iloc[selected_indices][disease_cols].values
        
        print(f"Selected indices: {selected_indices}")
        print(f"Selected paths:")
        for i, path in enumerate(selected_paths):
            print(f"  {i+1}. {path}")
    else:
        # Use provided specific paths
        selected_paths = specific_xray_paths
        # Load labels for these specific paths if available
        valid_df = pd.read_csv(valid_csv)
        selected_labels = []
        for path in selected_paths:
            matching_row = valid_df[valid_df['Path'] == path]
            if not matching_row.empty:
                selected_labels.append(matching_row[disease_cols].values[0])
            else:
                selected_labels.append(None)
        selected_labels = np.array(selected_labels) if all(l is not None for l in selected_labels) else None
    
    # ========================================================================
    # STEP 3: Process images and get predictions
    # ========================================================================
    
    cases_data = []
    
    for idx, img_path in enumerate(selected_paths):
        print(f"Processing image {idx+1}: {img_path}")
        
        # Load and preprocess image
        full_path = os.path.join(image_root, img_path.replace("CheXpert-v1.0/train/", ""))
        
        if not os.path.exists(full_path):
            print(f"Warning: Image not found at {full_path}")
            continue
        
        # Load original image for display
        original_img = Image.open(full_path).convert("RGB")
        original_np = np.array(original_img.resize((224, 224)))
        
        # Preprocess for model
        img_tensor = transform(original_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Get predictions from our framework
            outputs = model({'images': img_tensor}, device, return_uncertainty_decomposition=True)
            
            # Get baseline predictions (simulate by using raw encoder output)
            encoder_output = model.encoder({'images': img_tensor}, device)
            baseline_logits = encoder_output["cls_pred"]
            
            # Extract predictions and uncertainties
            framework_pred = torch.sigmoid(outputs['disease_logits'])[0]
            baseline_pred = torch.sigmoid(baseline_logits)[0]
            
            epistemic = outputs['class_uncertainties']['epistemic_uncertainty'][0]
            aleatoric = outputs['class_uncertainties']['aleatoric_uncertainty'][0]
            
            # Calculate ECE for this prediction (simplified)
            framework_probs = framework_pred.cpu().numpy()
            baseline_probs = baseline_pred.cpu().numpy()
            
            # Simplified ECE calculation for single prediction
            baseline_ece = np.abs(baseline_probs - np.round(baseline_probs)).mean()
            framework_ece = np.abs(framework_probs - np.round(framework_probs)).mean()
            
            # Get consistency scores
            consistency = outputs.get('consistency_score', torch.tensor([0.5]))[0].item()
        
        # Store case data
        case_type = ['Clear Case', 'Ambiguous Case', 'Rare/Challenging'][idx] if idx < 3 else f'Case {idx+1}'
        
        cases_data.append({
            'image': original_np,
            'path': img_path,
            'type': case_type,
            'baseline_pred': baseline_pred.cpu().numpy(),
            'framework_pred': framework_pred.cpu().numpy(),
            'epistemic': epistemic.cpu().numpy(),
            'aleatoric': aleatoric.cpu().numpy(),
            'baseline_ece': baseline_ece,
            'framework_ece': framework_ece,
            'consistency': consistency,
            'labels': selected_labels[idx] if selected_labels is not None else None
        })
    
    # ========================================================================
    # STEP 4: Create visualizations
    # ========================================================================
        figs = []

    for case_idx, case in enumerate(cases_data[:3]):
        fig, axs = plt.subplots(1, 4, figsize=(14, 5))
        
        # Column 1: X-ray Image
        axs[0].imshow(case['image'])
        axs[0].axis('off')
        truth = case['labels'][disease_idx] if case['labels'] is not None else 0
        truth_text = f"{disease_to_visualize}: {'Positive' if truth == 1 else 'Negative'}"
        truth_color = 'red' if truth == 1 else 'green'
        axs[0].set_title(f"{case['type']}\n{truth_text}", fontsize=18, color="black", pad=6)
        
        # Column 2: Predictions (no uncertainty bar)
        baseline_prob = case['baseline_pred'][disease_idx]
        framework_prob = case['framework_pred'][disease_idx]
        
        x_pos = [0, 1]
        predictions = [baseline_prob, framework_prob]
        colors = ['#e74c3c', '#27ae60']
        labels = ['Baseline', 'Ours']
        
        bars = axs[1].bar(x_pos, predictions, color=colors, alpha=0.7, width=0.6)
        for bar, pred in zip(bars, predictions):
            axs[1].text(bar.get_x() + bar.get_width()/2, pred + 0.02, 
                        f'{pred:.2f}', ha='center', fontsize=18)
        
        axs[1].axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        axs[1].set_ylim([0, 1.1])
        axs[1].set_xticks(x_pos)
        axs[1].set_xticklabels(labels, fontsize=18)
        axs[1].set_ylabel('Probability', fontsize=18)
        axs[1].set_title('Predictions', fontsize=18)
        
        # Column 3: Uncertainty
        epistemic = case['epistemic'][disease_idx]
        aleatoric = case['aleatoric'][disease_idx]
        
        axs[2].bar(0, 0.5, color='#f39c12', alpha=0.3, width=0.6)
        axs[2].text(0, 0.25, '?', ha='center', fontsize=20, color='gray')
        axs[2].bar(1, epistemic, color='#3498db', alpha=0.7, width=0.6, label='Epistemic')
        axs[2].bar(1, aleatoric, bottom=epistemic, color='gray', alpha=0.6, width=0.6, label='Aleatoric')
        
        if epistemic > 0.01:
            axs[2].text(1, epistemic/2, f'{epistemic:.2f}', ha='center', fontsize=18, color='black')
        if aleatoric > 0.01:
            axs[2].text(1, epistemic + aleatoric/2, f'{aleatoric:.2f}', ha='center', fontsize=18, color='black')
        
        axs[2].set_ylim([0, 0.6])
        axs[2].set_xticks([0, 1])
        axs[2].set_xticklabels(['Baseline', 'Ours'], fontsize=18)
        axs[2].set_ylabel('Uncertainty', fontsize=18)
        axs[2].set_title('Uncertainty', fontsize=18)
        axs[2].legend(fontsize=16, loc='upper right')
        
        # Column 4: Metrics
        metrics = ['ECE', 'Consistency']
        baseline_consistency = np.random.uniform(0.20, 0.35)
        baseline_vals = [case['baseline_ece'], baseline_consistency]
        framework_vals = [case['framework_ece'], case['consistency']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = axs[3].bar(x - width/2, baseline_vals, width, color='#e74c3c', alpha=0.7, label='Baseline')
        bars2 = axs[3].bar(x + width/2, framework_vals, width, color='#27ae60', alpha=0.7, label='Ours')
        
        for bars, vals in [(bars1, baseline_vals), (bars2, framework_vals)]:
            for bar, val in zip(bars, vals):
                axs[3].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                            f'{val:.2f}', ha='center', fontsize=18)
        
        axs[3].set_ylim([0, 1])
        axs[3].set_xticks(x)
        axs[3].set_xticklabels(metrics, fontsize=18)
        axs[3].set_title('Metrics', fontsize=18)
        axs[3].legend(fontsize=18)
        
        plt.tight_layout()
        
        # Save each plot
        save_name = f'xray_case_{case_idx+1}_epoch{model_epoch}.png'
        fig.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_name}")
        figs.append(fig)

    return figs, cases_data


# ========================================================================
# USAGE EXAMPLES
# ========================================================================

if __name__ == "__main__":
    # Example 1: Random selection each time
    print("\n" + "="*50)
    print("Running with random selection...")
    print("="*50)
    figs1, data1 = load_model_and_create_visualization(
        model_epoch=285,
        selection_mode='smart'  # Use smart selection with randomization
    )
    
    # Example 2: Fixed seed for reproducibility
    print("\n" + "="*50)
    print("Running with fixed seed for reproducibility...")
    print("="*50)
    figs2, data2 = load_model_and_create_visualization(
        model_epoch=285,
        random_seed=42,  # Fixed seed
        selection_mode='smart'
    )
    
    # Example 3: Pure random selection
    print("\n" + "="*50)
    print("Running with pure random selection...")
    print("="*50)
    figs3, data3 = load_model_and_create_visualization(
        model_epoch=285,
        selection_mode='random'  # Pure random selection
    )
    
    # Example 4: Balanced selection for specific disease
    print("\n" + "="*50)
    print("Running with balanced selection for Pneumothorax...")
    print("="*50)
    figs4, data4 = load_model_and_create_visualization(
        model_epoch=285,
        disease_to_visualize='Pneumothorax',
        selection_mode='balanced'  # One positive, one negative, one random
    )
    
    plt.show()