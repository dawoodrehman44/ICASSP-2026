def validate_epoch(model, valid_loader, device, epoch):
    """Enhanced validation with comprehensive metrics"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_epistemic = []
    all_aleatoric = []
    all_consistency = []
    all_total_uncertainty = []
    
    metrics_calculator = AdvancedMetricsCalculator()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(valid_loader):
            if batch is None or any(x is None for x in batch):
                print(f"Skipping invalid validation batch {batch_idx}")
                continue
            
            try:
                images, labels, reports = batch
                images = images.to(device)
                labels = labels.to(device)
                
                # Get predictions with uncertainty
                outputs = model({'images': images}, device, mc_dropout=True, n_mc=50)
                
                # Store results
                preds = torch.sigmoid(outputs['disease_logits']).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())
                
                if 'epistemic_uncertainty' in outputs:
                    all_epistemic.append(outputs['epistemic_uncertainty'].cpu().numpy())
                
                # Get detailed outputs
                detailed_outputs = model({'images': images}, device, return_uncertainty_decomposition=True)
                
                if 'class_uncertainties' in detailed_outputs:
                    all_aleatoric.append(detailed_outputs['class_uncertainties']['aleatoric_uncertainty'].cpu().numpy())
                    all_total_uncertainty.append(detailed_outputs['class_uncertainties']['total_uncertainty'].cpu().numpy())
                
                if 'consistency_score' in detailed_outputs:
                    all_consistency.append(detailed_outputs['consistency_score'].cpu().numpy())
                    
            except Exception as e:
                print(f"Validation error in batch {batch_idx}: {e}")
                continue
    
    if not all_preds:
        return None
    
    # Concatenate results
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # Basic metrics
    metrics = {}
    
    # ROC-AUC per class and macro average
    auc_scores = []
    for i in range(all_preds.shape[1]):
        if len(np.unique(all_labels[:, i])) > 1:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            auc_scores.append(auc)
    
    metrics['roc_auc_macro'] = np.mean(auc_scores) if auc_scores else 0.5
    metrics['roc_auc_per_class'] = auc_scores
    
    # Average Precision
    ap_scores = []
    for i in range(all_preds.shape[1]):
        if len(np.unique(all_labels[:, i])) > 1:
            ap = average_precision_score(all_labels[:, i], all_preds[:, i])
            ap_scores.append(ap)
    
    metrics['average_precision'] = np.mean(ap_scores) if ap_scores else 0.0
    
    # Calibration metrics
    calibration_metrics = metrics_calculator.compute_calibration_metrics(all_preds, all_labels)
    metrics.update(calibration_metrics)
    
    # Uncertainty metrics if available
    if all_epistemic and all_aleatoric and all_total_uncertainty:
        all_epistemic = np.concatenate(all_epistemic)
        all_aleatoric = np.concatenate(all_aleatoric)
        all_total_uncertainty = np.concatenate(all_total_uncertainty)
        
        # Store raw uncertainty values for tracking
        metrics['epistemic_uncertainty_mean'] = np.mean(all_epistemic)
        metrics['epistemic_uncertainty_std'] = np.std(all_epistemic)
        metrics['aleatoric_uncertainty_mean'] = np.mean(all_aleatoric)
        metrics['aleatoric_uncertainty_std'] = np.std(all_aleatoric)
        metrics['total_uncertainty_mean'] = np.mean(all_total_uncertainty)
        
        # Compute uncertainty-error correlation
        errors = np.abs(all_preds - all_labels)
        correlations = []
        for i in range(all_preds.shape[1]):
            if np.std(all_total_uncertainty[:, i]) > 0 and np.std(errors[:, i]) > 0:
                corr = np.corrcoef(all_total_uncertainty[:, i], errors[:, i])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        
        metrics['uncertainty_error_correlation'] = np.mean(correlations) if correlations else 0.0
        
        # Epistemic/Aleatoric ratio
        eps_ratio = all_epistemic / (all_total_uncertainty + 1e-8)
        metrics['epistemic_ratio_mean'] = np.mean(eps_ratio)
        metrics['epistemic_ratio_std'] = np.std(eps_ratio)
    
    # Consistency metrics
    if all_consistency:
        all_consistency = np.concatenate(all_consistency)
        metrics['mean_consistency'] = np.mean(all_consistency)
        metrics['std_consistency'] = np.std(all_consistency)
    
    return metrics

