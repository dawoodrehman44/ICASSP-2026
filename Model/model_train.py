def train_epoch(model, train_loader, optimizer, device, epoch, gradient_accumulation_steps=2):
    """
    UPDATED training function with epoch tracking for adaptive behavior
    """
    model.train()
    
    # SET EPOCH for adaptive behavior
    if hasattr(model, 'set_epoch'):
        model.set_epoch(epoch)
    
    # Set epoch for all VariationalLinear layers
    for module in model.modules():
        if isinstance(module, VariationalLinear):
            module._current_epoch = epoch
    
    # Warmup strategy - freeze variance for first few epochs
    if epoch < 5:
        for module in model.modules():
            if isinstance(module, VariationalLinear):
                module.weight_logvar.requires_grad = False
                module.bias_logvar.requires_grad = False
    else:
        for module in model.modules():
            if isinstance(module, VariationalLinear):
                module.weight_logvar.requires_grad = True
                module.bias_logvar.requires_grad = True

    total_loss = 0.0
    loss_components = {}
    num_batches = 0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(train_loader):
        if batch is None:
            continue

        images, labels, reports = batch
        batch_dict = {
            "images": images.to(device, non_blocking=True),
            "labels": labels.to(device, non_blocking=True),
            "reports": reports
        }

        try:
            # Forward pass
            outputs = model(batch_dict, device)

            # Compute loss with epoch information
            loss, loss_dict = model.compute_loss(outputs, batch_dict["labels"], epoch=epoch)

            # Check for reasonable loss values
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 100:
                print(f"Skipping batch {batch_idx} due to extreme loss: {loss.item()}")
                continue

            loss = loss / gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation_steps
            num_batches += 1

            # Accumulate loss components
            for key, value in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += value

            # ENHANCED LOGGING with KL monitoring
            if batch_idx % 50 == 0:
                kl_val = loss_dict.get('kl_divergence', 0.0)
                class_val = loss_dict.get('classification', 0.0)
                print(f'Epoch {epoch+1}, Batch {batch_idx}: '
                      f'Loss={loss.item():.4f}, '
                      f'Classification={class_val:.4f}, '
                      f'KL={kl_val:.6f}, '
                      f'KL/Class Ratio={kl_val/(class_val+1e-8):.4f}')

        except Exception as e:
            print(f"Training error in batch {batch_idx}: {e}")
            continue

    # Final gradient step
    if num_batches % gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    # Average losses
    avg_loss = total_loss / max(num_batches, 1)
    for key in loss_components:
        loss_components[key] /= max(num_batches, 1)

    return avg_loss, loss_components
