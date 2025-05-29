import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import PearsonCorrCoef

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def train_model_double_decoder(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, results_dir):
    writer = SummaryWriter(log_dir=os.path.join(results_dir, 'tensorboard_logs'))
    import time

    start_time = time.time()
    for batch_idx, (inputs, targets, _) in enumerate(train_loader):
        if batch_idx == 10:  # Check the first few batches
            break
    end_time = time.time()
    print(f"Time to load 10 batches: {end_time - start_time:.2f} seconds")

    """Train the model and save the best based on validation metrics"""
    best_val_loss = float('inf')
    best_epoch = 0
    train_losses = []
    val_losses = []

    with open(os.path.join(results_dir, 'train_metrics.txt'), 'w') as f:
        pass
        
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        if epoch >= 2:
            criterion.lam = 0.1
        
        # Training phase
        model.train()
        train_loss = 0.0

        batch = 0

        log_interval = 10
        for batch_idx, (inputs, targets, _) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            depth_pred, uncertainty = model(inputs)
            loss = criterion(depth_pred, targets, uncertainty)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)

            sirmse_loss = criterion.sirmse
            correlation_loss = (1 - criterion.correlation)
            
            """
            with open(os.path.join(results_dir, 'train_metrics.txt'), 'a') as f:
                f.write(f"({epoch+1}, {batch_idx+1}: {loss.item():.4f}")
            """
            
            if batch_idx % log_interval == 0:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('train/total_loss', loss.item(), global_step)
                writer.add_scalar('train/sirmse', sirmse_loss.item(), global_step)
                writer.add_scalar('train/correlation_loss', correlation_loss.item(), global_step)

            batch += 1
        
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets, _) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                depth_pred, uncertainty = model(inputs)
                loss = criterion(depth_pred, targets, uncertainty)

                sirmse_loss = criterion.sirmse
                correlation_loss = (1 - criterion.correlation)
                
                val_loss += loss.item() * inputs.size(0)

                if batch_idx % log_interval == 0:
                    global_step = epoch * len(val_loader) + batch_idx
                    writer.add_scalar('val/total_loss', loss.item(), global_step)
                    writer.add_scalar('val/sirmse', sirmse_loss.item(), global_step)
                    writer.add_scalar('val/correlation_loss', correlation_loss.item(), global_step)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
                
        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(results_dir, 'best_model.pth'))
            print(f"New best model saved at epoch {epoch+1} with validation loss: {val_loss:.4f}")
        if epoch == 4:
            torch.save(model.state_dict(), os.path.join(results_dir, 'epoch_5_model.pth'))
            print(f"Model Saved from epoch {epoch+1} with validation loss: {val_loss:.4f}")
        if epoch == 9:
            torch.save(model.state_dict(), os.path.join(results_dir, 'epoch_10_model.pth'))
            print(f"Model Saved from epoch {epoch+1} with validation loss: {val_loss:.4f}")
    
    print(f"\nBest model was from epoch {best_epoch+1} with validation loss: {best_val_loss:.4f}")
    
    # Load the best model
    model.load_state_dict(torch.load(os.path.join(results_dir, 'best_model.pth')))

    writer.close()
    
    return model

def evaluate_model_double_decoder(model, val_loader, device, results_dir):
    """Evaluate the model and compute metrics on validation set"""
    model.eval()
    
    mae = 0.0
    rmse = 0.0
    rel = 0.0
    delta1 = 0.0
    delta2 = 0.0
    delta3 = 0.0
    sirmse = 0.0
    
    total_samples = 0
    target_shape = None

    uncertainty_error_correlation = []
    
    with torch.no_grad():
        pearson = PearsonCorrCoef().to(device)

        for inputs, targets, filenames in tqdm(val_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            total_samples += batch_size
            
            if target_shape is None:
                target_shape = targets.shape
            
            # Forward pass
            outputs, uncertainty = model(inputs)

            pred = outputs.squeeze()
            target = targets.squeeze()

            pred_flat = pred.view(pred.size(0), -1)
            target_flat = target.view(target.size(0), -1)

            log_pred = torch.log(pred_flat + 1e-8)
            log_target = torch.log(target_flat + 1e-8)

            delta = log_target - log_pred

            alpha = delta.mean(dim=1, keepdim=True)
            
            SIRMSE_pixel = ((-delta + alpha).square().clamp_min(1e-12)).sqrt()

            SIRMSE_flat = SIRMSE_pixel.flatten()
            uncertainty_flat = uncertainty.squeeze().flatten()

            correlation = pearson(SIRMSE_flat, uncertainty_flat)

            uncertainty_error_correlation.append(correlation)
            
            # Resize outputs to match target dimensions
            outputs = nn.functional.interpolate(
                outputs,
                size=targets.shape[-2:],  # Match height and width of targets
                mode='bilinear',
                align_corners=True
            )

            uncertainty = nn.functional.interpolate(
                uncertainty,
                size=targets.shape[-2:],  # Match height and width of targets
                mode='bilinear',
                align_corners=True
            )
            
            # Calculate metrics
            abs_diff = torch.abs(outputs - targets)
            mae += torch.sum(abs_diff).item()
            rmse += torch.sum(torch.pow(abs_diff, 2)).item()
            rel += torch.sum(abs_diff / (targets + 1e-6)).item()
            
            # Calculate scale-invariant RMSE for each image in the batch
            for i in range(batch_size):
                # Convert tensors to numpy arrays
                pred_np = outputs[i].cpu().squeeze().numpy()
                target_np = targets[i].cpu().squeeze().numpy()
                
                EPSILON = 1e-6
                
                valid_target = target_np > EPSILON
                if not np.any(valid_target):
                    continue
                
                target_valid = target_np[valid_target]
                pred_valid = pred_np[valid_target]
                
                log_target = np.log(target_valid)
                
                pred_valid = np.where(pred_valid > EPSILON, pred_valid, EPSILON)
                log_pred = np.log(pred_valid)
                
                # Calculate scale-invariant error
                diff = log_pred - log_target
                diff_mean = np.mean(diff)
                
                # Calculate RMSE for this image
                sirmse += np.sqrt(np.mean((diff - diff_mean) ** 2))
            
            # Calculate thresholded accuracy
            max_ratio = torch.max(outputs / (targets + 1e-6), targets / (outputs + 1e-6))
            delta1 += torch.sum(max_ratio < 1.25).item()
            delta2 += torch.sum(max_ratio < 1.25**2).item()
            delta3 += torch.sum(max_ratio < 1.25**3).item()
            
            # Save some sample predictions
            if total_samples <= 5 * batch_size:
                for i in range(min(batch_size, 5)):
                    idx = total_samples - batch_size + i
                    
                    # Convert tensors to numpy arrays
                    input_np = inputs[i].cpu().permute(1, 2, 0).numpy()
                    target_np = targets[i].cpu().squeeze().numpy()
                    output_np = outputs[i].cpu().squeeze().numpy()
                    uncertainty_np = uncertainty[i].cpu().squeeze().numpy()
                    
                    
                    # Normalize for visualization
                    input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min() + 1e-6)
                    
                    # Create visualization
                    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

                    axes[0].imshow(input_np)
                    axes[0].set_title("RGB Input")
                    axes[0].axis('off')

                    axes[1].imshow(target_np, cmap='plasma')
                    axes[1].set_title("Ground Truth Depth")
                    axes[1].axis('off')

                    axes[2].imshow(output_np, cmap='plasma')
                    axes[2].set_title("Predicted Depth")
                    axes[2].axis('off')

                    # For the uncertainty map, save the image object
                    im = axes[3].imshow(uncertainty_np, cmap='plasma')
                    axes[3].set_title("Uncertainty Map")
                    axes[3].axis('off')

                    # Add colorbar for the uncertainty map
                    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

                    plt.tight_layout()
                    plt.savefig(os.path.join(results_dir, f"sample_{idx}.png"))
                    plt.close()
            
            # Free up memory
            del inputs, targets, outputs, abs_diff, max_ratio, uncertainty
            
        # Clear CUDA cache
        torch.cuda.empty_cache()
    
    # Calculate final metrics using stored target shape
    total_pixels = target_shape[1] * target_shape[2] * target_shape[3]  # channels * height * width
    mae /= total_samples * total_pixels
    rmse = np.sqrt(rmse / (total_samples * total_pixels))
    rel /= total_samples * total_pixels
    sirmse = sirmse / total_samples
    delta1 /= total_samples * total_pixels
    delta2 /= total_samples * total_pixels
    delta3 /= total_samples * total_pixels

    if uncertainty_error_correlation:
        correlation_tensor = torch.stack(uncertainty_error_correlation)
        avg_uncertainty_correlation = correlation_tensor.mean().cpu().item()
    else:
        avg_uncertainty_correlation = 0.0
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        'siRMSE': sirmse,
        'REL': rel,
        'Delta1': delta1,
        'Delta2': delta2,
        'Delta3': delta3,
        'Uncertainty_Error_Correlation': avg_uncertainty_correlation 
    }

    writer = SummaryWriter(log_dir=os.path.join(results_dir, 'tensorboard_logs'))
    for name, value in metrics.items():
        writer.add_scalar(f'Metrics/{name}', value, 0)  # Use step=0 for final metrics
    writer.close()
    
    return metrics

def generate_test_predictions_double_decoder(model, test_loader, device, predictions_dir):
    """Generate predictions for the test set without ground truth"""
    model.eval()

    EPSILON = 1e-6
    
    # Ensure predictions directory exists
    ensure_dir(predictions_dir)
    
    with torch.no_grad():
        for inputs, filenames in tqdm(test_loader, desc="Generating Test Predictions"):
            inputs = inputs.to(device)
            batch_size = inputs.size(0)
            
            all_outputs = []
            # Forward pass
            outputs, uncertainty = model(inputs)
            
            # Resize outputs to match original input dimensions (426x560)
            outputs = nn.functional.interpolate(
                outputs,
                size=(426, 560),  # Original input dimensions
                mode='bilinear',
                align_corners=True
            )

            outputs = torch.clamp(outputs, min=EPSILON)

            uncertainty = nn.functional.interpolate(
                uncertainty,
                size=(426, 560),  # Original input dimensions
                mode='bilinear',
                align_corners=True
            )
            
            # Save all test predictions
            for i in range(batch_size):
                # Get filename without extension
                filename = filenames[i].split(' ')[1]
                
                # Save depth map prediction as numpy array
                depth_pred = outputs[i].cpu().squeeze().numpy()
                np.save(os.path.join(predictions_dir, f"{filename}"), depth_pred)

                # Save uncertainty map
                new_filename = filename.replace('depth.npy', 'uncertainty.npy')
                uncertainty_map = uncertainty[i].cpu().squeeze().numpy()
                np.save(os.path.join(predictions_dir, f"{new_filename}"), uncertainty_map)
            
            # Clean up memory
            del inputs, all_outputs, outputs, uncertainty
        
        # Clear cache after test predictions
        torch.cuda.empty_cache()