import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, results_dir):
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
        
        # Training phase
        model.train()
        train_loss = 0.0

        batch = 0
        
        for inputs, targets, _ in tqdm(train_loader, desc="Training"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)

            with open(os.path.join(results_dir, 'train_metrics.txt'), 'a') as f:
                f.write(f"({epoch+1}, {batch+1}: {loss.item():.4f}")

            batch += 1
        
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets, _ in tqdm(val_loader, desc="Validation"):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
                
        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(results_dir, 'best_model.pth'))
            print(f"New best model saved at epoch {epoch+1} with validation loss: {val_loss:.4f}")
    
    print(f"\nBest model was from epoch {best_epoch+1} with validation loss: {best_val_loss:.4f}")
    
    # Load the best model
    model.load_state_dict(torch.load(os.path.join(results_dir, 'best_model.pth')))
    
    return model


def evaluate_model(model, val_loader, device, results_dir, monte_carlo_samples=20):
    """Evaluate the model and compute metrics on validation set"""
    model.eval()

    def enable_dropout(model):
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()  # Set dropout layers to training mode
    
    enable_dropout(model)
    
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
        for inputs, targets, filenames in tqdm(val_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            total_samples += batch_size
            
            if target_shape is None:
                target_shape = targets.shape
            

            all_outputs = []
            for _ in range(monte_carlo_samples):
                outputs = model(inputs)
                
                # Resize outputs if needed
                outputs = nn.functional.interpolate(
                    outputs,
                    size=targets.shape[-2:],
                    mode='bilinear',
                    align_corners=True
                )
                all_outputs.append(outputs)
            
            # Stack all predictions
            stacked_outputs = torch.stack(all_outputs, dim=0)  # [MC_samples, batch_size, 1, H, W]
            
            # Calculate mean prediction and uncertainty
            mean_outputs = torch.mean(stacked_outputs, dim=0)  # [batch_size, 1, H, W]
            uncertainty = torch.std(stacked_outputs, dim=0)    # [batch_size, 1, H, W]
            
            # Use mean prediction for metric calculation
            outputs = mean_outputs
            
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
            
            # Calculate correlation between error and uncertainty
            for i in range(batch_size):
                error_map = abs_diff[i].cpu().squeeze().numpy()
                uncertainty_map = uncertainty[i].cpu().squeeze().numpy()
                
                # Flatten the maps
                error_flat = error_map.flatten()
                uncertainty_flat = uncertainty_map.flatten()
                
                # Calculate correlation
                if len(error_flat) > 1:  # Need at least 2 points for correlation
                    corr = np.corrcoef(error_flat, uncertainty_flat)[0, 1]
                    if not np.isnan(corr):
                        uncertainty_error_correlation.append(corr)
            
            # Save some sample predictions
            if total_samples <= 5 * batch_size:
                for i in range(min(batch_size, 5)):
                    idx = total_samples - batch_size + i
                    
                    # Convert tensors to numpy arrays
                    input_np = inputs[i].cpu().permute(1, 2, 0).numpy()
                    target_np = targets[i].cpu().squeeze().numpy()
                    output_np = outputs[i].cpu().squeeze().numpy()
                    
                    # Normalize for visualization
                    input_np = (input_np - input_np.min()) / (input_np.max() - input_np.min() + 1e-6)
                    
                    # Create visualization
                    plt.figure(figsize=(15, 5))
                    
                    plt.subplot(1, 3, 1)
                    plt.imshow(input_np)
                    plt.title("RGB Input")
                    plt.axis('off')
                    
                    plt.subplot(1, 3, 2)
                    plt.imshow(target_np, cmap='plasma')
                    plt.title("Ground Truth Depth")
                    plt.axis('off')
                    
                    plt.subplot(1, 3, 3)
                    plt.imshow(output_np, cmap='plasma')
                    plt.title("Predicted Depth")
                    plt.axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(results_dir, f"sample_{idx}.png"))
                    plt.close()
            
            # Free up memory
            del inputs, targets, outputs, abs_diff, max_ratio, all_outputs, stacked_outputs, uncertainty
            
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

    avg_uncertainty_correlation = np.mean(uncertainty_error_correlation) if uncertainty_error_correlation else 0.0
    
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
    
    return metrics

def generate_test_predictions(model, test_loader, device, predictions_dir, monte_carlo_samples=20):
    """Generate predictions for the test set without ground truth"""
    model.eval()

    def enable_dropout(model):
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
    
    enable_dropout(model)
    
    # Ensure predictions directory exists
    ensure_dir(predictions_dir)
    
    with torch.no_grad():
        for inputs, filenames in tqdm(test_loader, desc="Generating Test Predictions"):
            inputs = inputs.to(device)
            batch_size = inputs.size(0)
            
            all_outputs = []
            for _ in range(monte_carlo_samples):
                outputs = model(inputs)
                
                # Resize outputs to match original dimensions
                outputs = nn.functional.interpolate(
                    outputs,
                    size=(426, 560),  # Original input dimensions
                    mode='bilinear',
                    align_corners=True
                )
                all_outputs.append(outputs)
            
            # Stack and compute statistics
            stacked_outputs = torch.stack(all_outputs, dim=0)  # [MC_samples, B, 1, H, W]
            outputs = torch.mean(stacked_outputs, dim=0)
            uncertainty = torch.std(stacked_outputs, dim=0)
            
            # Save all test predictions
            for i in range(batch_size):
                # Get filename without extension
                filename = filenames[i].split(' ')[1]
                
                # Save depth map prediction as numpy array
                depth_pred = outputs[i].cpu().squeeze().numpy()
                np.save(os.path.join(predictions_dir, f"{filename}"), depth_pred)

                # Save uncertainty map
                uncertainty_map = uncertainty[i].cpu().squeeze().numpy()
                np.save(os.path.join(predictions_dir, f"{filename}_uncertainty.npy"), uncertainty_map)
            
            # Clean up memory
            del inputs, all_outputs, stacked_outputs, outputs, uncertainty
        
        # Clear cache after test predictions
        torch.cuda.empty_cache()

def get_best_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device
