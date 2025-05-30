import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import time

from src.utils import ensure_dir
from src.models import UncertaintyUNEt
from src.transforms import target_transform
from src.datasets import DepthDataset


data_dir = Path('./data')
train_dir = os.path.join(data_dir, 'train/train')
test_dir = os.path.join(data_dir, 'test/test')
train_list_file = os.path.join(data_dir, 'train_list.txt')
test_list_file = os.path.join(data_dir, 'test_list.txt')
output_dir = Path('./output')
results_dir = os.path.join(output_dir, 'results')
predictions_dir = os.path.join(output_dir, 'predictions')

BATCH_SIZE = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# DEVICE = get_best_device()
INPUT_SIZE = (426, 560)
NUM_WORKERS = 4
PIN_MEMORY = True

def get_reference_patches(rgb, depth, patch_size=7, stride=4):
    half_size = patch_size // 2
    h, w = rgb.shape[2], rgb.shape[3]
    patches = []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            if (y - half_size >= 0 and y + half_size < h and
                x - half_size >= 0 and x + half_size < w):
                color_patch = rgb[0, :, y - half_size : y + half_size + 1,
                                      x - half_size : x + half_size + 1]
                depth_patch = depth[0, :, y - half_size : y + half_size + 1,
                                         x - half_size : x + half_size + 1]
                patches.append({'color': color_patch,
                                'depth': depth_patch,
                               'center': (y,x)})
    return patches

def compute_patch_distance(reference_patch, comparing_patch, alpha=0.4, beta=30.0):
    Ir, Dr = reference_patch['color'], reference_patch['depth']
    Ii, Di = comparing_patch['color'], comparing_patch['depth']

    Dr_med, Di_med = np.median(Dr), np.median(Di)

    color_dist = np.linalg.norm(Ir - Ii)

    depth_dist = np.linalg.norm((Dr - Dr_med) - (Di - Di_med))

    return alpha * color_dist + beta * depth_dist

def get_similar_patches(reference_patch, other_patches, ref_index, k=40):
    distances = []
    for i, comparing_patch in enumerate(other_patches):
        if i == ref_index:
            distances.append(float('inf'))
        else:
            distance = compute_patch_distance(reference_patch, comparing_patch)
            distances.append(distance)

    nearest_indices = np.argsort(distances)[:k-1]
    similar_patches = [other_patches[i] for i in nearest_indices]

    return similar_patches

def assemble_patch_matrix(reference_patch, similar_patches):
    patches = [reference_patch] + similar_patches
    k = len(patches)
    m = patches[0]['color'].shape[1]
    patch_vectors = []

    for patch in patches:
        color = patch['color'].reshape(3,-1)
        depth = patch['depth'].reshape(1,-1)
        vector = np.concatenate([color,depth], axis=0).reshape(-1)
        patch_vectors.append(vector)

    patch_matrix = np.stack(patch_vectors, axis=1)

    return patch_matrix

def svd_lowrank_approx(M, ranks=[3]):
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    reconstructions = {}
    losses = {}
    for r in ranks:
        U_r = U[:, :r]
        S_r = np.diag(S[:r])
        Vt_r = Vt[:r, :]
        reconstructions[r] = U_r @ S_r @ Vt_r
        loss = np.sum(S[r:] ** 2)
        losses[r] = loss
    return reconstructions, losses

def reconstruct_depth_from_patch_matrix(patch_matrix, patch_size):
    m = patch_size
    k = patch_matrix.shape[1]
    depth_patches = []
    for i in range(k):
        patch_vec = patch_matrix[:, i]
        depth_vec = patch_vec[3*m*m:]
        depth_patch = depth_vec.reshape(m,m)
        depth_patches.append(depth_patch)

    return depth_patches

def aggregate_depth_patches(patch_locations, depth_patches, original_depth, image_shape=INPUT_SIZE):
    H, W = image_shape
    depth_sum = np.zeros((H,W))
    depth_count = np.zeros((H,W))
    m = depth_patches[0].shape[0]
    half_size = m //2

    for (y,x), patch in zip(patch_locations, depth_patches):
        for dy in range(-half_size, half_size+1):
            for dx in range(-half_size, half_size+1):
                iy, ix = y + dy, x+dx
                if 0 <= iy < H and 0 <= ix < W:
                    depth_sum[iy, ix] += patch[dy + half_size, dx + half_size]
                    depth_count[iy, ix] += 1

    valid_mask = depth_count > 0
    final_depth = np.zeros((H,W))
    final_depth[valid_mask] = depth_sum[valid_mask]/ depth_count[valid_mask]

    updated_depth = np.copy(original_depth)
    updated_depth[valid_mask] = final_depth[valid_mask]

    return updated_depth

def lowrank(rgb, depth, uncertainty, index, ranks):
    """
    rgb: rgb image of the input
    depth: estimated depth map of the rgb image
    uncertainty: estimated uncertainty map
    """
    
    patch_size = 7

    reference_patches = get_reference_patches(rgb, depth, patch_size)

    uncertainty_np = uncertainty[0,0].cpu().numpy()
    mean_uncertainty = np.mean(uncertainty_np)
    std_uncertainty = np.std(uncertainty_np)
    threshold = mean_uncertainty +2*std_uncertainty


    for ref_index, reference_patch in enumerate(reference_patches):

        y, x = reference_patch['center']

        patch_uncertainty = uncertainty_np[y:y+patch_size, x:x+patch_size]
        patch_mean_uncertainty = np.mean(patch_uncertainty)

        if patch_mean_uncertainty > threshold:
            similar_patches = get_similar_patches(reference_patch, reference_patches, ref_index)
    
            M = assemble_patch_matrix(reference_patch, similar_patches)
    
            mean_M = np.mean(M, axis=1, keepdims=True)
            M -= mean_M
    
            M_approx_dict, losses_dict = svd_lowrank_approx(M, ranks)
    
            reconstructed_patches_dict = {}
            for r, M_approx in M_approx_dict.items():
                # Add back the mean
                M_approx += mean_M
                # Reconstruct depth patches
                reconstructed_depth_patches = reconstruct_depth_from_patch_matrix(M_approx, patch_size)
                reconstructed_patches_dict[r] = reconstructed_depth_patches
            
            depth_maps_by_rank = {}
            for r, reconstructed_depth_patches in reconstructed_patches_dict.items():
                # Aggregate patches for this rank
                patch_info = [reference_patch['center']] + [patch['center'] for patch in similar_patches]
                depth_2d = depth[0, 0]
                updated_depth = aggregate_depth_patches(patch_info, reconstructed_depth_patches, depth_2d)
                
                # Store the updated depth map for this rank
                depth_maps_by_rank[r] = updated_depth.copy() 

    return depth_maps_by_rank, losses_dict

    

def transform_fn(depth):
    return target_transform(depth, INPUT_SIZE)

if __name__ == "__main__":
    print('working')
    # Create output directories
    ensure_dir(results_dir)
    ensure_dir(predictions_dir)
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Data augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # transform_fn = lambda depth: target_transform(depth, INPUT_SIZE)

    # Create training dataset with ground truth
    train_full_dataset = DepthDataset(
        data_dir=train_dir,
        list_file=train_list_file, 
        transform=train_transform,
        target_transform=transform_fn,
        has_gt=True
    )
    
    # Create test dataset without ground truth
    test_dataset = DepthDataset(
        data_dir=test_dir,
        list_file=test_list_file,
        transform=test_transform,
        has_gt=False  # Test set has no ground truth
    )
    
    # Split training dataset into train and validation
    total_size = len(train_full_dataset)
    train_size = int(0.01 * total_size)  # 85% for training
    val_size = int(0.01 * total_size)   # 15% for validation
    rest_size = total_size - train_size - val_size
    
    # Set a fixed random seed for reproducibility
    torch.manual_seed(0)
    
    train_dataset, val_dataset, _ = torch.utils.data.random_split(
        train_full_dataset, [train_size, val_size, rest_size]
    )
    
    # Create data loaders with memory optimizations
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY,
        drop_last=True,
        persistent_workers=False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY
    )
    
    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    
    # Clear CUDA cache before model initialization
    torch.cuda.empty_cache()
    
    # Display GPU memory info
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Initially allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
    model = UncertaintyUNEt(dropout_rate=0.0)
    model = nn.DataParallel(model)
    model = model.to(DEVICE)
    print(f"Using device: {DEVICE}")

    model.eval()

    # Print memory usage after model initialization
    if torch.cuda.is_available():
        print(f"Memory allocated after model init: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

    # Load checkpoint
    model.load_state_dict(torch.load('./outputs/results/best_model.pth', map_location=DEVICE))
    print("Loaded model weights from best_model.pth")

    n = 50  # number of random images
    dataset_length = len(val_dataset)
    np.random.seed(42)
    random_indices = np.random.choice(dataset_length, n, replace=False)

    ranks = [3, 5, 7, 10, 15]

    sirmse_before = []
    sirmse_after = {r: [] for r in ranks}
    losses_by_rank = {r: [] for r in ranks}

    total_start = time.perf_counter()
    

    for i, index in enumerate(random_indices):
        img_start = time.perf_counter()
        print(f"currently processing image {i+1} out of {n}")
        
        rgb, gt_depth, filename = val_dataset.__getitem__(index)
    
        with torch.no_grad():
            rgb = rgb.unsqueeze(0).to(DEVICE)
    
            estimated_depth, uncertainty = model(rgb)
    
            pred_np = estimated_depth[0].cpu().squeeze().numpy()
            target_np = gt_depth[0].cpu().squeeze().numpy()
    
            EPSILON = 1e-6
                    
            valid_target = target_np > EPSILON
            
            target_valid = target_np[valid_target]
            pred_valid = pred_np[valid_target]
            
            log_target = np.log(target_valid)
            
            pred_valid = np.where(pred_valid > EPSILON, pred_valid, EPSILON)
            log_pred = np.log(pred_valid)
            
            # Calculate scale-invariant error
            diff = log_pred - log_target
            diff_mean = np.mean(diff)
            
            # Calculate RMSE for this image
            sirmse = np.sqrt(np.mean((diff - diff_mean) ** 2))
    
            sirmse_before.append(sirmse)
    
            # Resize outputs to match original input dimensions (426x560)
            estimated_depth = nn.functional.interpolate(
                estimated_depth,
                size=(426, 560),  # Original input dimensions
                mode='bilinear',
                align_corners=True
            )
    
            uncertainty = nn.functional.interpolate(
                uncertainty,
                size=(426, 560),  # Original input dimensions
                mode='bilinear',
                align_corners=True
            )

            depth_np = estimated_depth[0,0].cpu().numpy()
            uncertainty_np = uncertainty[0,0].cpu().numpy()

            # Create a figure with two subplots (side by side)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            # Plot depth map
            ax1.axis('off')
            im1 = ax1.imshow(depth_np, cmap='plasma')
            ax1.set_title('Depth Map')
            fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

            # Plot uncertainty map
            ax2.axis('off')
            im2 = ax2.imshow(uncertainty_np, cmap='viridis')
            ax2.set_title('Uncertainty Map')
            fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

            plt.tight_layout()
            plt.savefig(f'./updating_images/original_and_uncertainty_{i}.png', bbox_inches='tight', pad_inches=0)
            plt.close()
    
            updated_depth_dict, losses_dict = lowrank(rgb, estimated_depth, uncertainty, i, ranks)

            num_ranks = len(ranks)

            fig, axes = plt.subplots(1, num_ranks, figsize=(4*num_ranks, 4))

            for ax, r in zip(axes, ranks):
                ax.imshow(updated_depth_dict[r], cmap='plasma')
                ax.set_title(f'Rank {r}')
                ax.axis('off')

            plt.tight_layout()
            plt.savefig(f'./updating_images/updated_depth_{i}.png', bbox_inches='tight', pad_inches=0)
            plt.close()

            for r in ranks:
                pred_np = updated_depth_dict[r]
                target_np = gt_depth[0].cpu().squeeze().numpy()
                EPSILON = 1e-6
                valid_target = target_np > EPSILON
                target_valid = target_np[valid_target]
                pred_valid = pred_np[valid_target]
                log_target = np.log(target_valid)
                pred_valid = np.where(pred_valid > EPSILON, pred_valid, EPSILON)
                log_pred = np.log(pred_valid)
                diff = log_pred - log_target
                diff_mean = np.mean(diff)
                sirmse = np.sqrt(np.mean((diff - diff_mean) ** 2))
                
                # Append the metric to the list for this rank
                sirmse_after[r].append(sirmse)
                losses_by_rank[r].append(losses_dict[r])

            img_end = time.perf_counter()
            print(f"Processed image number {i+1} in {img_end - img_start:.4f} seconds")
            print(f'loss before: {sirmse_before}')
            print(f'loss after: {sirmse_after}')

    avg_losses_by_rank = {r: np.mean(losses_by_rank[r]) for r in ranks}
    plt.figure(figsize=(8, 5))
    plt.plot(list(avg_losses_by_rank.keys()), list(avg_losses_by_rank.values()), marker='o', color='teal')
    plt.title('Average Loss for Different Ranks')
    plt.xlabel('Rank')
    plt.ylabel('Average Loss (Sum of squared discarded singular values)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./updating_images/average_losses_by_rank.png')
    plt.close()
    before_average = np.mean(sirmse_before)
    before_std = np.std(sirmse_before)
    avg_sirmse_after_by_rank = {r: np.mean(sirmse_after[r]) for r in ranks}
    std_sirmse_after_by_rank = {r: np.std(sirmse_after[r]) for r in ranks}
    total_end = time.perf_counter()
    with open("./comparison_lowrank_approximation.txt", 'w') as f:
        f.write(f"Average Before: {before_average:.6f}\n")
        f.write(f"Std Before: {before_std:.6f}\n")
        for r, avg_after in avg_sirmse_after_by_rank.items():
            f.write(f"Average After (Rank {r}): {avg_after:.6f}\n")
        for r, std_after in std_sirmse_after_by_rank.items():
            f.write(f"Std After (Rank {r}): {std_after:.6f}\n")
        for r, avg_loss in avg_losses_by_rank.items():
            f.write(f"Average Frobenius Loss (Rank {r}): {avg_loss:.6f}\n")
        f.write(f"Updating time total: {total_end - total_start:.4f}")
        f.write(f"Updated indexes: {random_indices}")
        
        
