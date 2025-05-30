import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path

from src.utils_unet import ensure_dir, train_model, evaluate_model, generate_test_predictions, get_best_device
from src.models import SimpleUNet, AdvancedUNEt, AdvancedUNetPooling
from src.transforms import target_transform
from src.datasets import DepthDataset
from src.loss import scaleinvariant_RMSE

#data_dir = '/kaggle/input/ethz-cil-monocular-depth-estimation-2025'
data_dir = Path('./data')
train_dir = os.path.join(data_dir, 'train/train')
test_dir = os.path.join(data_dir, 'test/test')
train_list_file = os.path.join(data_dir, 'train_list.txt')
test_list_file = os.path.join(data_dir, 'test_list.txt')
output_dir = Path('./output')

BATCH_SIZE = 4
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 10
DEVICE = get_best_device()
INPUT_SIZE = (426, 560)
NUM_WORKERS = 1
PIN_MEMORY = True

# Define the model, loss and start epoch. If we already trained up to epoch 3, we would input 4 for start epoch to continue improving the model
MODEL_CLASS_LIST = [
    (SimpleUNet, scaleinvariant_RMSE(), 0),
    (AdvancedUNEt, scaleinvariant_RMSE(), 0),
    (AdvancedUNetPooling, scaleinvariant_RMSE(), 0)
    ]
def main():

    for MODEL_CLASS, LOSS, START_EPOCH in MODEL_CLASS_LIST:

        ############
        results_dir = os.path.join(output_dir, f'results_{MODEL_CLASS.__name__}_{LOSS.__class__.__name__}')
        predictions_dir = os.path.join(output_dir, f'predictions_{MODEL_CLASS.__name__}_{LOSS.__class__.__name__}')
        ###########


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

        transform_fn = lambda depth: target_transform(depth, INPUT_SIZE)

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
        train_size = int(0.85 * total_size)  # 85% for training
        val_size = int(total_size - train_size)  # 15% for validation

        # Set a fixed random seed for reproducibility
        torch.manual_seed(0)

        train_dataset, val_dataset = torch.utils.data.random_split(
            train_full_dataset, [train_size, val_size]
        )

        # Create data loaders with memory optimizations
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=PIN_MEMORY,
            drop_last=True,
            persistent_workers=False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=PIN_MEMORY
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
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

        model = MODEL_CLASS()
        model = nn.DataParallel(model)
        model = model.to(DEVICE)
        print(f"Using device: {DEVICE}")

        # Print memory usage after model initialization
        if torch.cuda.is_available():
            print(f"Memory allocated after model init: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

        # Define loss function and optimizer
        criterion = LOSS
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        if START_EPOCH > 0:
            model.load_state_dict(torch.load(os.path.join(results_dir, 'best_model.pth')))

        # Train the model
        print("Starting training...")
        model = train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, DEVICE, results_dir=results_dir, start_epoch=START_EPOCH, val_criterion=scaleinvariant_RMSE())
        # Evaluate the model on validation set
        print("Evaluating model on validation set...")
        metrics = evaluate_model(model, val_loader, DEVICE, results_dir)

        # Print metrics
        print("\nValidation Metrics:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        # Save metrics to file
        with open(os.path.join(results_dir, 'validation_metrics.txt'), 'w') as f:
            for name, value in metrics.items():
                f.write(f"{name}: {value:.4f}\n")
        # Generate predictions for the test set
        print("Generating predictions for test set...")
        generate_test_predictions(model, test_loader, DEVICE, predictions_dir)
        
        print(f"Results saved to {results_dir}")
        print(f"All test depth map predictions saved to {predictions_dir}")

main()
