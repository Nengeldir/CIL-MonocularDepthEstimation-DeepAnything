import os
import glob
import random

# Define the dataset path
data_path = "data"  # Update this if your dataset is in a different location

# Define the output directory for splits
split_dir = os.path.join("splits", "deep_anything")
os.makedirs(split_dir, exist_ok=True)

# Get all training image paths
train_images = sorted(glob.glob(os.path.join(data_path, "train", "train", "sample_*_rgb.png")))

# Shuffle the images for randomness
random.seed(42)  # Ensure reproducibility
random.shuffle(train_images)

# Define the validation split ratio
val_split = 0.1  # 10% of the data for validation
split_index = int(len(train_images) * (1 - val_split))

# Split into training and validation sets
train_set = train_images[:split_index]
val_set = train_images[split_index:]

# Write train_files.txt
with open(os.path.join(split_dir, "train_files.txt"), "w") as f:
    for img_path in train_set:
        # Write relative paths
        relative_path = os.path.relpath(img_path, data_path)
        f.write(relative_path + "\n")

print(f"Generated {len(train_set)} entries in train_files.txt")

# Write val_files.txt
with open(os.path.join(split_dir, "val_files.txt"), "w") as f:
    for img_path in val_set:
        # Write relative paths
        relative_path = os.path.relpath(img_path, data_path)
        f.write(relative_path + "\n")

print(f"Generated {len(val_set)} entries in val_files.txt")