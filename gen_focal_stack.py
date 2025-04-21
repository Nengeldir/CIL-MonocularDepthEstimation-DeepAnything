import torch
import sys
import numpy
import os
import imageio
import matplotlib.pyplot as plt
from utils.synthetic import camera_lib
import numpy as np  # Ensure numpy is imported


STACKSIZE = 10

def pprint(list):
    for item in list:
        print(item)

cwd = os.getcwd()
dataPath_test = os.path.join(cwd, 'data\\test\\test')
dataPath_train = os.path.join(cwd, 'data\\train\\train')


# get list of all png and depth files in the train directory
train_image_ids = os.listdir(dataPath_train)
train_image_ids = [f for f in train_image_ids if f.endswith('_rgb.png')]

train_depth_ids = os.listdir(dataPath_train)
train_depth_ids = [f for f in train_depth_ids if f.endswith('_depth.npy')]

# get list of all png files in the test directory
test_image_ids = os.listdir(dataPath_test)
test_image_ids = [f for f in test_image_ids if f.endswith('.png')]

assert len(train_image_ids) == len(train_depth_ids), "Number of images and depth files do not match."

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

renderer = camera_lib.GaussPSF(9).to(device)
camera = camera_lib.ThinLenCamera(device=device)


# Generate focal stack for training images
for image_id in train_image_ids:
    # Load images
    image_path = os.path.join(dataPath_train, image_id)
    if not os.path.exists(image_path):
        print(f"Image {image_path} does not exist.")
        raise FileNotFoundError(f"Image {image_path} does not exist.")
    
    depth_path = os.path.join(dataPath_train, image_id.replace('_rgb.png', '_depth.npy'))
    if not os.path.exists(depth_path):
        print(f"Depth file {depth_path} does not exist.")
        raise FileNotFoundError(f"Depth file {depth_path} does not exist.")
    
    # Both files exist, proceed to generate focal stack

    # Read rgb_image and depth_image, convert to float and move to device
    rgb_image = imageio.imread(image_path)
    rgb_image = torch.from_numpy(rgb_image).float()
    rgb_image = rgb_image.to(device)

    depth_image = numpy.load(depth_path)
    depth_image = torch.from_numpy(depth_image).float()
    depth_image = depth_image.to(device)

    # Convert depth map to meters
    # depth_img / 1000.0 # Uncomment if depth is in mm (I guess they are in m already)

    depth_image = depth_image.unsqueeze(0)

    rgb_image = rgb_image.permute(2, 0, 1)  # Change to (C, H, W) format

    # Normalize depth and rgb image to [0, 1] range
    rgb_image = rgb_image / 255.0
    min_depth = torch.min(depth_image)
    max_depth = torch.max(depth_image)

    focus_distances = torch.linspace(min_depth, max_depth, steps = STACKSIZE).to(device)

    focal_stack = camera_lib.render_defocus(
        rgb_image,
        depth_image,
        camera,
        renderer,
        focus_distances
    ).to(device)

    # Save focal stack
    for i in range(STACKSIZE):
        output_path = os.path.join(dataPath_train, image_id.replace('_rgb.png', f'_focal_stack_{i}.png'))
        if os.path.exists(output_path):
            print(f"Focal stack {i} for {image_id} already exists. Skipping.")
            continue
        else:
            # Convert the focal stack image to uint8
            focal_stack_image = (focal_stack[i].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            imageio.imwrite(output_path, focal_stack_image)