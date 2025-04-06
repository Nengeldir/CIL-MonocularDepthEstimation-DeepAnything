import torch

def target_transform(depth, input_size):
    # Resize the depth map to match input size
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(0).unsqueeze(0), 
        size=input_size, 
        mode='bilinear', 
        align_corners=True
    ).squeeze()
    
    # Add channel dimension to match model output
    depth = depth.unsqueeze(0)
    return depth