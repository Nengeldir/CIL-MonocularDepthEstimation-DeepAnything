import torch
import torch.nn as nn
import torch.nn.functional as F


class scaleinvariant_RMSE(nn.Module):
    def __init__(self):
        super(scaleinvariant_RMSE, self).__init__()

    def forward(self, pred, target):
        pred = pred.squeeze()
        target = target.squeeze()

        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        log_pred = torch.log(pred_flat + 1e-8)
        log_target = torch.log(target_flat + 1e-8)

        delta = log_target - log_pred

        alpha = delta.mean(dim=1, keepdim=True)

        loss = ((-delta + alpha) ** 2).mean(dim=1).sqrt()

        return loss.mean()

class ScaleInvariantGradientLoss(nn.Module):
    def __init__(self):
        super(ScaleInvariantGradientLoss, self).__init__()

    def forward(self, pred, target):
        pred = pred.squeeze()
        target = target.squeeze()

        # Normalize depth to unit mean (scale-invariant)
        pred = pred / (pred.mean(dim=(1, 2), keepdim=True) + 1e-8)
        target = target / (target.mean(dim=(1, 2), keepdim=True) + 1e-8)

        # Compute gradients
        pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
        pred_dy = pred[..., 1:, :] - pred[..., :-1, :]

        target_dx = target[..., :, 1:] - target[..., :, :-1]
        target_dy = target[..., 1:, :] - target[..., :-1, :]

        # Pad to original size
        dx_loss = F.pad((pred_dx - target_dx) ** 2, (0, 1), mode='replicate')
        dy_loss = F.pad((pred_dy - target_dy) ** 2, (0, 0, 0, 1), mode='replicate')

        return (dx_loss + dy_loss).mean()

def ssim(img1, img2, C1=0.01 ** 2, C2=0.03 ** 2):
    """Compute SSIM (Structural Similarity Index Measure)."""
    mu1 = F.avg_pool2d(img1, 3, 1, 1)
    mu2 = F.avg_pool2d(img2, 3, 1, 1)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 ** 2, 3, 1, 1) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 ** 2, 3, 1, 1) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, 3, 1, 1) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return torch.clamp((1 - ssim_map) / 2, 0, 1)

class ScaleInvariantPhotometricLoss(nn.Module):
    def __init__(self):
        super(ScaleInvariantPhotometricLoss, self).__init__()

    def forward(self, reconstructed, target):
        # Normalize both images to zero mean and unit variance
        reconstructed = (reconstructed - reconstructed.mean(dim=(1,2,3), keepdim=True)) / (reconstructed.std(dim=(1,2,3), keepdim=True) + 1e-8)
        target = (target - target.mean(dim=(1,2,3), keepdim=True)) / (target.std(dim=(1,2,3), keepdim=True) + 1e-8)

        ssim_loss = ssim(reconstructed, target)

        loss = ssim_loss
        return loss.mean()


class CombinedLoss(nn.Module):
    def __init__(self, sirmse_ratio, gradient_ratio, photometric_ratio):

        super(CombinedLoss, self).__init__()
        self.SIRMSE_ratio = sirmse_ratio
        self.gradient_ratio = gradient_ratio
        self.photometric_ratio = photometric_ratio

        self.SIRMSE_Loss = scaleinvariant_RMSE()
        self.Gradient_Loss = ScaleInvariantGradientLoss()
        self.PHOTO_Loss = ScaleInvariantPhotometricLoss()


    def forward(self, pred, target):
        return (self.SIRMSE_ratio * self.SIRMSE_Loss(pred, target) + self.gradient_ratio * self.Gradient_Loss(pred, target)
                + self.photometric_ratio * self.PHOTO_Loss(pred, target))

