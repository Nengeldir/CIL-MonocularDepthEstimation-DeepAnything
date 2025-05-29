import torch
import torch.nn as nn
from torchmetrics import PearsonCorrCoef

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


class SIRMSECorrelation(nn.Module):
    def __init__(self, device, lam=0.1):
        super(SIRMSECorrelation, self).__init__()
        self.pearson = PearsonCorrCoef()
        self.pearson = PearsonCorrCoef().to(device)
        self.lam = lam

        self.sirmse = None
        self.correlation = None
    
    def forward(self, pred, target, uncertainty):
        pred = pred.squeeze()
        target = target.squeeze()

        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        log_pred = torch.log(pred_flat + 1e-8)
        log_target = torch.log(target_flat + 1e-8)

        delta = log_target - log_pred

        alpha = delta.mean(dim=1, keepdim=True)
        
        SIRMSE_pixel = ((-delta + alpha).square().clamp_min(1e-12)).sqrt()

        sirmse = ((-delta + alpha).square().clamp_min(1e-12)).mean(dim=1).sqrt().mean()

        SIRMSE_flat = SIRMSE_pixel.detach().flatten()
        uncertainty_flat = uncertainty.squeeze().flatten()

        correlation = self.pearson(SIRMSE_flat, uncertainty_flat)

        self.sirmse = sirmse.detach()
        self.correlation = correlation.detach()

        loss = (1-self.lam) * sirmse + self.lam * (1-correlation)
        return loss


