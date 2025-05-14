import torch
import torch.nn as nn

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
