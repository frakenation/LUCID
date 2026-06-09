import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicalConstraintLoss(nn.Module):
    def forward(self, background, flare, lq_image):
        lol_gamma = torch.pow(background.clamp(min=1e-7), 1 / 2.2)
        flare_gamma = torch.pow(flare.clamp(min=1e-7), 1 / 2.2)
        lq_gamma = (lol_gamma + flare_gamma).clamp(0, 1)
        reconstructed = torch.pow(lq_gamma, 2.2)
        return F.mse_loss(reconstructed, lq_image)


class FlareDisentanglementLoss(nn.Module):
    def __init__(self, lambda_supervised=1.0, lambda_physical=0.5, lambda_orthogonal=0.2):
        super().__init__()
        self.lambda_supervised = lambda_supervised
        self.lambda_physical = lambda_physical
        self.lambda_orthogonal = lambda_orthogonal
        self.physical_loss = PhysicalConstraintLoss()

    def forward(self, background, flare, lq_image, orth_loss,
                flare_gt=None, lol_gt=None):
        losses = {}
        total_loss = 0

        reconstruction_loss = self.physical_loss(background, flare, lq_image)
        total_loss += self.lambda_physical * reconstruction_loss
        losses['physical_reconstruction'] = reconstruction_loss

        total_loss += self.lambda_orthogonal * orth_loss
        losses['orthogonal'] = orth_loss

        supervised_loss = 0
        if flare_gt is not None:
            flare_loss = F.l1_loss(flare, flare_gt)
            supervised_loss += flare_loss
            losses['flare_supervised'] = flare_loss

        if lol_gt is not None:
            lol_loss = F.l1_loss(background, lol_gt)
            supervised_loss += lol_loss * 0.5
            losses['lol_supervised'] = lol_loss

        if supervised_loss > 0:
            total_loss += self.lambda_supervised * supervised_loss

        losses['total'] = total_loss
        return total_loss, losses
