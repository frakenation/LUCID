import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicalConstraintLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def sparsity_loss(self, x):
        return torch.mean(torch.abs(x))

    def edge_preservation_loss(self, pred, target):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        sobel_x = sobel_x.view(1, 1, 3, 3).repeat(pred.size(1), 1, 1, 1).to(pred.device)
        sobel_y = sobel_y.view(1, 1, 3, 3).repeat(pred.size(1), 1, 1, 1).to(pred.device)

        pred_edge_x = F.conv2d(pred, sobel_x, padding=1, groups=pred.size(1))
        pred_edge_y = F.conv2d(pred, sobel_y, padding=1, groups=pred.size(1))
        target_edge_x = F.conv2d(target, sobel_x, padding=1, groups=target.size(1))
        target_edge_y = F.conv2d(target, sobel_y, padding=1, groups=target.size(1))

        edge_loss_x = F.l1_loss(pred_edge_x, target_edge_x)
        edge_loss_y = F.l1_loss(pred_edge_y, target_edge_y)
        return edge_loss_x + edge_loss_y

    def forward(self, background_context, background, flare, lq_image):
        losses = {}

        lol_gamma = torch.pow(background.clamp(min=1e-7), 1 / 2.2)
        flare_gamma = torch.pow(flare.clamp(min=1e-7), 1 / 2.2)
        lq_gamma = (lol_gamma + flare_gamma).clamp(0, 1)
        reconstructed = torch.pow(lq_gamma, 2.2)

        losses['reconstruction'] = F.mse_loss(reconstructed, lq_image)
        losses['flare_sparsity'] = self.sparsity_loss(flare) * 0.0
        losses['background_context_range'] = F.relu(-background_context).mean()
        losses['background_range'] = (F.relu(-background) + F.relu(background - 1)).mean()
        losses['flare_range'] = F.relu(-flare).mean()
        losses['detail_preservation'] = self.edge_preservation_loss(background, lq_image) * 0.3
        return losses


class FlareDisentanglementLoss(nn.Module):
    def __init__(self, lambda_supervised=1.0, lambda_physical=0.5, lambda_orthogonal=0.2):
        super().__init__()
        self.lambda_supervised = lambda_supervised
        self.lambda_physical = lambda_physical
        self.lambda_orthogonal = lambda_orthogonal
        self.physical_loss = PhysicalConstraintLoss()

    def forward(self, background_context, background, flare, lq_image, orth_loss,
                flare_gt=None, lol_gt=None):
        losses = {}
        total_loss = 0

        physical_losses = self.physical_loss(background_context, background, flare, lq_image)
        physical_total = sum(physical_losses.values())
        total_loss += self.lambda_physical * physical_total
        losses.update({f'physical_{k}': v for k, v in physical_losses.items()})

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
