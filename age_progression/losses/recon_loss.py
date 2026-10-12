import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    """
    Reconstruction Loss (L_recon).
    Minimizes pixel-level (L1) and gradient differences between predicted and target images.
    Preserves structural stability and prevents synthesis artifacts.
    """

    def __init__(self, l1_weight: float = 1.0, grad_weight: float = 0.5):
        super().__init__()
        self.l1_weight = l1_weight
        self.grad_weight = grad_weight

    def _image_gradients(self, img: torch.Tensor):
        dx = img[:, :, :, 1:] - img[:, :, :, :-1]
        dy = img[:, :, 1:, :] - img[:, :, :-1, :]
        return dx, dy

    def forward(self, pred_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_img: Generated image (B, 3, H, W)
            target_img: Ground truth or reference face image (B, 3, H, W)
        """
        l1_loss = F.l1_loss(pred_img, target_img)

        # Gradient consistency loss (preserves edge sharpness and facial contours)
        pred_dx, pred_dy = self._image_gradients(pred_img)
        tgt_dx, tgt_dy = self._image_gradients(target_img)
        grad_loss = F.l1_loss(pred_dx, tgt_dx) + F.l1_loss(pred_dy, tgt_dy)

        return self.l1_weight * l1_loss + self.grad_weight * grad_loss

