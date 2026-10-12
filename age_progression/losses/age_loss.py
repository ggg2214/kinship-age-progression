from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class AgeEstimator(nn.Module):
    """
    Facial Age Estimation Backbone (MiVOLO-style architecture).
    Predicts continuous age in years [0, 100] from face images.
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Predicts continuous age in years (B, 1)."""
        feats = self.features(img)
        age = self.regressor(feats)
        return age.squeeze(-1)  # (B,)


class AgeEstimationLoss(nn.Module):
    """
    Age Progression Goal Loss (L_Age).
    Evaluates whether synthesized face image correctly matches the target chronological age.
    """

    def __init__(self, age_estimator: Optional[nn.Module] = None, loss_type: str = "l1"):
        super().__init__()
        self.age_estimator = age_estimator or AgeEstimator(pretrained=False)
        self.loss_type = loss_type

    def forward(self, pred_img: torch.Tensor, target_age: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_img: Synthesized aged face image (B, 3, H, W)
            target_age: Target age in years (B,)
        Returns:
            Scalar age discrepancy loss
        """
        predicted_age = self.age_estimator(pred_img)
        target_age = target_age.float()

        if self.loss_type == "l1":
            # Normalizing by 100 for gradient stability across different ages
            loss_age = F.l1_loss(predicted_age, target_age) / 100.0
        elif self.loss_type == "smooth_l1":
            loss_age = F.smooth_l1_loss(predicted_age, target_age) / 100.0
        else:
            loss_age = F.mse_loss(predicted_age, target_age) / 100.0

        return loss_age

