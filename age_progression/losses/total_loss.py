from typing import Dict, Optional, Any
import torch
import torch.nn as nn

from ..config import AgeProgressionConfig
from .recon_loss import ReconstructionLoss
from .id_loss import IdentityLoss
from .age_loss import AgeEstimationLoss, AgeEstimator
from .kinship_loss import KinshipLoss


class TotalAgeProgressionLoss(nn.Module):
    """
    Composite Loss Optimizer for Age Progression:
    Loss_Total = lambda_1 * Loss_Recon + lambda_2 * Loss_ID + lambda_3 * Loss_Age + lambda_4 * Loss_Kinship
    """

    def __init__(
        self,
        config: Optional[AgeProgressionConfig] = None,
        id_encoder: Optional[nn.Module] = None,
        age_estimator: Optional[nn.Module] = None
    ):
        super().__init__()
        self.config = config or AgeProgressionConfig()

        # Component loss functions
        self.recon_loss_fn = ReconstructionLoss()
        self.id_loss_fn = IdentityLoss(id_encoder=id_encoder) if id_encoder is not None else None
        self.age_loss_fn = AgeEstimationLoss(age_estimator=age_estimator)
        self.kinship_loss_fn = KinshipLoss(embed_dim=self.config.id_embed_dim)

    @property
    def lambda_recon(self) -> float:
        return self.config.lambda_recon

    @lambda_recon.setter
    def lambda_recon(self, val: float):
        self.config.lambda_recon = val

    @property
    def lambda_id(self) -> float:
        return self.config.lambda_id

    @lambda_id.setter
    def lambda_id(self, val: float):
        self.config.lambda_id = val

    @property
    def lambda_age(self) -> float:
        return self.config.lambda_age

    @lambda_age.setter
    def lambda_age(self, val: float):
        self.config.lambda_age = val

    @property
    def lambda_kinship(self) -> float:
        return self.config.lambda_kinship

    @lambda_kinship.setter
    def lambda_kinship(self, val: float):
        self.config.lambda_kinship = val

    def set_id_encoder(self, id_encoder: nn.Module):
        """Sets the identity encoder if not provided at initialization."""
        self.id_loss_fn = IdentityLoss(id_encoder=id_encoder)

    def forward(
        self,
        model_outputs: Dict[str, Any],
        child_img: torch.Tensor,
        target_age: torch.Tensor,
        ground_truth_target_img: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the total combined loss and its individual components.

        Args:
            model_outputs: Dictionary returned by AgeProgressionModel.forward
            child_img: Original child face image (B, 3, H, W)
            target_age: Desired target age in years (B,)
            ground_truth_target_img: Optional actual older image (for paired datasets)
        Returns:
            Dictionary containing 'loss_total' and each component loss
        """
        pred_img = model_outputs["pred_img"]
        v_c = model_outputs.get("v_c")
        v_f = model_outputs.get("v_f")
        v_m = model_outputs.get("v_m")
        v_kin = model_outputs.get("v_kin")

        # 1. Loss_Recon: Structural & Pixel Reconstruction Loss
        # If ground truth older image is available, compare to it; otherwise compare to child base geometry
        ref_img = ground_truth_target_img if ground_truth_target_img is not None else child_img
        loss_recon = self.recon_loss_fn(pred_img, ref_img)

        # 2. Loss_ID: ArcFace Identity Preservation Loss
        if self.id_loss_fn is not None:
            loss_id = self.id_loss_fn(pred_img, orig_v_c=v_c)
        else:
            loss_id = torch.tensor(0.0, device=pred_img.device)

        # 3. Loss_Age: MiVOLO-style Target Age Discrepancy Loss
        loss_age = self.age_loss_fn(pred_img, target_age)

        # 4. Loss_Kinship: Siamese Kinship Preservation Loss
        # Extract embedding of generated image for kinship comparison
        if self.id_loss_fn is not None and self.id_loss_fn.id_encoder is not None:
            v_pred = self.id_loss_fn.id_encoder(pred_img)
            loss_kinship = self.kinship_loss_fn(v_pred=v_pred, v_kin=v_kin, v_f=v_f, v_m=v_m)
        else:
            loss_kinship = torch.tensor(0.0, device=pred_img.device)

        # Weighted composite loss formula from code_artifact.md:
        # Loss_Total = lambda_1 * Loss_Recon + lambda_2 * Loss_ID + lambda_3 * Loss_Age + lambda_4 * Loss_Kinship
        loss_total = (
            self.config.lambda_recon * loss_recon +
            self.config.lambda_id * loss_id +
            self.config.lambda_age * loss_age +
            self.config.lambda_kinship * loss_kinship
        )

        return {
            "loss_total": loss_total,
            "loss_recon": loss_recon,
            "loss_id": loss_id,
            "loss_age": loss_age,
            "loss_kinship": loss_kinship
        }

