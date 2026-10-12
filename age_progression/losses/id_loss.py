from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityLoss(nn.Module):
    """
    ArcFace-based Identity Preservation Loss (L_ID).
    
    Enforces that synthesized aged face maintains high biometric facial similarity
    with the original missing child's features, preventing identity drifting.
    """

    def __init__(self, id_encoder: Optional[nn.Module] = None):
        super().__init__()
        self.id_encoder = id_encoder

    def forward(
        self,
        pred_img: torch.Tensor,
        orig_img: Optional[torch.Tensor] = None,
        orig_v_c: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pred_img: Synthesized aged face image (B, 3, H, W)
            orig_img: Original child face image (B, 3, H, W), optional if orig_v_c provided
            orig_v_c: Pre-extracted child identity embedding (B, 512), optional
        Returns:
            Scalar identity cosine distance loss
        """
        if self.id_encoder is not None:
            v_pred = self.id_encoder(pred_img)
            if orig_v_c is None:
                assert orig_img is not None, "Either orig_img or orig_v_c must be provided"
                with torch.no_grad():
                    orig_v_c = self.id_encoder(orig_img)
        else:
            raise ValueError("id_encoder must be provided to IdentityLoss")

        # Cosine distance: 1 - cos(v_pred, orig_v_c)
        cos_sim = F.cosine_similarity(v_pred, orig_v_c, dim=-1)
        loss_id = 1.0 - cos_sim.mean()
        return loss_id

