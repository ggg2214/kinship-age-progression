from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseKinshipVerifier(nn.Module):
    """
    Siamese Kinship Verification Network.
    Evaluates whether two facial embeddings share biological kinship (FIW benchmark style).
    Takes difference and Hadamard product of features to estimate kinship probability.
    """

    def __init__(self, embed_dim: int = 512):
        super().__init__()
        # Input: |v1 - v2| and (v1 * v2) -> 2 * embed_dim
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """Outputs kinship logit between two embeddings."""
        diff = torch.abs(v1 - v2)
        prod = v1 * v2
        feat = torch.cat([diff, prod], dim=-1)
        return self.regress_logit(feat)

    def regress_logit(self, feat: torch.Tensor) -> torch.Tensor:
        return self.fc(feat).squeeze(-1)


class KinshipLoss(nn.Module):
    """
    Kinship Preservation Loss (L_Kinship).
    
    Enforces genetic resemblance between synthesized face and biological parents,
    preventing arbitrary aging and following the genetic trajectory of the parents.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        verifier: Optional[nn.Module] = None,
        use_cosine_constraint: bool = True
    ):
        super().__init__()
        self.verifier = verifier or SiameseKinshipVerifier(embed_dim=embed_dim)
        self.use_cosine_constraint = use_cosine_constraint

    def forward(
        self,
        v_pred: torch.Tensor,
        v_kin: Optional[torch.Tensor] = None,
        v_f: Optional[torch.Tensor] = None,
        v_m: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            v_pred: Identity embedding of generated face (B, embed_dim)
            v_kin: Cross-attention synthesized kinship target vector (B, embed_dim)
            v_f: Father embedding (B, embed_dim), optional
            v_m: Mother embedding (B, embed_dim), optional
        Returns:
            Scalar kinship loss
        """
        # If no kinship information or parents provided, return 0
        if v_kin is None and v_f is None and v_m is None:
            return torch.tensor(0.0, device=v_pred.device, requires_grad=True)

        total_kin_loss = torch.tensor(0.0, device=v_pred.device)
        terms = 0

        # 1. Alignment with synthesized Kinship Vector (V_Kin)
        if v_kin is not None:
            kin_cos = F.cosine_similarity(v_pred, v_kin, dim=-1)
            loss_vkin = 1.0 - kin_cos.mean()
            total_kin_loss = total_kin_loss + loss_vkin
            terms += 1

        # 2. Siamese biological relationship scoring with available parents
        parent_embeddings = [p for p in (v_f, v_m) if p is not None]
        for v_parent in parent_embeddings:
            # Kinship logit should indicate positive biological relation (target = 1)
            logit = self.verifier(v_pred, v_parent)
            bce_loss = F.binary_cross_entropy_with_logits(
                logit,
                torch.ones_like(logit)
            )
            total_kin_loss = total_kin_loss + bce_loss
            terms += 1

        return total_kin_loss / max(1, terms)

