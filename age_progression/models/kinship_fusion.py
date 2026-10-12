from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class KinshipCrossAttentionFusion(nn.Module):
    """
    Kinship Feature Fusion Module based on Cross-Attention Mechanism.
    
    Maps Child facial features (Query) to Parent facial features (Key/Value).
    Enables selective genetic feature inheritance (e.g., father's jaw, mother's eyes)
    yielding the synthesized Kinship Vector (V_Kin).
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_regions: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_regions = num_regions
        self.region_dim = embed_dim // num_regions
        assert embed_dim % num_regions == 0, "embed_dim must be divisible by num_regions"

        # Projects global ArcFace vector (B, 512) into regional facial tokens (B, num_regions, region_dim)
        self.child_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        self.parent_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )

        # Multi-Head Cross Attention: Q = Child, K = Parents, V = Parents
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feed Forward Network for genetic feature refinement
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout)
        )

        # Final projection to Kinship Vector (V_Kin)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        v_c: torch.Tensor,
        v_f: Optional[torch.Tensor] = None,
        v_m: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            v_c: Child facial embedding (B, embed_dim)
            v_f: Father facial embedding (B, embed_dim), optional
            v_m: Mother facial embedding (B, embed_dim), optional
        Returns:
            v_kin: Synthesized Kinship Vector (B, embed_dim)
            attn_weights: Cross-attention weights (B, num_heads, num_regions, num_parents)
        """
        batch_size = v_c.shape[0]

        # Handle case where both parents are absent
        if v_f is None and v_m is None:
            # Fallback: Child self-identity representation without external kinship injection
            v_kin = self.out_proj(self.child_proj(v_c))
            return v_kin, None

        # Build parent feature tokens
        parents = []
        if v_f is not None:
            parents.append(self.parent_proj(v_f).unsqueeze(1))  # (B, 1, embed_dim)
        if v_m is not None:
            parents.append(self.parent_proj(v_m).unsqueeze(1))  # (B, 1, embed_dim)

        # Concatenate available parents into key/value tokens (B, K, embed_dim)
        parent_tokens = torch.cat(parents, dim=1)

        # Child query tokens (B, 1, embed_dim)
        child_tokens = self.child_proj(v_c).unsqueeze(1)

        # Cross Attention: Child attends to parents
        attn_out, attn_weights = self.cross_attn(
            query=child_tokens,
            key=parent_tokens,
            value=parent_tokens,
            need_weights=True
        )

        # Residual connection & LayerNorm
        fused = self.norm1(child_tokens + attn_out)

        # FFN refinement
        fused = self.norm2(fused + self.ffn(fused))

        # Flatten & final projection -> V_Kin (B, embed_dim)
        v_kin = self.out_proj(fused.squeeze(1))

        return v_kin, attn_weights

