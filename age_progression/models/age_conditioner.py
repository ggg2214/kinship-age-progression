import math
from typing import Optional
import torch
import torch.nn as nn


class SinusoidalAgeEmbedding(nn.Module):
    """Sinusoidal positional encoding for continuous or discrete ages."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, age: torch.Tensor) -> torch.Tensor:
        """
        Args:
            age: Tensor of shape (B,) or (B, 1) with age values (e.g. 5, 25, 60)
        Returns:
            Tensor of shape (B, embed_dim)
        """
        if age.dim() == 1:
            age = age.unsqueeze(-1)  # (B, 1)

        half_dim = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32, device=age.device) / half_dim
        )
        args = age.float() * freqs.unsqueeze(0)  # (B, half_dim)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.embed_dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class AgeConditioner(nn.Module):
    """
    Encodes target age and optional age delta (delta_age = target_age - source_age)
    into a continuous age representation vector.
    """

    def __init__(self, age_embed_dim: int = 128, out_dim: int = 512):
        super().__init__()
        self.age_embed_dim = age_embed_dim
        self.sinusoidal = SinusoidalAgeEmbedding(age_embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(age_embed_dim * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(
        self,
        target_age: torch.Tensor,
        source_age: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            target_age: Target age tensor (B,)
            source_age: Optional source child age (B,). If None, delta is computed from target_age.
        Returns:
            v_age: Conditioning age vector (B, out_dim)
        """
        target_emb = self.sinusoidal(target_age)
        
        if source_age is not None:
            delta_age = target_age - source_age
        else:
            # Default reference base age is 5 years old (typical young missing child base age)
            delta_age = target_age - torch.full_like(target_age, 5.0)
            
        delta_emb = self.sinusoidal(delta_age)
        combined = torch.cat([target_emb, delta_emb], dim=-1)
        return self.mlp(combined)


class ConditioningFusion(nn.Module):
    """
    Fuses Identity Vector (V_C), Kinship Vector (V_Kin), and Age Vector (V_Age)
    into a unified conditioning latent representation W_cond.
    """

    def __init__(self, id_dim: int = 512, kin_dim: int = 512, age_dim: int = 512, latent_dim: int = 512):
        super().__init__()
        total_in = id_dim + kin_dim + age_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_in, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, v_c: torch.Tensor, v_kin: torch.Tensor, v_age: torch.Tensor) -> torch.Tensor:
        """
        Args:
            v_c: Child identity embedding (B, id_dim)
            v_kin: Kinship vector from Cross-Attention (B, kin_dim)
            v_age: Target age embedding (B, age_dim)
        Returns:
            w_cond: Fused latent code (B, latent_dim)
        """
        x = torch.cat([v_c, v_kin, v_age], dim=-1)
        return self.fusion(x)

