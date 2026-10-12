from typing import Dict, Optional, Tuple, Any
import torch
import torch.nn as nn

from ..config import AgeProgressionConfig
from .arcface_encoder import ArcFaceEncoder
from .kinship_fusion import KinshipCrossAttentionFusion
from .age_conditioner import AgeConditioner, ConditioningFusion
from .generator import AgeProgressionGenerator


class AgeProgressionModel(nn.Module):
    """
    End-to-End Age Progression System with Kinship Guided Generation.
    
    Synthesizes aged face images preserving biometric identity (V_C) and
    inheriting genetic traits from biological parents (V_F, V_M) via Cross-Attention.
    """

    def __init__(self, config: Optional[AgeProgressionConfig] = None):
        super().__init__()
        self.config = config or AgeProgressionConfig()

        # 1. Identity & Facial Feature Extractor (ArcFace)
        self.id_encoder = ArcFaceEncoder(
            embed_dim=self.config.id_embed_dim,
            pretrained=False
        )

        # 2. Kinship Cross-Attention Fusion Module
        self.kinship_fusion = KinshipCrossAttentionFusion(
            embed_dim=self.config.id_embed_dim,
            num_heads=self.config.num_fusion_heads,
            dropout=self.config.dropout
        )

        # 3. Age Conditioner
        self.age_conditioner = AgeConditioner(
            age_embed_dim=self.config.age_embed_dim,
            out_dim=self.config.latent_dim
        )

        # 4. Conditioning Fusion Layer (V_C + V_Kin + V_Age -> W_cond)
        self.cond_fusion = ConditioningFusion(
            id_dim=self.config.id_embed_dim,
            kin_dim=self.config.id_embed_dim,
            age_dim=self.config.latent_dim,
            latent_dim=self.config.latent_dim
        )

        # 5. Conditional Age Progression Image Generator
        self.generator = AgeProgressionGenerator(
            in_channels=self.config.in_channels,
            out_channels=self.config.out_channels,
            base_channels=self.config.generator_base_channels,
            latent_dim=self.config.latent_dim
        )

    def forward(
        self,
        child_img: torch.Tensor,
        target_age: torch.Tensor,
        father_img: Optional[torch.Tensor] = None,
        mother_img: Optional[torch.Tensor] = None,
        source_age: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Forward pass of Age Progression Pipeline.

        Args:
            child_img: Source child face tensor (B, 3, H, W)
            target_age: Target age tensor (B,)
            father_img: Optional father face tensor (B, 3, H, W)
            mother_img: Optional mother face tensor (B, 3, H, W)
            source_age: Optional source child age tensor (B,)

        Returns:
            dict containing:
                'pred_img': Generated target age face image (B, 3, H, W)
                'v_c': Child identity embedding (B, 512)
                'v_f': Father identity embedding (B, 512) or None
                'v_m': Mother identity embedding (B, 512) or None
                'v_kin': Fused Kinship vector (B, 512)
                'v_age': Target age embedding (B, 512)
                'w_cond': Combined conditioning code (B, 512)
                'attn_weights': Cross-attention kinship weights
        """
        # Step 1: Feature Extraction (ArcFace)
        v_c = self.id_encoder(child_img)
        v_f = self.id_encoder(father_img) if father_img is not None else None
        v_m = self.id_encoder(mother_img) if mother_img is not None else None

        # Step 2: Kinship Cross-Attention Fusion (Query: V_C, Key/Val: V_F, V_M)
        v_kin, attn_weights = self.kinship_fusion(v_c=v_c, v_f=v_f, v_m=v_m)

        # Step 3: Age Conditioning
        v_age = self.age_conditioner(target_age=target_age, source_age=source_age)

        # Step 4: Latent Code Synthesis (W_cond)
        w_cond = self.cond_fusion(v_c=v_c, v_kin=v_kin, v_age=v_age)

        # Step 5: Image Synthesis via Modulated Generator
        pred_img = self.generator(child_img=child_img, w_cond=w_cond)

        return {
            "pred_img": pred_img,
            "v_c": v_c,
            "v_f": v_f,
            "v_m": v_m,
            "v_kin": v_kin,
            "v_age": v_age,
            "w_cond": w_cond,
            "attn_weights": attn_weights
        }

