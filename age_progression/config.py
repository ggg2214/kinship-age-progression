from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class AgeProgressionConfig:
    """Configuration hyperparameters for Age Progression System."""

    # Image specifications
    image_size: int = 256
    in_channels: int = 3
    out_channels: int = 3

    # Feature dimensions
    id_embed_dim: int = 512       # ArcFace standard embedding size
    age_embed_dim: int = 128      # Dimension of learned/fourier age embedding
    latent_dim: int = 512         # Unified conditioned latent representation

    # Architecture parameters
    generator_base_channels: int = 64
    num_fusion_heads: int = 8
    dropout: float = 0.1

    # Loss weights according to code_artifact.md:
    # Loss_Total = lambda_1 * Loss_Recon + lambda_2 * Loss_ID + lambda_3 * Loss_Age + lambda_4 * Loss_Kinship
    lambda_recon: float = 1.0     # lambda_1: Reconstruction loss weight
    lambda_id: float = 2.0        # lambda_2: ArcFace identity preservation weight
    lambda_age: float = 1.5       # lambda_3: Target age estimation accuracy weight
    lambda_kinship: float = 1.0   # lambda_4: Kinship resemblance verification weight

    # Training & Runtime
    device: str = "cpu"           # 'cuda' or 'cpu'
    batch_size: int = 4
    learning_rate: float = 2e-4
    min_age: int = 0
    max_age: int = 100

