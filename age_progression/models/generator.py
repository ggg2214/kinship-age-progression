import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaINBlock(nn.Module):
    """
    Adaptive Instance Normalization (AdaIN) Residual Block.
    Modulates intermediate feature statistics using conditioning latent code (W_cond),
    allowing continuous age aging and genetic facial alterations.
    """

    def __init__(self, channels: int, latent_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.inst_norm1 = nn.InstanceNorm2d(channels, affine=False)
        self.inst_norm2 = nn.InstanceNorm2d(channels, affine=False)
        self.act = nn.LeakyReLU(0.2, inplace=True)

        # Affine projections for scale (gamma) and bias (beta) from latent vector
        self.style_mod1 = nn.Linear(latent_dim, channels * 2)
        self.style_mod2 = nn.Linear(latent_dim, channels * 2)

    def forward(self, x: torch.Tensor, w_cond: torch.Tensor) -> torch.Tensor:
        residual = x

        # Layer 1
        h = self.conv1(x)
        h = self.inst_norm1(h)
        style1 = self.style_mod1(w_cond).unsqueeze(-1).unsqueeze(-1)
        gamma1, beta1 = style1.chunk(2, dim=1)
        h = h * (1.0 + gamma1) + beta1
        h = self.act(h)

        # Layer 2
        h = self.conv2(h)
        h = self.inst_norm2(h)
        style2 = self.style_mod2(w_cond).unsqueeze(-1).unsqueeze(-1)
        gamma2, beta2 = style2.chunk(2, dim=1)
        h = h * (1.0 + gamma2) + beta2

        return self.act(residual + h)


class AgeProgressionGenerator(nn.Module):
    """
    Age Progression Image Generator / Decoder.
    Combines child facial identity from input image with conditional latent vector
    (Child Identity + Kinship Vector + Target Age Vector) using AdaIN residual blocks.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        latent_dim: int = 512
    ):
        super().__init__()
        self.base_channels = base_channels

        # --- Content Encoder ---
        # 256x256 -> 128x128
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 128x128 -> 64x64
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 64x64 -> 32x32
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 32x32 -> 16x16
        self.enc4 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # --- Bottleneck Style / AdaIN Residual Blocks (16x16) ---
        self.res_block1 = AdaINBlock(base_channels * 8, latent_dim)
        self.res_block2 = AdaINBlock(base_channels * 8, latent_dim)
        self.res_block3 = AdaINBlock(base_channels * 8, latent_dim)
        self.res_block4 = AdaINBlock(base_channels * 8, latent_dim)

        # --- Decoder with Skip Connections ---
        # 16x16 -> 32x32
        self.upconv1 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1)
        self.dec_norm1 = nn.InstanceNorm2d(base_channels * 4)
        self.dec_style1 = AdaINBlock(base_channels * 4, latent_dim)

        # 32x32 -> 64x64
        self.upconv2 = nn.ConvTranspose2d(base_channels * 4 + base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.dec_norm2 = nn.InstanceNorm2d(base_channels * 2)
        self.dec_style2 = AdaINBlock(base_channels * 2, latent_dim)

        # 64x64 -> 128x128
        self.upconv3 = nn.ConvTranspose2d(base_channels * 2 + base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        self.dec_norm3 = nn.InstanceNorm2d(base_channels)
        self.dec_style3 = AdaINBlock(base_channels, latent_dim)

        # 128x128 -> 256x256
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_channels + base_channels, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, out_channels, kernel_size=7, padding=3),
            nn.Tanh()
        )

    def forward(self, child_img: torch.Tensor, w_cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            child_img: Source child face image (B, 3, 256, 256) in [-1, 1]
            w_cond: Fused conditioning vector (B, latent_dim)
        Returns:
            aged_img: Synthesized aged face image (B, 3, 256, 256) in [-1, 1]
        """
        # Encoder
        e1 = self.enc1(child_img)  # (B, 64, 256, 256)
        e2 = self.enc2(e1)         # (B, 128, 128, 128)
        e3 = self.enc3(e2)         # (B, 256, 64, 64)
        e4 = self.enc4(e3)         # (B, 512, 32, 32)
        
        # Bottleneck AdaIN blocks
        b = self.res_block1(e4, w_cond)
        b = self.res_block2(b, w_cond)
        b = self.res_block3(b, w_cond)
        b = self.res_block4(b, w_cond)

        # Decoder with skip connections
        d1 = F.leaky_relu(self.dec_norm1(self.upconv1(b)), 0.2)
        d1 = self.dec_style1(d1, w_cond)

        d2_in = torch.cat([d1, e3], dim=1)
        d2 = F.leaky_relu(self.dec_norm2(self.upconv2(d2_in)), 0.2)
        d2 = self.dec_style2(d2, w_cond)

        d3_in = torch.cat([d2, e2], dim=1)
        d3 = F.leaky_relu(self.dec_norm3(self.upconv3(d3_in)), 0.2)
        d3 = self.dec_style3(d3, w_cond)

        out_in = torch.cat([d3, e1], dim=1)
        aged_img = self.out_conv(out_in)

        return aged_img

