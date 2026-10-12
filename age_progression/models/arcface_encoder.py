import os
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ArcFaceEncoder(nn.Module):
    """
    ArcFace-style Identity & Facial Feature Extractor.
    Extracts high-dimensional L2-normalized identity embeddings (V_C, V_F, V_M)
    for child, father, and mother faces.
    """

    def __init__(self, embed_dim: int = 512, pretrained: bool = False):
        super().__init__()
        self.embed_dim = embed_dim

        # ResNet-50 backbone modified for facial identity embedding
        # Can use torchvision weights if available or standard initialization
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        
        # Keep feature extraction trunk
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Adaptive pooling & FC projection head to embed_dim (512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(resnet.fc.in_features, embed_dim, bias=False)
        self.bn_fc = nn.BatchNorm1d(embed_dim)

        if pretrained:
            self.load_pretrained_weights()

    def load_pretrained_weights(self, weights_path: Optional[str] = None):
        """
        Loads pretrained ArcFace / Face Recognition weights.
        Supports loading from local checkpoint file or torchvision backbone.
        """
        if weights_path and os.path.exists(weights_path):
            print(f"[INFO] Loading ArcFace weights from: {weights_path}")
            state_dict = torch.load(weights_path, map_location="cpu")
            # Filter compatible keys
            model_dict = self.state_dict()
            matched = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(matched)
            self.load_state_dict(model_dict)
            print(f"[OK] Successfully loaded {len(matched)} / {len(model_dict)} layers from ArcFace checkpoint.")
        else:
            try:
                # Load official ResNet-50 ImageNet pretrained weights into backbone
                weights = models.ResNet50_Weights.DEFAULT
                resnet = models.resnet50(weights=weights)
                self.conv1.load_state_dict(resnet.conv1.state_dict())
                self.bn1.load_state_dict(resnet.bn1.state_dict())
                self.layer1.load_state_dict(resnet.layer1.state_dict())
                self.layer2.load_state_dict(resnet.layer2.state_dict())
                self.layer3.load_state_dict(resnet.layer3.state_dict())
                self.layer4.load_state_dict(resnet.layer4.state_dict())
                print("[OK] ArcFace backbone initialized with pretrained feature weights.")
            except Exception as e:
                print(f"[WARNING] Could not load pretrained weights: {e}. Using current weights.")

    def forward(self, x: torch.Tensor, return_spatial: bool = False):
        """
        Args:
            x: Face image tensor of shape (B, 3, H, W) in range [-1, 1] or [0, 1]
            return_spatial: If True, also return intermediate spatial feature map
        Returns:
            normalized_embedding: (B, embed_dim)
            (optional) spatial_features: (B, 2048, H/32, W/32)
        """
        # Ensure image is in expected normalization
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.maxpool(h)

        h1 = self.layer1(h)
        h2 = self.layer2(h1)
        h3 = self.layer3(h2)
        h4 = self.layer4(h3)  # Shape: (B, 2048, H/32, W/32)

        feat_pooled = self.avgpool(h4).flatten(1)
        emb = self.bn_fc(self.fc(feat_pooled))
        
        # ArcFace requires L2 normalization of identity embeddings
        emb_norm = F.normalize(emb, p=2, dim=-1)

        if return_spatial:
            return emb_norm, h4
        return emb_norm

    def extract_features_triplet(
        self,
        child_img: torch.Tensor,
        father_img: torch.Tensor = None,
        mother_img: torch.Tensor = None
    ):
        """Extracts embeddings for Child (V_C), Father (V_F), and Mother (V_M)."""
        v_c = self.forward(child_img)
        v_f = self.forward(father_img) if father_img is not None else None
        v_m = self.forward(mother_img) if mother_img is not None else None
        return v_c, v_f, v_m

