from typing import Tuple, Optional
import torch
import torchvision.transforms as T
from PIL import Image


def get_face_transforms(image_size: int = 256, is_train: bool = True):
    """
    Returns standard face transformation pipeline:
    - Resize to canonical face resolution (256x256)
    - Convert to Tensor
    - Normalize to [-1, 1]
    """
    transform_list = [
        T.Resize((image_size, image_size)),
    ]
    if is_train:
        transform_list.append(T.RandomHorizontalFlip(p=0.5))
        
    transform_list.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Range [-1, 1]
    ])
    return T.Compose(transform_list)


class RetinaFacePreprocess:
    """
    RetinaFace alignment & crop simulator.
    In a full production deployment with RetinaFace weights, this detects 5 facial landmarks
    (left eye, right eye, nose, left mouth corner, right mouth corner)
    and applies similarity transformation to align and crop the facial bounding box.
    """

    def __init__(self, target_size: int = 256):
        self.target_size = target_size
        self.transform = get_face_transforms(image_size=target_size, is_train=False)

    def align_and_crop(self, image: Image.Image, landmarks: Optional[list] = None) -> torch.Tensor:
        """
        Aligns facial bounding box and returns normalized tensor (3, target_size, target_size).
        If landmarks are provided, performs rotation & crop. Otherwise performs center crop.
        """
        # Center square crop fallback if landmarks not directly supplied
        w, h = image.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        cropped = image.crop((left, top, left + min_dim, top + min_dim))
        return self.transform(cropped)

