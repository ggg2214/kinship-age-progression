import os
from typing import Dict, Optional, List, Any
import torch
from torch.utils.data import Dataset
from PIL import Image

from .transforms import get_face_transforms


class KinshipAgeDataset(Dataset):
    """
    Dataset representing Child-Father-Mother triplets with chronological ages.
    Supports missing parent photos (returns None or empty tensor).
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        image_size: int = 256,
        is_train: bool = True
    ):
        """
        samples format: list of dicts:
        {
            "child_path": str,
            "father_path": Optional[str],
            "mother_path": Optional[str],
            "source_age": float,
            "target_age": float,
            "ground_truth_path": Optional[str]
        }
        """
        self.samples = samples
        self.image_size = image_size
        self.transforms = get_face_transforms(image_size=image_size, is_train=is_train)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Optional[str], fallback_zero: bool = False) -> Optional[torch.Tensor]:
        if path is None:
            return torch.zeros(3, self.image_size, self.image_size) if fallback_zero else None
        try:
            if not os.path.exists(path):
                return torch.zeros(3, self.image_size, self.image_size) if fallback_zero else None
            img = Image.open(path).convert("RGB")
            return self.transforms(img)
        except Exception:
            return torch.zeros(3, self.image_size, self.image_size) if fallback_zero else None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        child_img = self._load_image(item.get("child_path"), fallback_zero=True)
        father_img = self._load_image(item.get("father_path"))
        mother_img = self._load_image(item.get("mother_path"))
        gt_img = self._load_image(item.get("ground_truth_path"))

        return {
            "child_img": child_img,
            "father_img": father_img,
            "mother_img": mother_img,
            "ground_truth_img": gt_img,
            "source_age": torch.tensor(item.get("source_age", 5.0), dtype=torch.float32),
            "target_age": torch.tensor(item.get("target_age", 25.0), dtype=torch.float32)
        }


def collate_kinship_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collator handling optional parent and ground truth images."""
    child_imgs = [item["child_img"] for item in batch]
    child_tensor = torch.stack(child_imgs) if all(c is not None for c in child_imgs) else None

    father_imgs = [item["father_img"] for item in batch]
    father_tensor = torch.stack(father_imgs) if all(f is not None for f in father_imgs) else None

    mother_imgs = [item["mother_img"] for item in batch]
    mother_tensor = torch.stack(mother_imgs) if all(m is not None for m in mother_imgs) else None

    gt_imgs = [item.get("ground_truth_img") for item in batch]
    gt_tensor = torch.stack(gt_imgs) if all(g is not None for g in gt_imgs) else None

    source_ages = torch.stack([item["source_age"] for item in batch])
    target_ages = torch.stack([item["target_age"] for item in batch])

    return {
        "child_img": child_tensor,
        "father_img": father_tensor,
        "mother_img": mother_tensor,
        "ground_truth_img": gt_tensor,
        "source_age": source_ages,
        "target_age": target_ages
    }


class MockKinshipAgeDataset(Dataset):
    """
    Synthetic dataset for unit tests, pipeline smoke tests, and dry runs.
    Generates synthetic facial tensors with realistic dimensions.
    """

    def __init__(self, num_samples: int = 16, image_size: int = 256):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Generate random normalized face tensors in [-1, 1]
        child_img = torch.randn(3, self.image_size, self.image_size).clamp(-1, 1)
        father_img = torch.randn(3, self.image_size, self.image_size).clamp(-1, 1)
        mother_img = torch.randn(3, self.image_size, self.image_size).clamp(-1, 1)
        
        source_age = torch.tensor(float(5 + (idx % 5)), dtype=torch.float32)
        target_age = torch.tensor(float(20 + (idx % 50)), dtype=torch.float32)

        return {
            "child_img": child_img,
            "father_img": father_img,
            "mother_img": mother_img,
            "source_age": source_age,
            "target_age": target_age
        }

