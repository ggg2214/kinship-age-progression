"""
Face Restoration and Deblurring Preprocessor
Enhances degraded, blurry, or low-resolution historical child face photos
prior to age progression synthesis.
"""

import os
from typing import Optional, Union
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import torch


class FaceEnhancer:
    """
    Lightweight, high-performance facial enhancement pipeline.
    Applies unsharp masking, contrast normalization, edge enhancement,
    and bilateral skin smoothing to recover facial landmarks from blurry inputs.
    """

    def __init__(self, sharpen_factor: float = 1.4, contrast_factor: float = 1.15):
        self.sharpen_factor = sharpen_factor
        self.contrast_factor = contrast_factor

    def enhance(self, img_input: Union[str, Image.Image, torch.Tensor]) -> Image.Image:
        """
        Enhances input face image.
        Returns:
            enhanced_pil: PIL.Image
        """
        # 1. Convert to PIL Image
        if isinstance(img_input, str):
            pil_img = Image.open(img_input).convert("RGB")
        elif isinstance(img_input, Image.Image):
            pil_img = img_input.convert("RGB")
        elif isinstance(img_input, torch.Tensor):
            t = img_input.detach().cpu().squeeze()
            if t.dim() == 3 and t.shape[0] == 3:
                t = (t * 0.5 + 0.5).clamp(0, 1)
                arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                pil_img = Image.fromarray(arr)
            else:
                raise ValueError("Expected 3-channel tensor (3, H, W)")
        else:
            raise TypeError("Unsupported image input type")

        w, h = pil_img.size

        # 2. Resize to high-clarity intermediate size if low-res (< 256)
        if w < 256 or h < 256:
            pil_img = pil_img.resize((max(w, 256), max(h, 256)), Image.Resampling.LANCZOS)

        # 3. Denoise skin with subtle median filter
        denoised = pil_img.filter(ImageFilter.MedianFilter(size=3))

        # 4. Detail preservation / Unsharp Masking
        # Blend original with high-pass filtered version
        unsharp = denoised.filter(
            ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)
        )

        # 5. Contrast and Color normalization
        contrast_enhancer = ImageEnhance.Contrast(unsharp)
        enhanced = contrast_enhancer.enhance(self.contrast_factor)

        # 6. Sharpness boost for eyes, nose, and mouth contours
        sharpness_enhancer = ImageEnhance.Sharpness(enhanced)
        final_img = sharpness_enhancer.enhance(self.sharpen_factor)

        return final_img


def restore_face_cli():
    import argparse
    parser = argparse.ArgumentParser(description="Enhance low-quality face image")
    parser.add_argument("--input", type=str, required=True, help="Input blurry face path")
    parser.add_argument("--output", type=str, default="enhanced_face.png", help="Output path")
    args = parser.parse_args()

    enhancer = FaceEnhancer()
    res = enhancer.enhance(args.input)
    res.save(args.output)
    print(f"[OK] Enhanced image saved to: {args.output}")


if __name__ == "__main__":
    restore_face_cli()
