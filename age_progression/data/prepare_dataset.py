"""
Dataset Preparation & Ingestion Utility for Kinship Age Progression
Formats family triplet datasets (UTKFace, K-Face, FIW, or local directories)
into structured JSON manifests for KinshipAgeDataset.
"""

import os
import json
import argparse
from typing import List, Dict, Any, Optional
from PIL import Image, ImageDraw


def create_sample_mock_data(output_dir: str = "sample_data", num_families: int = 4):
    """
    Creates a sample mock family directory with synthetic face images
    for demonstration and testing without requiring external downloads.
    """
    os.makedirs(output_dir, exist_ok=True)
    manifest: List[Dict[str, Any]] = []

    colors = [
        ("child", (240, 190, 160)),
        ("father", (190, 150, 120)),
        ("mother", (220, 175, 145)),
    ]

    for fam_id in range(1, num_families + 1):
        fam_dir = os.path.join(output_dir, f"family_{fam_id:03d}")
        os.makedirs(fam_dir, exist_ok=True)

        child_path = os.path.join(fam_dir, "child_age06.jpg")
        father_path = os.path.join(fam_dir, "father_age38.jpg")
        mother_path = os.path.join(fam_dir, "mother_age35.jpg")

        # Generate simple stylized synthetic face images
        for path, (label, base_color) in zip([child_path, father_path, mother_path], colors):
            img = Image.new("RGB", (256, 256), color=base_color)
            draw = ImageDraw.Draw(img)
            # Draw facial features (eyes, mouth)
            draw.ellipse([60, 80, 95, 115], fill=(50, 50, 60))    # Left eye
            draw.ellipse([160, 80, 195, 115], fill=(50, 50, 60))  # Right eye
            draw.polygon([(128, 110), (115, 150), (140, 150)], fill=(120, 80, 70)) # Nose
            draw.arc([90, 160, 166, 190], start=0, end=180, fill=(150, 40, 40), width=4) # Mouth
            draw.text((10, 10), f"Fam#{fam_id} {label}", fill=(20, 20, 20))
            img.save(path, quality=92)

        manifest.append({
            "family_id": f"family_{fam_id:03d}",
            "child_path": child_path,
            "father_path": father_path,
            "mother_path": mother_path,
            "source_age": 6.0,
            "target_age": 30.0
        })

    manifest_path = os.path.join(output_dir, "dataset_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[OK] Generated {num_families} synthetic family folders in: {output_dir}")
    print(f"[OK] Manifest saved to: {manifest_path}")
    return manifest_path


def parse_directory(data_dir: str, output_manifest: str) -> str:
    """
    Scans a directory expecting subfolders per family or named files.
    """
    manifest = []
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    for root, dirs, files in os.walk(data_dir):
        image_files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not image_files:
            continue

        child_file = next((f for f in image_files if "child" in f.lower()), None)
        father_file = next((f for f in image_files if "father" in f.lower()), None)
        mother_file = next((f for f in image_files if "mother" in f.lower()), None)

        if child_file:
            manifest.append({
                "family_id": os.path.basename(root),
                "child_path": os.path.join(root, child_file),
                "father_path": os.path.join(root, father_file) if father_file else None,
                "mother_path": os.path.join(root, mother_file) if mother_file else None,
                "source_age": 5.0,
                "target_age": 30.0
            })

    with open(output_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[OK] Parsed {len(manifest)} triplets from {data_dir} -> {output_manifest}")
    return output_manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Kinship Age Progression Dataset")
    parser.add_argument("--create-mock", action="store_true", help="Generate synthetic test family dataset")
    parser.add_argument("--data-dir", type=str, default="sample_data", help="Target or source directory")
    parser.add_argument("--manifest", type=str, default="dataset_manifest.json", help="Manifest output path")
    args = parser.parse_args()

    if args.create_mock:
        create_sample_mock_data(output_dir=args.data_dir)
    else:
        parse_directory(args.data_dir, args.manifest)

