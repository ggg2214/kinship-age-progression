"""
FG-NET Longitudinal Face Aging Dataset Downloader and Triplet Builder
Downloads the real FG-NET dataset (1,002 images of 82 subjects, ages 0 to 69)
and builds exact same-person longitudinal age progression pairs.
"""

import os
import sys
import re
import urllib.request
import zipfile
import json
import argparse
from typing import List, Dict, Any, Optional

# Ensure project root is in sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


FGNET_MIRROR_URL = "http://yanweifu.github.io/FG_NET_data/FGNET.zip"


def parse_fgnet_filename(filename: str) -> Optional[Dict[str, Any]]:
    """
    FG-NET naming format: [SubjectID]A[Age][Suffix].[ext]
    Examples:
      001A02.JPG -> Subject 001, Age 2
      001A18a.JPG -> Subject 001, Age 18
      045A43.JPG -> Subject 045, Age 43
    """
    base = os.path.basename(filename)
    match = re.match(r"^(\d{3})[aA](\d{2})", base)
    if match:
        subject_id = match.group(1)
        age = int(match.group(2))
        return {
            "subject_id": subject_id,
            "age": age,
            "filename": base,
            "path": filename
        }
    return None


def build_fgnet_manifest(images_dir: str, output_manifest: str) -> List[Dict[str, Any]]:
    """
    Groups FG-NET images by subject and builds longitudinal age progression pairs:
    - Input Child face: exact subject at age <= 12
    - Target Ground Truth face: exact same subject at age >= 18
    """
    all_files = [
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    records_by_subject: Dict[str, List[Dict[str, Any]]] = {}
    for f in all_files:
        info = parse_fgnet_filename(f)
        if info:
            sub = info["subject_id"]
            if sub not in records_by_subject:
                records_by_subject[sub] = []
            records_by_subject[sub].append(info)

    pairs: List[Dict[str, Any]] = []

    for sub_id, subject_images in sorted(records_by_subject.items()):
        # Sort by chronological age
        subject_images.sort(key=lambda x: x["age"])
        child_images = [img for img in subject_images if img["age"] <= 12]
        adult_images = [img for img in subject_images if img["age"] >= 18]

        # Form paired progressions
        for c in child_images:
            for a in adult_images:
                pairs.append({
                    "sample_id": f"fgnet_{sub_id}_c{c['age']:02d}_t{a['age']:02d}",
                    "subject_id": sub_id,
                    "child_path": c["path"],
                    "father_path": None,  # Longitudinal same-person pair
                    "mother_path": None,
                    "source_age": float(c["age"]),
                    "target_age": float(a["age"]),
                    "ground_truth_path": a["path"]
                })

    os.makedirs(os.path.dirname(output_manifest) or ".", exist_ok=True)
    with open(output_manifest, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)

    print(f"[OK] Parsed FG-NET: {len(all_files)} images across {len(records_by_subject)} subjects.", flush=True)
    print(f"[OK] Generated {len(pairs)} exact longitudinal pairs saved to: {output_manifest}", flush=True)
    return pairs


def setup_fgnet(dest_dir: str = "data_fgnet", output_manifest: str = "data_fgnet/manifest_fgnet.json"):
    os.makedirs(dest_dir, exist_ok=True)
    images_dir = os.path.join(dest_dir, "images")
    alt_images_dir = os.path.join(dest_dir, "FGNET", "images")

    # Check if images already extracted in either location
    target_dir = None
    if os.path.exists(images_dir) and len(os.listdir(images_dir)) >= 100:
        target_dir = images_dir
    elif os.path.exists(alt_images_dir) and len(os.listdir(alt_images_dir)) >= 100:
        target_dir = alt_images_dir

    if target_dir is None:
        zip_path = os.path.join(dest_dir, "FGNET.zip")
        if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 40_000_000:
            print(f"[INFO] Downloading FG-NET from mirror: {FGNET_MIRROR_URL}...", flush=True)
            urllib.request.urlretrieve(FGNET_MIRROR_URL, zip_path)
            print("[OK] Download finished.", flush=True)

        print(f"[INFO] Extracting {zip_path} -> {dest_dir}...", flush=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(dest_dir)
        print("[OK] Extraction complete.", flush=True)

        if os.path.exists(images_dir) and len(os.listdir(images_dir)) >= 100:
            target_dir = images_dir
        elif os.path.exists(alt_images_dir) and len(os.listdir(alt_images_dir)) >= 100:
            target_dir = alt_images_dir
        else:
            raise FileNotFoundError(f"Could not locate extracted FG-NET images in {images_dir} or {alt_images_dir}")

    build_fgnet_manifest(images_dir=target_dir, output_manifest=output_manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and setup FG-NET longitudinal aging dataset")
    parser.add_argument("--dest-dir", type=str, default="data_fgnet", help="Destination directory")
    parser.add_argument("--manifest", type=str, default="data_fgnet/manifest_fgnet.json", help="Manifest output path")
    args = parser.parse_args()

    setup_fgnet(dest_dir=args.dest_dir, output_manifest=args.manifest)
