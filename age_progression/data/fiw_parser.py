"""
Families In the Wild (FIW) Kinship Dataset Parser
Parses the large-scale FIW kinship benchmark into structured Family Triplet manifests.
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Optional

# Ensure project root is in sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def parse_fiw_families(fiw_root: str, output_manifest: str = "data_fiw/manifest_fiw.json") -> List[Dict[str, Any]]:
    """
    Parses Families In the Wild directory tree.
    Standard FIW directory hierarchy:
      fiw_root/
        F0001/
          MID1/ (e.g. Father photos)
          MID2/ (e.g. Mother photos)
          MID3/ (e.g. Child photos)
    """
    triplets = []
    print(f"[INFO] Scanning FIW directory: {fiw_root}...", flush=True)

    if not os.path.exists(fiw_root):
        print(f"[WARNING] FIW directory does not exist: {fiw_root}", flush=True)
        return []

    family_dirs = [
        os.path.join(fiw_root, d) for d in os.listdir(fiw_root)
        if os.path.isdir(os.path.join(fiw_root, d)) and d.upper().startswith("F")
    ]

    for fam_dir in family_dirs:
        fam_id = os.path.basename(fam_dir)
        member_dirs = [
            os.path.join(fam_dir, m) for m in os.listdir(fam_dir)
            if os.path.isdir(os.path.join(fam_dir, m)) and m.upper().startswith("MID")
        ]

        if len(member_dirs) < 2:
            continue

        # Get photos per member
        member_photos = {}
        for m_dir in member_dirs:
            photos = [
                os.path.join(m_dir, f) for f in os.listdir(m_dir)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ]
            if photos:
                member_photos[os.path.basename(m_dir)] = photos

        # Form triplet combinations
        mids = list(member_photos.keys())
        if len(mids) >= 3:
            # MID1: Father, MID2: Mother, MID3: Child
            c_photos = member_photos[mids[2]]
            f_photos = member_photos[mids[0]]
            m_photos = member_photos[mids[1]]
            triplets.append({
                "family_id": fam_id,
                "child_path": c_photos[0],
                "father_path": f_photos[0],
                "mother_path": m_photos[0],
                "source_age": 7.0,
                "target_age": 30.0
            })
        elif len(mids) == 2:
            # Parent + Child
            c_photos = member_photos[mids[1]]
            p_photos = member_photos[mids[0]]
            triplets.append({
                "family_id": fam_id,
                "child_path": c_photos[0],
                "father_path": p_photos[0],
                "mother_path": None,
                "source_age": 7.0,
                "target_age": 30.0
            })

    os.makedirs(os.path.dirname(output_manifest) or ".", exist_ok=True)
    with open(output_manifest, "w", encoding="utf-8") as f:
        json.dump(triplets, f, indent=2, ensure_ascii=False)

    print(f"[OK] Extracted {len(triplets)} family triplets from FIW -> {output_manifest}", flush=True)
    return triplets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FIW Kinship Dataset Parser")
    parser.add_argument("--fiw-dir", type=str, required=True, help="Path to Families In the Wild root")
    parser.add_argument("--manifest", type=str, default="data_fiw/manifest_fiw.json", help="Manifest output path")
    args = parser.parse_args()

    parse_fiw_families(args.fiw_dir, args.manifest)
