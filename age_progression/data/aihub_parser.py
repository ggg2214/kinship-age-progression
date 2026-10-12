"""
AI Hub (aihub.or.kr) Korean Facial & Kinship Dataset Parser
Supports ingesting:
1. 한국인 가족 안면 데이터 (Korean Family Facial Dataset)
2. 한국인 연령대별 안면 이미지 데이터 (Korean Age-bracket Face Dataset)
3. K-Face 한국인 안면 이미지 데이터 (K-Face Identity Dataset)
"""

import os
import sys
import json
import glob
import argparse
from typing import List, Dict, Any, Optional
from PIL import Image

# Ensure project root is in sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def print_aihub_download_guide():
    """Prints command-line instructions for acquiring datasets via AI Hub."""
    guide = """
================================================================================
 AI Hub (aihub.or.kr) 한국인 안면 & 가족 데이터셋 다운로드 가이드
================================================================================
AI Hub 데이터는 공공 데이터 규정상 개인 인증(휴대폰/I-PIN) 및 활용 신청 승인이 필요합니다.
아래 절차를 따라 공식 aihubshell CLI를 통해 다운로드하실 수 있습니다:

1. AI Hub 회원가입 및 데이터 신청 (웹 브라우저)
   - 한국인 가족 안면 데이터: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=562
   - 한국인 연령대별 안면 이미지: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=560
   - K-Face 한국인 안면 이미지: https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=83

2. aihubshell 다운로드 및 인증키 설정 (Windows PowerShell)
   curl -O https://api.aihub.or.kr/api/aihubshell/download.do?fileNm=aihubshell_windows.zip
   Expand-Archive aihubshell_windows.zip -DestinationPath aihubshell
   .\\aihubshell\\aihubshell.exe -mode l -id <YOUR_AIHUB_ID> -pw <YOUR_AIHUB_PW>

3. 데이터 다운로드 실행 예시:
   .\\aihubshell\\aihubshell.exe -mode d -datasetkey <DATASET_KEY> -filekey <FILE_KEY>

4. 다운로드 완료 후 본 파서를 실행하여 Triplet 매니페스트로 자동 변환:
   python age_progression/data/aihub_parser.py --source-dir "C:/path/to/aihub_download" --type family
================================================================================
"""
    print(guide)


def parse_aihub_family(source_dir: str, output_manifest: str) -> List[Dict[str, Any]]:
    """
    Parses AI Hub Korean Family Facial Dataset into KinshipAgeDataset format.
    AI Hub Family structure typically groups members by family ID:
    e.g. {Family_ID}/{Person_Type}_{Person_ID} or JSON annotations.
    """
    triplets = []
    print(f"[INFO] Scanning AI Hub family directory: {source_dir}...")

    # Traverse directory
    for root, dirs, files in os.walk(source_dir):
        image_files = [os.path.join(root, f) for f in files if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        json_files = [os.path.join(root, f) for f in files if f.lower().endswith(".json")]

        # Case A: JSON Label with relationship metadata
        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                # Check for family members
                members = meta.get("family_members", meta.get("members", []))
                if isinstance(members, dict):
                    child_path = members.get("child", {}).get("image_path")
                    father_path = members.get("father", {}).get("image_path")
                    mother_path = members.get("mother", {}).get("image_path")
                    if child_path and (father_path or mother_path):
                        triplets.append({
                            "family_id": meta.get("family_id", os.path.basename(root)),
                            "child_path": child_path,
                            "father_path": father_path,
                            "mother_path": mother_path,
                            "source_age": float(members.get("child", {}).get("age", 6.0)),
                            "target_age": 30.0
                        })
            except Exception:
                continue

        # Case B: Directory naming conventions (P01_Father, P02_Mother, C01_Child)
        if not triplets and image_files:
            child = next((f for f in image_files if any(k in f.lower() for k in ["child", "c01", "son", "daughter", "자녀"])), None)
            father = next((f for f in image_files if any(k in f.lower() for k in ["father", "p01", "dad", "부"])), None)
            mother = next((f for f in image_files if any(k in f.lower() for k in ["mother", "p02", "mom", "모"])), None)

            if child:
                triplets.append({
                    "family_id": os.path.basename(root),
                    "child_path": child,
                    "father_path": father,
                    "mother_path": mother,
                    "source_age": 6.0,
                    "target_age": 30.0
                })

    os.makedirs(os.path.dirname(output_manifest) or ".", exist_ok=True)
    with open(output_manifest, "w", encoding="utf-8") as f:
        json.dump(triplets, f, indent=2, ensure_ascii=False)

    print(f"[OK] Parsed {len(triplets)} AI Hub family triplets -> {output_manifest}")
    return triplets


def parse_aihub_kface(source_dir: str, output_manifest: str) -> List[Dict[str, Any]]:
    """
    Parses AI Hub K-Face dataset (High resolution Korean Face dataset across angles & lights).
    """
    records = []
    print(f"[INFO] Scanning K-Face directory: {source_dir}...")
    for root, dirs, files in os.walk(source_dir):
        image_files = [os.path.join(root, f) for f in files if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        for img in image_files:
            subject_id = os.path.basename(os.path.dirname(root))
            records.append({
                "subject_id": subject_id,
                "image_path": img
            })

    manifest_path = output_manifest
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"[OK] Parsed {len(records)} K-Face images -> {manifest_path}")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Hub Korean Facial & Kinship Dataset Parser")
    parser.add_argument("--source-dir", type=str, default=None, help="Root path to downloaded AI Hub dataset")
    parser.add_argument("--output-manifest", type=str, default="data_aihub/manifest.json", help="Output JSON manifest path")
    parser.add_argument("--type", type=str, choices=["family", "kface", "guide"], default="guide", help="Parser mode or guide")
    args = parser.parse_args()

    if args.type == "guide" or args.source_dir is None:
        print_aihub_download_guide()
    elif args.type == "family":
        parse_aihub_family(args.source_dir, args.output_manifest)
    elif args.type == "kface":
        parse_aihub_kface(args.source_dir, args.output_manifest)

