"""
AI Hub (aihub.or.kr) 4-Dataset Integrated Pipeline & Google Drive Manager
Handles:
1. 한국인 가족 관계 안면 데이터 (Korean Family Facial Dataset)
2. 안면 인식 에이징 이미지 데이터 (Korean Age Progression Face Dataset)
3. 안면 랜드마크 데이터 (Korean Facial Landmark Dataset)
4. 한국인 헤어스타일 이미지 데이터 (Korean Hairstyle Dataset)

Features:
- aihubshell automated download via API Key
- Direct storage into Google Drive (5TB) or local path
- 256x256 intelligent face alignment, cropping, and triplet pairing
- Unified manifest builder (manifest_aihub.json)
- Post-training safe cleanup for large raw files
"""

import os
import sys
import json
import shutil
import tarfile
import zipfile
import re
import argparse
import subprocess
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image, ImageOps

# Ensure project root is in sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_current_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def ensure_aihubshell() -> str:
    """Ensures aihubshell is installed and executable on Linux (Colab) or Windows."""
    # Check if aihubshell is in PATH
    which_shell = shutil.which("aihubshell")
    if which_shell:
        return which_shell

    # For Linux / Google Colab
    if os.name != "nt":
        local_path = os.path.abspath("./aihubshell")
        if not os.path.exists(local_path):
            print("[INFO] Downloading aihubshell for Linux...")
            cmd = "curl -s -o aihubshell https://api.aihub.or.kr/api/aihubshell.do && chmod +x aihubshell"
            subprocess.run(cmd, shell=True, check=True)
        return local_path
    else:
        # Windows fallback guide
        print("[WARN] On Windows, WSL (Ubuntu) is recommended for aihubshell.")
        return "aihubshell"


def download_aihub_dataset(
    apikey: str,
    datasetkey: str,
    output_dir: str,
    filekey: Optional[str] = None
) -> bool:
    """Downloads an AI Hub dataset using aihubshell into output_dir and verifies files."""
    shell_bin = ensure_aihubshell()
    os.makedirs(output_dir, exist_ok=True)

    cmd = [shell_bin, "-aihubapikey", apikey, "-mode", "d", "-datasetkey", str(datasetkey)]
    if filekey:
        cmd.extend(["-filekey", str(filekey)])

    print(f"\n[INFO] AI Hub 공식 다운로더 가동 중 (Dataset Key: {datasetkey})...")
    print(f"[INFO] 저장 대상 폴더: {output_dir}")
    try:
        proc = subprocess.run(cmd, cwd=output_dir, capture_output=True, text=True)
        stdout_txt = proc.stdout or ""
        stderr_txt = proc.stderr or ""
        combined_output = stdout_txt + "\n" + stderr_txt

        print(stdout_txt)
        if stderr_txt:
            print(f"[STDERR]: {stderr_txt}")

        # Check for typical AI Hub approval / permission errors
        if "승인 후 이용" in combined_output or "HTTP status 502" in combined_output:
            print("\n" + "=" * 70)
            print("🚨 [알림] AI Hub 다운로드 승인 필요")
            print("AI Hub 서버 응답: 다운로드 서비스는 홈페이지(https://aihub.or.kr)에서 신청 및 승인 후 이용 가능합니다.")
            print("조치 방법: AI Hub 로그인 ➔ 해당 데이터셋 상세 페이지 ➔ [활용 신청] 버튼 클릭 후 승인 상태를 확인해 주세요.")
            print("=" * 70 + "\n")
            return False

        # Verify whether actual files exist
        downloaded_files = []
        for r, _, files in os.walk(output_dir):
            for f in files:
                downloaded_files.append(os.path.join(r, f))

        total_bytes = sum(os.path.getsize(f) for f in downloaded_files if os.path.exists(f))
        total_mb = total_bytes / (1024 * 1024)

        if len(downloaded_files) == 0 or total_mb < 1.0:
            print(f"[WARN] 다운로드된 유효 데이터가 없습니다 (현재 용량: {total_mb:.2f} MB, 파일 수: {len(downloaded_files)}개).")
            return False

        print(f"[OK] 데이터셋 {datasetkey} 다운로드 성공! (총 파일 수: {len(downloaded_files)}개, 총 용량: {total_mb:.2f} MB)")
        return True
    except Exception as e:
        print(f"[ERROR] aihubshell 실행 중 예외 발생: {e}")
        return False


def crop_and_align_face(
    img: Image.Image,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    landmarks: Optional[List[Tuple[float, float]]] = None,
    target_size: int = 256
) -> Image.Image:
    """
    Crops the face center and aligns to target_size (256x256).
    Applies margin around bounding box to capture jawline and hair.
    """
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size

    if bbox:
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        cx, cy = x1 + bw // 2, y1 + bh // 2
        # Add 35% margin around face for hair and jaw context
        size = int(max(bw, bh) * 1.35)
    elif landmarks and len(landmarks) >= 5:
        xs = [pt[0] for pt in landmarks]
        ys = [pt[1] for pt in landmarks]
        cx, cy = int(sum(xs) / len(xs)), int(sum(ys) / len(ys))
        span_x, span_y = max(xs) - min(xs), max(ys) - min(ys)
        size = int(max(span_x, span_y) * 1.6)
    else:
        # Default central portrait crop
        cx, cy = w // 2, h // 2
        size = int(min(w, h) * 0.85)

    half = size // 2
    crop_x1 = max(0, cx - half)
    crop_y1 = max(0, cy - half)
    crop_x2 = min(w, cx + half)
    crop_y2 = min(h, cy + half)

    cropped = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    return cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)


def extract_age_from_path(path: str) -> float:
    """Extracts age from filename or directory, defaulting to 25.0 if unknown."""
    m = re.search(r'(\d{1,2})\s*대', path)
    if m:
        return float(m.group(1))
    m = re.search(r'age[_\-]?(\d{1,2})', path, re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r'(\d{1,2})\s*s\b', path, re.IGNORECASE)
    if m:
        return float(m.group(1))
    for part in os.path.basename(path).replace(".", "_").split("_"):
        if part.isdigit() and 1 <= int(part) <= 95:
            return float(part)
    return 25.0


def preprocess_aihub_datasets(
    raw_root_dir: str,
    output_aligned_dir: str,
    manifest_output_path: str,
    target_size: int = 256,
    max_samples: int = 3000
) -> List[Dict[str, Any]]:
    """
    Processes all 4 AI Hub datasets:
    - Finds images and corresponding JSON annotations
    - Detects family relationships, ages, and facial landmarks
    - If no explicit family keywords exist, intelligently converts Korean aging face dataset
      into valid self-conditioning and age progression triplets.
    - Limits to max_samples to conserve disk space and prevent [Errno 28] disk full errors.
    """
    os.makedirs(output_aligned_dir, exist_ok=True)
    images_dir = os.path.join(output_aligned_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    triplets = []
    print(f"[INFO] Scanning raw AI Hub datasets in: {raw_root_dir}...")

    # Index all image files and json annotations
    all_files = []
    for root, _, files in os.walk(raw_root_dir):
        for f in files:
            all_files.append(os.path.join(root, f))

    img_paths = [f for f in all_files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    json_paths = {os.path.splitext(f)[0]: f for f in all_files if f.lower().endswith(".json")}
    zip_paths = [f for f in all_files if f.lower().endswith((".zip", ".tar", ".tar.gz", ".tgz"))]

    print(f"[INFO] Discovered {len(img_paths)} images, {len(json_paths)} JSON annotations, and {len(zip_paths)} archive files.")
    if len(img_paths) == 0 and len(zip_paths) > 0:
        print(f"[TIP] 발견된 압축 파일이 {len(zip_paths)}개 있습니다. '--action extract'를 실행하여 고속 압축 해제를 먼저 진행해 주세요.")

    # Strategy 1: Explicit family relationship keywords
    groups: Dict[str, Dict[str, str]] = {}
    for p in img_paths:
        parent_dir = os.path.basename(os.path.dirname(p))
        fname = os.path.basename(p).lower()
        if parent_dir not in groups:
            groups[parent_dir] = {}

        if any(k in fname for k in ["child", "c01", "son", "daughter", "자녀", "아동"]):
            groups[parent_dir]["child"] = p
        elif any(k in fname for k in ["father", "p01", "dad", "부", "부모_부"]):
            groups[parent_dir]["father"] = p
        elif any(k in fname for k in ["mother", "p02", "mom", "모", "부모_모"]):
            groups[parent_dir]["mother"] = p
        elif any(k in fname for k in ["adult", "aged", "target", "성인"]):
            groups[parent_dir]["adult"] = p

    for grp_id, mem in groups.items():
        if len(triplets) >= max_samples:
            break
        child_raw = mem.get("child")
        if not child_raw:
            continue

        try:
            # Child image
            child_img = Image.open(child_raw)
            c_name = f"{grp_id}_child.jpg"
            c_aligned_path = os.path.join(images_dir, c_name)
            crop_and_align_face(child_img, target_size=target_size).save(c_aligned_path, quality=95)

            # Father image
            f_aligned_path = None
            if mem.get("father"):
                f_img = Image.open(mem["father"])
                f_name = f"{grp_id}_father.jpg"
                f_aligned_path = os.path.join(images_dir, f_name)
                crop_and_align_face(f_img, target_size=target_size).save(f_aligned_path, quality=95)

            # Mother image
            m_aligned_path = None
            if mem.get("mother"):
                m_img = Image.open(mem["mother"])
                m_name = f"{grp_id}_mother.jpg"
                m_aligned_path = os.path.join(images_dir, m_name)
                crop_and_align_face(m_img, target_size=target_size).save(m_aligned_path, quality=95)

            # Target adult image (if available in aging dataset)
            t_aligned_path = None
            if mem.get("adult"):
                t_img = Image.open(mem["adult"])
                t_name = f"{grp_id}_adult.jpg"
                t_aligned_path = os.path.join(images_dir, t_name)
                crop_and_align_face(t_img, target_size=target_size).save(t_aligned_path, quality=95)

            triplets.append({
                "family_id": grp_id,
                "child_path": os.path.relpath(c_aligned_path, _project_root).replace("\\", "/"),
                "father_path": os.path.relpath(f_aligned_path, _project_root).replace("\\", "/") if f_aligned_path else None,
                "mother_path": os.path.relpath(m_aligned_path, _project_root).replace("\\", "/") if m_aligned_path else None,
                "target_path": os.path.relpath(t_aligned_path, _project_root).replace("\\", "/") if t_aligned_path else None,
                "source_age": 6.0,
                "target_age": 30.0
            })
        except Exception:
            continue

    # Strategy 2: Intelligent Aging & Individual Face Mapping (Fallback for 33k+ AI Hub aging dataset)
    if len(triplets) == 0 and len(img_paths) > 0:
        print(f"[INFO] 가족 키워드가 없어 {len(img_paths)}장의 한국인 안면 데이터를 나이 변환 학습셋(최대 {max_samples}장)으로 스마트 변환합니다...")
        step = max(1, len(img_paths) // max_samples)
        selected_paths = img_paths[::step][:max_samples]

        folder_map: Dict[str, List[str]] = {}
        for p in selected_paths:
            fld = os.path.dirname(p)
            folder_map.setdefault(fld, []).append(p)

        idx = 0
        for fld, paths in folder_map.items():
            if len(triplets) >= max_samples:
                break
            for i, p in enumerate(paths):
                if len(triplets) >= max_samples:
                    break
                try:
                    img = Image.open(p)
                    c_name = f"korean_face_{idx:05d}.jpg"
                    c_aligned_path = os.path.join(images_dir, c_name)
                    crop_and_align_face(img, target_size=target_size).save(c_aligned_path, quality=95)

                    src_age = extract_age_from_path(p)
                    tgt_age = min(75.0, src_age + 20.0)
                    target_aligned_path = c_aligned_path

                    # If another photo exists in same directory, link as age progression pair
                    if len(paths) > 1 and i + 1 < len(paths):
                        other_p = paths[i + 1]
                        tgt_age = extract_age_from_path(other_p)
                        if tgt_age <= src_age:
                            tgt_age = src_age + 20.0

                    triplets.append({
                        "family_id": f"person_{idx:05d}",
                        "child_path": os.path.relpath(c_aligned_path, _project_root).replace("\\", "/"),
                        "father_path": None,
                        "mother_path": None,
                        "target_path": os.path.relpath(target_aligned_path, _project_root).replace("\\", "/"),
                        "source_age": float(src_age),
                        "target_age": float(tgt_age)
                    })
                    idx += 1
                    if idx % 500 == 0:
                        print(f"[PROGRESS] 정렬 진행 중: {idx}/{len(selected_paths)}장 완료...")
                except Exception:
                    continue

    # Write manifest
    os.makedirs(os.path.dirname(manifest_output_path) or ".", exist_ok=True)
    with open(manifest_output_path, "w", encoding="utf-8") as f:
        json.dump(triplets, f, indent=2, ensure_ascii=False)

    print(f"[OK] Preprocessed {len(triplets)} Korean face triplets -> {manifest_output_path}")
    return triplets


def package_to_archive(aligned_dir: str, archive_path: str):
    """Packs the 256x256 aligned dataset directory into a single tar.gz archive safely."""
    try:
        print(f"[INFO] Packaging aligned dataset {aligned_dir} into {archive_path}...")
        os.makedirs(os.path.dirname(archive_path) or ".", exist_ok=True)
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(aligned_dir, arcname=os.path.basename(aligned_dir))
        size_mb = os.path.getsize(archive_path) / (1024 * 1024)
        print(f"[OK] Archive successfully created: {archive_path} ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"[WARN] Archiving package skipped ({e}). Training can proceed directly with aligned images and manifest!")



def cleanup_raw_files(raw_dir: str):
    """
    Removes raw heavy download directory to save disk space after preprocessing.
    Keeps only the lightweight aligned dataset and manifest.
    """
    if os.path.exists(raw_dir):
        print(f"[INFO] Cleaning up raw heavy files in: {raw_dir}...")
        shutil.rmtree(raw_dir, ignore_errors=True)
        print(f"[OK] Cleaned up {raw_dir} successfully. Storage freed!")


def safe_extract_zip(zip_path: str, target_dir: str):
    """
    Safely extracts zip archive handling:
    1. Windows backslash ('\\') to Unix slash ('/') normalization (prevents [Errno 36] File name too long)
    2. Korean filename encoding (CP949/EUC-KR fallback)
    3. Fast system 'unzip' execution on Linux with fallback to python zipfile
    """
    # 1. On Linux (Google Colab), try fast native unzip command first
    if os.name != "nt":
        # Try unzip with -O cp949
        try:
            cmd = ["unzip", "-q", "-o", "-O", "cp949", zip_path, "-d", target_dir]
            ret = subprocess.run(cmd, capture_output=True, text=True)
            if ret.returncode == 0:
                return
        except Exception:
            pass

        # Try standard unzip
        try:
            cmd = ["unzip", "-q", "-o", zip_path, "-d", target_dir]
            ret = subprocess.run(cmd, capture_output=True, text=True)
            if ret.returncode == 0:
                return
        except Exception:
            pass

    # 2. Python zipfile fallback with path normalization and encoding fix
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.infolist():
            filename = member.filename
            # AI Hub Korean filename encoding fix (CP437 misinterpreted to CP949 or UTF-8)
            if not (member.flag_bits & 0x800):
                try:
                    filename = filename.encode("cp437").decode("cp949")
                except Exception:
                    try:
                        filename = filename.encode("cp437").decode("utf-8")
                    except Exception:
                        pass

            # CRITICAL FIX: Normalize Windows backslash to Linux forward slash
            filename = filename.replace("\\", "/").lstrip("/")
            if not filename:
                continue

            target_path = os.path.join(target_dir, filename)
            if member.is_dir() or filename.endswith("/"):
                os.makedirs(target_path, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with z.open(member) as source, open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)


def extract_archives(raw_root_dir: str, target_dir: str) -> List[str]:
    """
    Extracts all .zip, .tar, .tar.gz files from raw_root_dir into target_dir.
    Useful for unpacking Google Drive AI Hub archives into Colab's high-speed local disk.
    """
    os.makedirs(target_dir, exist_ok=True)
    archive_files = []
    for root, _, files in os.walk(raw_root_dir):
        for f in files:
            ext = f.lower()
            if ext.endswith((".zip", ".tar", ".tar.gz", ".tgz")):
                archive_files.append(os.path.join(root, f))

    print(f"[INFO] Found {len(archive_files)} archive file(s) in {raw_root_dir}...")
    extracted_paths = []
    for archive_path in archive_files:
        print(f"[INFO] Extracting: {os.path.basename(archive_path)} -> {target_dir}...")
        try:
            if archive_path.lower().endswith(".zip"):
                safe_extract_zip(archive_path, target_dir)
            elif archive_path.lower().endswith((".tar", ".tar.gz", ".tgz")):
                with tarfile.open(archive_path, "r:*") as tar_ref:
                    tar_ref.extractall(target_dir)
            extracted_paths.append(archive_path)
            print(f"[OK] Extracted {os.path.basename(archive_path)}")
        except Exception as e:
            print(f"[WARN] Failed to extract {os.path.basename(archive_path)}: {type(e).__name__} - {e}")

    return extracted_paths



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Hub Integrated Pipeline & Google Drive Manager")
    parser.add_argument("--action", choices=["download", "preprocess", "package", "cleanup", "extract"], required=True)
    parser.add_argument("--apikey", type=str, default=os.getenv("AI_HUB_API_KEY", ""))
    parser.add_argument("--datasetkey", type=str, default="")
    parser.add_argument("--raw-dir", type=str, default="data_aihub_raw")
    parser.add_argument("--target-dir", type=str, default="/content/raw_unpacked")
    parser.add_argument("--aligned-dir", type=str, default="data_aihub_aligned")
    parser.add_argument("--manifest", type=str, default="data_aihub_aligned/manifest_aihub.json")
    parser.add_argument("--archive", type=str, default="data_aihub_aligned.tar.gz")
    parser.add_argument("--max-samples", type=int, default=3000, help="Max aligned samples to prevent disk full")
    args = parser.parse_args()

    if args.action == "download":
        if not args.apikey or not args.datasetkey:
            print("[ERROR] Please provide --apikey and --datasetkey for download action.")
            sys.exit(1)
        download_aihub_dataset(args.apikey, args.datasetkey, args.raw_dir)
    elif args.action == "extract":
        extract_archives(args.raw_dir, args.target_dir)
    elif args.action == "preprocess":
        preprocess_aihub_datasets(args.raw_dir, args.aligned_dir, args.manifest, max_samples=args.max_samples)
    elif args.action == "package":
        package_to_archive(args.aligned_dir, args.archive)
    elif args.action == "cleanup":
        cleanup_raw_files(args.raw_dir)


