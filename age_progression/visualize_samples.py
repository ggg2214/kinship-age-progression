"""
Age Progression Timeline & Grid Visualizer
Synthesizes face images across chronological age intervals (e.g. 10 -> 60)
and stitches them into a labeled timeline comparison strip.
"""

import os
import sys

# Ensure project root is in sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Ensure UTF-8 output encoding on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from typing import List, Optional, Union
import argparse
from PIL import Image, ImageDraw, ImageFont
import torch

from age_progression.config import AgeProgressionConfig
from age_progression.inference import AgeProgressionPipeline


def create_timeline_strip(
    pipeline: AgeProgressionPipeline,
    child_img: Union[str, Image.Image, torch.Tensor],
    ages: List[int],
    father_img: Optional[Union[str, Image.Image, torch.Tensor]] = None,
    mother_img: Optional[Union[str, Image.Image, torch.Tensor]] = None,
    source_age: int = 5,
    thumb_size: int = 160
) -> Image.Image:
    """
    Renders an age progression timeline from given child and optional parents.
    Returns a combined horizontal banner image.
    """
    # 1. Synthesize predictions for each target age
    generated_images = []
    meta_list = []
    for age in ages:
        aged_img, meta = pipeline.predict(
            child=child_img,
            target_age=age,
            father=father_img,
            mother=mother_img,
            source_age=source_age
        )
        aged_img_resized = aged_img.resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        generated_images.append(aged_img_resized)
        meta_list.append(meta)

    # 2. Prepare Child Source thumbnail
    if isinstance(child_img, (str, Image.Image)):
        if isinstance(child_img, str):
            c_pil = Image.open(child_img).convert("RGB")
        else:
            c_pil = child_img.convert("RGB")
        c_thumb = c_pil.resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
    else:
        # Denormalize tensor to PIL
        t = (child_img * 0.5 + 0.5).clamp(0, 1)
        arr = (t.permute(1, 2, 0).detach().cpu().numpy() * 255).astype("uint8")
        c_thumb = Image.fromarray(arr).resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)

    # 3. Canvas layout calculation
    total_cols = len(ages) + 1  # Source + Ages
    header_height = 40
    footer_height = 36
    padding = 14
    total_width = total_cols * thumb_size + (total_cols + 1) * padding
    total_height = thumb_size + header_height + footer_height + padding * 2

    canvas = Image.new("RGB", (total_width, total_height), color=(26, 30, 42))
    draw = ImageDraw.Draw(canvas)

    # Simple text rendering helper
    def draw_centered_text(text: str, cx: int, cy: int, color=(240, 240, 240)):
        bbox = draw.textbbox((0, 0), text)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((cx - w // 2, cy - h // 2), text, fill=color)

    # Draw Title
    title = f"Long-term Missing Child Age Progression (Source Age: {source_age})"
    if father_img is not None or mother_img is not None:
        title += " [Kinship-Guided]"
    else:
        title += " [Child-Only]"
    draw_centered_text(title, total_width // 2, header_height // 2, color=(129, 212, 250))

    # Draw Source Child
    curr_x = padding
    curr_y = header_height + padding
    canvas.paste(c_thumb, (curr_x, curr_y))
    # Border
    draw.rectangle([curr_x, curr_y, curr_x + thumb_size, curr_y + thumb_size], outline=(79, 195, 247), width=2)
    # Label
    draw_centered_text(f"Original ({source_age}y)", curr_x + thumb_size // 2, curr_y + thumb_size + 18, color=(79, 195, 247))

    # Draw Progression steps
    curr_x += thumb_size + padding
    for i, (age, img, meta) in enumerate(zip(ages, generated_images, meta_list)):
        canvas.paste(img, (curr_x, curr_y))
        draw.rectangle([curr_x, curr_y, curr_x + thumb_size, curr_y + thumb_size], outline=(100, 116, 139), width=1)
        draw_centered_text(f"Age {age}", curr_x + thumb_size // 2, curr_y + thumb_size + 18, color=(226, 232, 240))
        curr_x += thumb_size + padding

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Age Progression Timeline Visualizer")
    parser.add_argument("--child", type=str, default=None, help="Path to child face image")
    parser.add_argument("--father", type=str, default=None, help="Path to father face image (optional)")
    parser.add_argument("--mother", type=str, default=None, help="Path to mother face image (optional)")
    parser.add_argument("--ages", type=int, nargs="+", default=[10, 20, 30, 40, 50, 60], help="List of target ages")
    parser.add_argument("--source-age", type=int, default=5, help="Child source age")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/age_progression_latest.pt", help="Checkpoint path")
    parser.add_argument("--output", type=str, default="outputs/timelapse_grid.png", help="Output image file path")
    parser.add_argument("--device", type=str, default=None, help="Device to run inference on (cuda or cpu)")
    parser.add_argument("--image-size", type=int, default=256, help="Image resolution")
    parser.add_argument("--test-run", action="store_true", help="Run with synthetic test faces")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    config = AgeProgressionConfig(image_size=args.image_size, device=device)

    print(f"[INFO] Initializing pipeline on {device}...")
    pipeline = AgeProgressionPipeline(config=config, checkpoint_path=args.checkpoint, device=device)

    child_face = args.child
    father_face = args.father
    mother_face = args.mother
    source_age = args.source_age

    if child_face is None and not args.test_run:
        manifest_candidates = [
            "/content/data_aihub_aligned/manifest_aihub.json",
            "/content/drive/MyDrive/AIHUB Data/manifest_aihub.json",
            "data_aihub_aligned/manifest_aihub.json"
        ]
        found_sample = False
        for m_path in manifest_candidates:
            if os.path.exists(m_path):
                try:
                    import json
                    with open(m_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    m_dir = os.path.dirname(m_path)
                    for s in samples:
                        raw_c = s.get("child_path")
                        if not raw_c:
                            continue
                        candidates = [
                            raw_c,
                            os.path.join(m_dir, raw_c),
                            os.path.join(m_dir, "images", os.path.basename(raw_c)),
                            os.path.join("/content/data_aihub_aligned/images", os.path.basename(raw_c)),
                            os.path.join("/content/drive/MyDrive/AIHUB Data/images", os.path.basename(raw_c)),
                            os.path.join(_project_root, raw_c)
                        ]
                        resolved_child = None
                        for cand in candidates:
                            if os.path.exists(cand):
                                resolved_child = cand
                                break
                        if resolved_child:
                            child_face = resolved_child
                            source_age = int(s.get("source_age", 5))
                            found_sample = True
                            print(f"[INFO] Loaded real Korean face sample from manifest: {resolved_child} (Age: {source_age})")
                            break
                    if found_sample:
                        break
                except Exception:
                    pass

        if not found_sample:
            print("[INFO] No child image specified and no local manifest sample found. Using synthetic test mode...")
            child_face = torch.randn(3, args.image_size, args.image_size).clamp(-1, 1)
            father_face = torch.randn(3, args.image_size, args.image_size).clamp(-1, 1)
            mother_face = torch.randn(3, args.image_size, args.image_size).clamp(-1, 1)
    elif args.test_run:
        print("[INFO] Running in synthetic test mode (--test-run)...")
        child_face = torch.randn(3, args.image_size, args.image_size).clamp(-1, 1)
        father_face = torch.randn(3, args.image_size, args.image_size).clamp(-1, 1)
        mother_face = torch.randn(3, args.image_size, args.image_size).clamp(-1, 1)

    print(f"[INFO] Generating age progression for ages: {args.ages}...")
    grid = create_timeline_strip(
        pipeline=pipeline,
        child_img=child_face,
        ages=args.ages,
        father_img=father_face,
        mother_img=mother_face,
        source_age=args.source_age
    )

    try:
        grid.save(args.output)
        print(f"[OK] Timeline image saved successfully to: {args.output} (Size: {grid.size})")
    except Exception as e:
        local_fallback = os.path.join("outputs", os.path.basename(args.output))
        os.makedirs("outputs", exist_ok=True)
        grid.save(local_fallback)
        print(f"[WARN] Failed to save to {args.output} ({e}). Saved locally to: {local_fallback} (Size: {grid.size})")


if __name__ == "__main__":
    main()

