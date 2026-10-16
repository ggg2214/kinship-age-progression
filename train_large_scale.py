"""
One-Click Automated Large-Scale Curriculum Learning Runner
Designed for Google Colab (T4 GPU) & Local High-Performance Training.
Executes:
1. GPU Hardware & Environment Check
2. Multi-Dataset Aggregation (AI Hub + FG-NET + UTKFace + Family Triplets)
3. 3-Phase Curriculum Training with AMP & Multi-Worker Data Loading
4. Google Drive Model Weight Persistence
5. Automatic 10~80s Timeline Verification
"""

import os
import sys
import argparse
import subprocess
import torch

_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from age_progression.data.large_scale_loader import ensure_large_scale_manifest
from age_progression.train_curriculum import train_curriculum


def main():
    parser = argparse.ArgumentParser(description="One-Click Large-Scale Curriculum Training")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per curriculum phase")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default: 16 on GPU, 4 on CPU)")
    parser.add_argument("--image-size", type=int, default=256, help="Image resolution (256x256)")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    parser.add_argument("--output", type=str, default=None, help="Checkpoint save path")
    args = parser.parse_args()

    # 1. Device Detection
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 80)
    print(" 🚀 [장기 실종아동 나이 변환 AI: 대규모 데이터셋 통합 학습 파이프라인]")
    print("=" * 80)
    print(f" [가속 장치] {device.upper()}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  - 감지된 GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
        print(f"  - AMP (Automatic Mixed Precision): 활성화 (FP16 고속 연산)")
    else:
        print("  - [주의] 현재 CPU 모드로 실행 중입니다.")
        print("  - [권장 조치] Google Colab 메뉴 [런타임] ➔ [런타임 유형 변경] ➔ [T4 GPU] 선택 시 50배 이상 빠르게 학습됩니다.")

    batch_size = args.batch_size if args.batch_size else (16 if device == "cuda" else 4)
    print(f" [배치 크기] {batch_size}")
    print(f" [입력 해상도] {args.image_size}x{args.image_size}")
    print(f" [에포크 수] 단계별 {args.epochs} 에포크")

    # 2. Dataset Aggregation
    print("\n[Step 1/3] 대규모 데이터셋(AI Hub 한국인 + FG-NET 생애주기 + UTKFace) 수집 및 매니페스트 통합 중...")
    gdrive_manifest = "/content/drive/MyDrive/AIHUB Data/manifest_large_scale.json"
    local_manifest = "data/manifest_large_scale.json"
    target_manifest = gdrive_manifest if os.path.exists("/content/drive/MyDrive/AIHUB Data") else local_manifest

    manifest_path = ensure_large_scale_manifest(output_manifest=target_manifest)

    # 3. Model Output Directory
    if args.output:
        save_path = args.output
    elif os.path.exists("/content/drive/MyDrive/AIHUB Data"):
        save_path = "/content/drive/MyDrive/AIHUB Data/checkpoints/age_progression_korean_best.pt"
    else:
        save_path = "checkpoints/age_progression_korean_best.pt"

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    print(f" [저장 대상] {save_path}")

    # 4. Curriculum Training Execution
    print("\n[Step 2/3] 3단계 커리큘럼 대규모 학습 시작...")
    final_ckpt = train_curriculum(
        phases=[1, 2, 3],
        manifest_phase2=manifest_path,
        manifest_phase3=manifest_path,
        image_size=args.image_size,
        epochs_per_phase=args.epochs,
        batch_size=batch_size,
        device=device,
        output_checkpoint=save_path,
        large_scale=True,
        use_amp=(device == "cuda")
    )

    print(f"\n[OK] 대규모 커리큘럼 학습 성공! 최종 체크포인트: {final_ckpt}")

    # 5. Visual Verification Strip
    print("\n[Step 3/3] 학습 완료 모델 기반 10~80대 타임라인 시각화 생성...")
    output_vis = "/content/drive/MyDrive/AIHUB Data/korean_sample_timelapse.png" if os.path.exists("/content/drive/MyDrive/AIHUB Data") else "outputs/korean_sample_timelapse.png"
    vis_cmd = [
        sys.executable,
        "age_progression/visualize_samples.py",
        "--checkpoint", final_ckpt,
        "--output", output_vis,
        "--device", device
    ]
    try:
        subprocess.run(vis_cmd, check=True)
        print(f"[OK] 10대~80대 타임라인 이미지가 성공적으로 생성되었습니다 -> {output_vis}")
    except Exception as e:
        print(f"[WARN] 시각화 실행 알림: {e}")

    print("\n" + "=" * 80)
    print(" 🎉 [대규모 데이터셋 학습 및 검증 완료!]")
    print(f"  - 모델 가중치: {final_ckpt}")
    print(f"  - 타임라인 결과: {output_vis}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
