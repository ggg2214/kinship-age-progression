"""
Curriculum Learning Training Pipeline for Age Progression
Implements the 3-Phase Curriculum Strategy:
- Phase 1 (Foundation): High-fidelity facial identity and reconstruction pre-training.
- Phase 2 (Asian & Age Specialization): Tuning with Asian / Korean age progression datasets.
- Phase 3 (Kinship & Longitudinal Fusion): Longitudinal aging (FG-NET) and kinship guided fusion (AI Hub Family / FIW).
"""

import os
import sys
import json
import time
import shutil
import tempfile
import uuid
import argparse
from typing import Optional, List, Dict, Any
import torch
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Ensure UTF-8 output encoding on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from age_progression.config import AgeProgressionConfig
from age_progression.models.full_model import AgeProgressionModel
from age_progression.losses.total_loss import TotalAgeProgressionLoss
from age_progression.data.dataset import KinshipAgeDataset, MockKinshipAgeDataset, collate_kinship_batch


def safe_torch_save(obj: Any, save_path: str):
    """
    Safely saves torch objects to handle Google Drive FUSE filesystem limitations.
    Direct torch.save to /content/drive causes 'basic_ios::clear: iostream error' / 'unexpected pos 128 vs 0'
    because FUSE lacks random seek / inline zip container write support.
    We save to a local temp file first, then atomically copy to save_path.
    """
    target_dir = os.path.dirname(os.path.abspath(save_path))
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)

    # Save to local temporary file first to guarantee full seekable write
    temp_dir = tempfile.gettempdir()
    temp_name = f"ckpt_{uuid.uuid4().hex[:8]}_{os.path.basename(save_path)}"
    temp_path = os.path.join(temp_dir, temp_name)
    try:
        torch.save(obj, temp_path)
        shutil.copy2(temp_path, save_path)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass



def load_dataset_or_mock(manifest_path: Optional[str], image_size: int = 128, num_mock: int = 16, max_samples: Optional[int] = None) -> torch.utils.data.Dataset:
    if manifest_path and not os.path.exists(manifest_path):
        gdrive_manifest = "/content/drive/MyDrive/AIHUB Data/manifest_aihub.json"
        if os.path.exists(gdrive_manifest):
            print(f"[DATA] 로컬 매니페스트 대신 Google Drive 백업 매니페스트({gdrive_manifest})를 사용합니다.", flush=True)
            manifest_path = gdrive_manifest

    if manifest_path and os.path.exists(manifest_path):
        print(f"[DATA] Loading dataset manifest: {manifest_path}...", flush=True)
        with open(manifest_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        if len(samples) == 1:
            print(f"[DATA] 단일 샘플을 감지하여 안정적인 배치 학습(Batch Size >= 2)을 위해 4회 복제 증강합니다.", flush=True)
            samples = samples * 4
        elif max_samples and len(samples) > max_samples:
            print(f"[DATA] Slicing dataset from {len(samples)} to {max_samples} samples for curriculum stage.", flush=True)
            samples = samples[:max_samples]
        ds = KinshipAgeDataset(samples=samples, image_size=image_size, is_train=True)
        print(f"[DATA] Loaded {len(ds)} samples from manifest.", flush=True)
        return ds
    print(f"[DATA] Manifest not found ({manifest_path}). Using Mock dataset ({num_mock} samples) for smoke testing.", flush=True)
    return MockKinshipAgeDataset(num_samples=num_mock, image_size=image_size)


def run_phase_training(
    phase_num: int,
    phase_name: str,
    model: AgeProgressionModel,
    loss_fn: TotalAgeProgressionLoss,
    dataset: torch.utils.data.Dataset,
    epochs: int = 2,
    batch_size: int = 2,
    lr: float = 2e-4,
    device: str = "cpu",
    loss_weights: Optional[Dict[str, float]] = None
):
    print("\n" + "=" * 78)
    print(f" [CURRICULUM LEARNING] Starting Phase {phase_num}: {phase_name}")
    print("=" * 78)

    device_obj = torch.device(device)
    model.to(device_obj)
    loss_fn.to(device_obj)

    effective_batch_size = max(2, min(batch_size, len(dataset))) if len(dataset) >= 2 else 1
    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        collate_fn=collate_kinship_batch if isinstance(dataset, KinshipAgeDataset) else None,
        drop_last=(len(dataset) > effective_batch_size)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Adjust loss weights for curriculum stage if specified
    if loss_weights:
        loss_fn.lambda_recon = loss_weights.get("recon", loss_fn.lambda_recon)
        loss_fn.lambda_id = loss_weights.get("id", loss_fn.lambda_id)
        loss_fn.lambda_age = loss_weights.get("age", loss_fn.lambda_age)
        loss_fn.lambda_kinship = loss_weights.get("kinship", loss_fn.lambda_kinship)

    print(f"[CONFIG] Loss Weights: Recon={loss_fn.lambda_recon}, ID={loss_fn.lambda_id}, Age={loss_fn.lambda_age}, Kinship={loss_fn.lambda_kinship}")

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0

        for step, batch in enumerate(loader):
            optimizer.zero_grad()

            child_img = batch["child_img"].to(device_obj) if batch["child_img"] is not None else None
            father_img = batch["father_img"].to(device_obj) if batch.get("father_img") is not None else None
            mother_img = batch["mother_img"].to(device_obj) if batch.get("mother_img") is not None else None
            gt_img = batch["ground_truth_img"].to(device_obj) if batch.get("ground_truth_img") is not None else None
            target_age = batch["target_age"].to(device_obj)
            source_age = batch["source_age"].to(device_obj)

            outputs = model(
                child_img=child_img,
                target_age=target_age,
                father_img=father_img,
                mother_img=mother_img,
                source_age=source_age
            )

            losses = loss_fn(
                model_outputs=outputs,
                child_img=child_img,
                target_age=target_age,
                ground_truth_target_img=gt_img
            )

            loss = losses["loss_total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item()

            if (step + 1) % 10 == 0 or (step + 1) == len(loader):
                print(
                    f"  Phase {phase_num} [Ep {epoch}/{epochs} | Step {step+1:03d}/{len(loader):03d}] "
                    f"Loss: {loss.item():.4f} "
                    f"(Recon: {losses['loss_recon'].item():.3f}, ID: {losses['loss_id'].item():.3f}, "
                    f"Age: {losses['loss_age'].item():.3f}, Kin: {losses['loss_kinship'].item():.3f})",
                    flush=True
                )

        avg_loss = running_loss / max(1, len(loader))
        elapsed = time.time() - t0
        print(f"Phase {phase_num} Epoch [{epoch:02d}/{epochs:02d}] Finished! Avg Loss: {avg_loss:.4f} | Time: {elapsed:.2f}s", flush=True)


def train_curriculum(
    phases: List[int],
    manifest_phase2: Optional[str] = "data_utkface/manifest.json",
    manifest_phase3: Optional[str] = "data_fgnet/manifest_fgnet.json",
    image_size: int = 128,
    epochs_per_phase: int = 1,
    batch_size: int = 2,
    device: str = "cpu",
    output_checkpoint: str = "checkpoints/age_progression_curriculum.pt",
    max_samples: Optional[int] = None,
    manifest_phase4: Optional[str] = None,
    **kwargs
):
    os.makedirs(os.path.dirname(output_checkpoint) or ".", exist_ok=True)
    config = AgeProgressionConfig(image_size=image_size, device=device)
    device_obj = torch.device(device)

    print(f"[INFO] Initializing Curriculum Pipeline on {device} (Res: {image_size}x{image_size})...")
    model = AgeProgressionModel(config).to(device_obj)
    loss_fn = TotalAgeProgressionLoss(config=config, id_encoder=model.id_encoder).to(device_obj)

    base_ckpt_no_ext, ext = os.path.splitext(output_checkpoint)
    if not ext:
        ext = ".pt"

    # Optional Resume
    resume_checkpoint = kwargs.get("resume_checkpoint", None)
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"[INFO] Resuming model weights from previous checkpoint: {resume_checkpoint}...", flush=True)
        state = torch.load(resume_checkpoint, map_location=device_obj)
        model.load_state_dict(state)
        print("[OK] Resumed weights loaded successfully!", flush=True)

    # -------------------------------------------------------------
    # Phase 1: Foundation (기초 공사)
    # Focus on Image Reconstruction and Identity Preservation
    # -------------------------------------------------------------
    if 1 in phases:
        ckpt_p1 = f"{base_ckpt_no_ext}_phase1_foundation{ext}"
        if os.path.exists(ckpt_p1) and resume_checkpoint is None:
            print(f"[RESUME] Phase 1 체크포인트({ckpt_p1})가 이미 존재합니다. Phase 1 학습을 안전하게 건너뛰고 가중치를 로드합니다!", flush=True)
            try:
                model.load_state_dict(torch.load(ckpt_p1, map_location=device_obj))
                print("[OK] Phase 1 가중치 로드 성공! Phase 2로 즉시 이동합니다.", flush=True)
            except Exception as e:
                print(f"[WARN] 가중치 로드 실패 ({e}), Phase 1을 재학습합니다.", flush=True)
                ds1 = MockKinshipAgeDataset(num_samples=min(16, max_samples) if max_samples else 16, image_size=image_size)
                run_phase_training(
                    phase_num=1,
                    phase_name="Foundation (Reconstruction & Identity)",
                    model=model,
                    loss_fn=loss_fn,
                    dataset=ds1,
                    epochs=epochs_per_phase,
                    batch_size=batch_size,
                    device=device,
                    loss_weights={"recon": 2.0, "id": 2.5, "age": 0.5, "kinship": 0.0}
                )
                safe_torch_save(model.state_dict(), ckpt_p1)
        else:
            ds1 = MockKinshipAgeDataset(num_samples=min(16, max_samples) if max_samples else 16, image_size=image_size)
            run_phase_training(
                phase_num=1,
                phase_name="Foundation (Reconstruction & Identity)",
                model=model,
                loss_fn=loss_fn,
                dataset=ds1,
                epochs=epochs_per_phase,
                batch_size=batch_size,
                device=device,
                loss_weights={"recon": 2.0, "id": 2.5, "age": 0.5, "kinship": 0.0}
            )
            ckpt_p1 = f"{base_ckpt_no_ext}_phase1_foundation{ext}"
            safe_torch_save(model.state_dict(), ckpt_p1)
            print(f"[CHECKPOINT] Phase 1 intermediate weights saved -> {ckpt_p1}", flush=True)

    # -------------------------------------------------------------
    # Phase 2: Asian & Age Specialization (동양인 및 연령대 최적화)
    # Focus on Chronological Aging Curve & Feature Shifts
    # -------------------------------------------------------------
    if 2 in phases:
        ckpt_p2 = f"{base_ckpt_no_ext}_phase2_aging{ext}"
        if os.path.exists(ckpt_p2) and resume_checkpoint is None:
            print(f"[RESUME] Phase 2 체크포인트({ckpt_p2})가 이미 존재합니다. Phase 2 학습을 건너뛰고 가중치를 로드합니다!", flush=True)
            try:
                model.load_state_dict(torch.load(ckpt_p2, map_location=device_obj))
                print("[OK] Phase 2 가중치 로드 성공! Phase 3으로 즉시 이동합니다.", flush=True)
            except Exception:
                ds2 = load_dataset_or_mock(manifest_phase2, image_size=image_size, num_mock=16, max_samples=max_samples)
                run_phase_training(
                    phase_num=2,
                    phase_name="Asian & Age Specialization (UTKFace / AFAD / AI Hub)",
                    model=model,
                    loss_fn=loss_fn,
                    dataset=ds2,
                    epochs=epochs_per_phase,
                    batch_size=batch_size,
                    device=device,
                    loss_weights={"recon": 1.0, "id": 2.0, "age": 2.0, "kinship": 0.5}
                )
                safe_torch_save(model.state_dict(), ckpt_p2)
        else:
            ds2 = load_dataset_or_mock(manifest_phase2, image_size=image_size, num_mock=16, max_samples=max_samples)
            run_phase_training(
                phase_num=2,
                phase_name="Asian & Age Specialization (UTKFace / AFAD / AI Hub)",
                model=model,
                loss_fn=loss_fn,
                dataset=ds2,
                epochs=epochs_per_phase,
                batch_size=batch_size,
                device=device,
                loss_weights={"recon": 1.0, "id": 2.0, "age": 2.0, "kinship": 0.5}
            )
            ckpt_p2 = f"{base_ckpt_no_ext}_phase2_aging{ext}"
            safe_torch_save(model.state_dict(), ckpt_p2)
            print(f"[CHECKPOINT] Phase 2 intermediate weights saved -> {ckpt_p2}", flush=True)

    # -------------------------------------------------------------
    # Phase 3: Kinship & Longitudinal Fusion (노화 패턴 및 혈연 주입)
    # Focus on Longitudinal ground truth (FG-NET) & Kinship guidance
    # -------------------------------------------------------------
    if 3 in phases:
        ds3 = load_dataset_or_mock(manifest_phase3, image_size=image_size, num_mock=16, max_samples=max_samples)
        run_phase_training(
            phase_num=3,
            phase_name="Kinship & Longitudinal Fusion (FG-NET & AI Hub Family)",
            model=model,
            loss_fn=loss_fn,
            dataset=ds3,
            epochs=epochs_per_phase,
            batch_size=batch_size,
            device=device,
            loss_weights={"recon": 1.0, "id": 2.0, "age": 1.5, "kinship": 1.5}
        )
        ckpt_p3 = f"{base_ckpt_no_ext}_phase3_kinship{ext}"
        safe_torch_save(model.state_dict(), ckpt_p3)
        print(f"[CHECKPOINT] Phase 3 intermediate weights saved -> {ckpt_p3}", flush=True)

    # -------------------------------------------------------------
    # Phase 4: Hairstyle & Detail Tuning (헤어 질감 및 세부 텍스처 미세조정)
    # Strong Identity Weight (id=3.0) to prevent style entanglement!
    # -------------------------------------------------------------
    if 4 in phases:
        m_p4_target = manifest_phase4 or kwargs.get("manifest_phase4", None) or manifest_phase2
        ds4 = load_dataset_or_mock(m_p4_target, image_size=image_size, num_mock=16, max_samples=max_samples)
        run_phase_training(
            phase_num=4,
            phase_name="Hairstyle & Detail Tuning (AI Hub Hairstyle & Landmark)",
            model=model,
            loss_fn=loss_fn,
            dataset=ds4,
            epochs=epochs_per_phase,
            batch_size=batch_size,
            device=device,
            loss_weights={"recon": 1.5, "id": 3.0, "age": 1.0, "kinship": 1.0}
        )
        ckpt_p4 = f"{base_ckpt_no_ext}_phase4_hairstyle{ext}"
        safe_torch_save(model.state_dict(), ckpt_p4)
        print(f"[CHECKPOINT] Phase 4 intermediate weights saved -> {ckpt_p4}", flush=True)

    safe_torch_save(model.state_dict(), output_checkpoint)
    print(f"\n[OK] Curriculum learning complete! Final model checkpoint saved to: {output_checkpoint}", flush=True)
    return output_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curriculum Learning for Age Progression")
    parser.add_argument("--phase", type=str, default="all", choices=["1", "2", "3", "4", "all"], help="Which phase to train")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs per curriculum phase")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--image-size", type=int, default=128, help="Image resolution")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional max samples to slice from dataset")
    parser.add_argument("--manifest", type=str, default=None, help="Unified manifest path for all phases")
    parser.add_argument("--manifest-phase2", type=str, default="data_utkface/manifest.json", help="Manifest for Phase 2")
    parser.add_argument("--manifest-phase3", type=str, default="data_fgnet/manifest_fgnet.json", help="Manifest for Phase 3")
    parser.add_argument("--manifest-phase4", type=str, default=None, help="Manifest for Phase 4")
    parser.add_argument("--resume-checkpoint", type=str, default=None, help="Resume training from an existing checkpoint")
    parser.add_argument("--output", "--save-path", dest="output", type=str, default="checkpoints/age_progression_curriculum.pt", help="Checkpoint output path")
    args = parser.parse_args()

    m_p2 = args.manifest if args.manifest else args.manifest_phase2
    m_p3 = args.manifest if args.manifest else args.manifest_phase3
    m_p4 = args.manifest if args.manifest else args.manifest_phase4

    phase_list = [1, 2, 3, 4] if args.phase == "all" else [int(args.phase)]
    train_curriculum(
        phases=phase_list,
        manifest_phase2=m_p2,
        manifest_phase3=m_p3,
        image_size=args.image_size,
        epochs_per_phase=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        output_checkpoint=args.output,
        max_samples=args.max_samples,
        manifest_phase4=m_p4,
        resume_checkpoint=args.resume_checkpoint
    )
