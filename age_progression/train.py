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

import argparse
import time
from typing import Optional
import torch
from torch.utils.data import DataLoader

import json
from age_progression.config import AgeProgressionConfig
from age_progression.models.full_model import AgeProgressionModel
from age_progression.losses.total_loss import TotalAgeProgressionLoss
from age_progression.data.dataset import KinshipAgeDataset, MockKinshipAgeDataset


def train_age_progression(
    epochs: int = 5,
    batch_size: int = 2,
    lr: float = 2e-4,
    image_size: int = 256,
    device: str = "cpu",
    manifest_path: Optional[str] = None,
    checkpoint_dir: str = "checkpoints"
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    device_obj = torch.device(device)
    config = AgeProgressionConfig(image_size=image_size, batch_size=batch_size, learning_rate=lr, device=device)

    print(f"[INFO] Initializing Training Pipeline on {device} (Resolution: {image_size}x{image_size})...", flush=True)
    model = AgeProgressionModel(config).to(device_obj)
    criterion = TotalAgeProgressionLoss(config=config, id_encoder=model.id_encoder).to(device_obj)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Use dataset (Real dataset if manifest provided and exists, otherwise Mock)
    if manifest_path and os.path.exists(manifest_path):
        print(f"[INFO] Loading real dataset manifest: {manifest_path}...", flush=True)
        with open(manifest_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
        dataset = KinshipAgeDataset(samples=samples, image_size=config.image_size, is_train=True)
        print(f"[INFO] Successfully loaded {len(dataset)} real triplet samples.", flush=True)
    else:
        print("[INFO] No manifest provided. Using synthetic MockKinshipAgeDataset for smoke testing.", flush=True)
        dataset = MockKinshipAgeDataset(num_samples=16, image_size=config.image_size)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"[INFO] Starting training for {epochs} epochs ({len(loader)} batches/epoch)...", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0

        for step, batch in enumerate(loader):
            optimizer.zero_grad()

            child_img = batch["child_img"].to(device_obj)
            father_img = batch["father_img"].to(device_obj)
            mother_img = batch["mother_img"].to(device_obj)
            target_age = batch["target_age"].to(device_obj)
            source_age = batch["source_age"].to(device_obj)

            outputs = model(
                child_img=child_img,
                target_age=target_age,
                father_img=father_img,
                mother_img=mother_img,
                source_age=source_age
            )

            loss_dict = criterion(
                model_outputs=outputs,
                child_img=child_img,
                target_age=target_age
            )

            loss = loss_dict["loss_total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / max(1, len(loader))
        elapsed = time.time() - epoch_start
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Loss_Total: {avg_loss:.4f} | "
            f"L_Recon: {loss_dict['loss_recon'].item():.4f} | "
            f"L_ID: {loss_dict['loss_id'].item():.4f} | "
            f"L_Age: {loss_dict['loss_age'].item():.4f} | "
            f"L_Kin: {loss_dict['loss_kinship'].item():.4f} | "
            f"Time: {elapsed:.2f}s",
            flush=True
        )

    ckpt_path = os.path.join(checkpoint_dir, "age_progression_latest.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"[INFO] Checkpoint saved to: {ckpt_path}", flush=True)
    return ckpt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Age Progression System")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--manifest", type=str, default=None, help="Path to real dataset JSON manifest")
    args = parser.parse_args()

    train_age_progression(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        image_size=args.image_size,
        device=args.device,
        manifest_path=args.manifest
    )
