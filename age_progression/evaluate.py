"""
Quantitative Evaluation and Ablation Benchmark for Age Progression
Compares Solo Age Progression vs Kinship Guided Age Progression across:
1. Identity Preservation (ArcFace Cosine Similarity)
2. Age Accuracy (Mean Absolute Error on Age Estimator)
3. Kinship Resemblance (Siamese distance to parents)
4. Attention weights distribution (Father vs Mother influence)
"""

import sys
import os

# Ensure project root is in sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Ensure UTF-8 output encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import argparse
from typing import Dict, List, Optional, Any
import torch
import numpy as np

from age_progression.config import AgeProgressionConfig
from age_progression.models.full_model import AgeProgressionModel
from age_progression.losses.total_loss import TotalAgeProgressionLoss
from age_progression.data.dataset import MockKinshipAgeDataset, KinshipAgeDataset, collate_kinship_batch
import json


def evaluate_batch(
    model: AgeProgressionModel,
    loss_calculator: TotalAgeProgressionLoss,
    batch: Dict[str, torch.Tensor],
    with_kinship: bool = True
) -> Dict[str, float]:
    """Evaluates a batch in either Solo or Kinship-guided mode."""
    child_img = batch["child_img"]
    target_age = batch["target_age"]
    source_age = batch["source_age"]
    father_img = batch.get("father_img") if with_kinship else None
    mother_img = batch.get("mother_img") if with_kinship else None
    gt_img = batch.get("ground_truth_img")

    with torch.no_grad():
        outputs = model(
            child_img=child_img,
            target_age=target_age,
            father_img=father_img,
            mother_img=mother_img,
            source_age=source_age
        )

        losses = loss_calculator(
            model_outputs=outputs,
            child_img=child_img,
            target_age=target_age,
            ground_truth_target_img=gt_img
        )

        pred_img = outputs["pred_img"]
        # 1. Identity Cosine Similarity (higher is better, [0, 1])
        v_c = outputs["v_c"]
        v_pred = model.id_encoder(pred_img)
        cos_sim = torch.sum(v_c * v_pred, dim=-1).mean().item()

        # 2. Age MAE
        pred_age = loss_calculator.age_loss_fn.age_estimator(pred_img)
        age_mae = torch.abs(pred_age - target_age).mean().item()

        # 3. Kinship Resemblance (Cosine similarity to parents)
        kinship_sim = 0.0
        if with_kinship and father_img is not None and mother_img is not None:
            v_f = model.id_encoder(father_img)
            v_m = model.id_encoder(mother_img)
            sim_f = torch.sum(v_pred * v_f, dim=-1).mean().item()
            sim_m = torch.sum(v_pred * v_m, dim=-1).mean().item()
            kinship_sim = (sim_f + sim_m) / 2.0

        # 4. Attention statistics (Father % vs Mother %)
        attn_father = 0.5
        attn_mother = 0.5
        if with_kinship and outputs["attn_weights"] is not None:
            avg_attn = outputs["attn_weights"].squeeze(1).mean(dim=0).cpu().numpy()
            attn_father = float(avg_attn[0])
            attn_mother = float(avg_attn[1])

    return {
        "loss_total": float(losses["loss_total"].item()),
        "loss_recon": float(losses["loss_recon"].item()),
        "identity_similarity": float(cos_sim),
        "age_mae": float(age_mae),
        "kinship_similarity": float(kinship_sim),
        "attn_father_pct": float(attn_father * 100),
        "attn_mother_pct": float(attn_mother * 100)
    }


def run_benchmark(
    num_samples: int = 16,
    checkpoint_path: Optional[str] = None,
    manifest_path: Optional[str] = None
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = AgeProgressionConfig(image_size=128, device=device)

    print("=" * 75)
    print(f"[BENCHMARK] Running Age Progression Evaluation & Ablation Study on {device.upper()}")
    print("=" * 75)

    model = AgeProgressionModel(config).to(device)
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"[INFO] Loading checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    loss_calculator = TotalAgeProgressionLoss(config=config, id_encoder=model.id_encoder).to(device)

    from torch.utils.data import DataLoader
    if manifest_path and os.path.exists(manifest_path):
        print(f"[DATA] Loading real evaluation manifest: {manifest_path}...")
        with open(manifest_path, "r", encoding="utf-8") as f:
            samples = json.load(f)[:num_samples]
        dataset = KinshipAgeDataset(samples=samples, image_size=config.image_size, is_train=False)
        loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_kinship_batch)
    else:
        dataset = MockKinshipAgeDataset(num_samples=num_samples, image_size=config.image_size)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)

    results_kinship = []
    results_solo = []

    for batch in loader:
        batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        # 1. Kinship Guided Mode
        res_k = evaluate_batch(model, loss_calculator, batch_dev, with_kinship=True)
        results_kinship.append(res_k)

        # 2. Solo Child-only Mode (Ablation)
        res_s = evaluate_batch(model, loss_calculator, batch_dev, with_kinship=False)
        results_solo.append(res_s)

    # Average metrics
    def avg_metric(res_list, key):
        return np.mean([r[key] for r in res_list])

    k_id = avg_metric(results_kinship, "identity_similarity")
    k_age = avg_metric(results_kinship, "age_mae")
    k_kin = avg_metric(results_kinship, "kinship_similarity")
    k_loss = avg_metric(results_kinship, "loss_total")
    k_f_attn = avg_metric(results_kinship, "attn_father_pct")
    k_m_attn = avg_metric(results_kinship, "attn_mother_pct")

    s_id = avg_metric(results_solo, "identity_similarity")
    s_age = avg_metric(results_solo, "age_mae")
    s_kin = avg_metric(results_solo, "kinship_similarity")
    s_loss = avg_metric(results_solo, "loss_total")

    print("\n" + "-" * 75)
    print(" Ablation Study: Solo Mode vs Kinship-Guided Mode")
    print("-" * 75)
    print(f"{'Metric':<32} | {'Solo (Child Only)':<18} | {'Kinship Guided':<18} | {'Delta':<10}")
    print("-" * 75)
    print(f"{'Identity Cosine Similarity (↑)':<32} | {s_id:>18.4f} | {k_id:>18.4f} | {k_id - s_id:>+10.4f}")
    print(f"{'Age Prediction MAE (years) (↓)':<32} | {s_age:>18.2f} | {k_age:>18.2f} | {k_age - s_age:>+10.2f}")
    print(f"{'Kinship Resemblance Sim (↑)':<32} | {s_kin:>18.4f} | {k_kin:>18.4f} | {k_kin - s_kin:>+10.4f}")
    print(f"{'Total Multi-Task Loss (↓)':<32} | {s_loss:>18.4f} | {k_loss:>18.4f} | {k_loss - s_loss:>+10.4f}")
    print("-" * 75)
    print(f"Kinship Attention Distribution: Father {k_f_attn:.1f}% vs Mother {k_m_attn:.1f}%")
    print("=" * 75)
    print("[DONE] Benchmark evaluation complete.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Age Progression Model")
    parser.add_argument("--num-samples", type=int, default=16, help="Number of evaluation samples")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/age_progression_curriculum.pt", help="Checkpoint path")
    parser.add_argument("--manifest", type=str, default=None, help="Optional dataset manifest path")
    args = parser.parse_args()

    run_benchmark(num_samples=args.num_samples, checkpoint_path=args.checkpoint, manifest_path=args.manifest)
