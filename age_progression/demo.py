"""
Demo and Verification Script for Age Progression System
Validates end-to-end model construction, forward passes,
multi-task loss calculation, and inference pipeline.
"""

import sys
import time

# Ensure UTF-8 output encoding on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch

from age_progression.config import AgeProgressionConfig
from age_progression.models.full_model import AgeProgressionModel
from age_progression.losses.total_loss import TotalAgeProgressionLoss
from age_progression.inference import AgeProgressionPipeline


def run_verification():
    print("=" * 70)
    print("[STEP 1/5] Initializing Age Progression Architecture & Config...")
    print("=" * 70)

    config = AgeProgressionConfig(
        image_size=128,          # 128 for rapid unit verification
        id_embed_dim=512,
        age_embed_dim=128,
        latent_dim=512,
        generator_base_channels=32,  # Compact base channels for quick demo execution
        num_fusion_heads=8,
        lambda_recon=1.0,
        lambda_id=2.0,
        lambda_age=1.5,
        lambda_kinship=1.0,
        device="cpu"
    )

    start_time = time.time()
    model = AgeProgressionModel(config)
    loss_calculator = TotalAgeProgressionLoss(config=config, id_encoder=model.id_encoder)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"[OK] Model successfully instantiated.")
    print(f"   - Architecture: ResNet-50 ArcFace + Kinship Cross-Attention + AdaIN Generator")
    print(f"   - Trainable parameters: {total_params:,}")
    print(f"   - Target resolution: {config.image_size}x{config.image_size}")

    print("\n" + "=" * 70)
    print("[STEP 2/5] Testing Forward Pass with Kinship (Child + Father + Mother)...")
    print("=" * 70)

    batch_size = 2
    dummy_child = torch.randn(batch_size, 3, config.image_size, config.image_size).clamp(-1, 1)
    dummy_father = torch.randn(batch_size, 3, config.image_size, config.image_size).clamp(-1, 1)
    dummy_mother = torch.randn(batch_size, 3, config.image_size, config.image_size).clamp(-1, 1)
    target_age = torch.tensor([25.0, 45.0], dtype=torch.float32)
    source_age = torch.tensor([5.0, 7.0], dtype=torch.float32)

    outputs = model(
        child_img=dummy_child,
        target_age=target_age,
        father_img=dummy_father,
        mother_img=dummy_mother,
        source_age=source_age
    )

    pred_img = outputs["pred_img"]
    v_c = outputs["v_c"]
    v_kin = outputs["v_kin"]
    w_cond = outputs["w_cond"]
    attn_weights = outputs["attn_weights"]

    print(f"[OK] Triplet Forward Pass completed:")
    print(f"   - Input Child shape:        {tuple(dummy_child.shape)}")
    print(f"   - Output Pred shape:         {tuple(pred_img.shape)}")
    print(f"   - Child ID embedding (V_C):  {tuple(v_c.shape)}")
    print(f"   - Kinship Vector (V_Kin):    {tuple(v_kin.shape)}")
    print(f"   - Combined Latent (W_cond):  {tuple(w_cond.shape)}")
    if attn_weights is not None:
        print(f"   - Cross-Attention weights:   {tuple(attn_weights.shape)}")

    print("\n" + "=" * 70)
    print("[STEP 3/5] Testing Fallback Handling (Missing Parent Photos)...")
    print("=" * 70)

    # Test single-parent or zero-parent fallback
    outputs_solo = model(
        child_img=dummy_child,
        target_age=target_age,
        father_img=None,
        mother_img=None,
        source_age=source_age
    )
    print(f"[OK] Fallback Forward Pass completed without error:")
    print(f"   - Solo Pred shape: {tuple(outputs_solo['pred_img'].shape)}")
    print(f"   - Attention fallback handled: {outputs_solo['attn_weights'] is None}")

    print("\n" + "=" * 70)
    print("[STEP 4/5] Computing Multi-Task Composite Losses & Gradient Flow...")
    print("=" * 70)
    print(f"   Loss Formula: Loss_Total = {config.lambda_recon}*L_Recon + {config.lambda_id}*L_ID + {config.lambda_age}*L_Age + {config.lambda_kinship}*L_Kinship")

    losses = loss_calculator(
        model_outputs=outputs,
        child_img=dummy_child,
        target_age=target_age
    )

    print(f"   |-- Loss_Recon:   {losses['loss_recon'].item():.4f}")
    print(f"   |-- Loss_ID:      {losses['loss_id'].item():.4f} (ArcFace Cosine Discrepancy)")
    print(f"   |-- Loss_Age:     {losses['loss_age'].item():.4f} (Target Age MAE / 100)")
    print(f"   |-- Loss_Kinship: {losses['loss_kinship'].item():.4f} (Siamese Verification)")
    print(f"   +-- Loss_Total:   {losses['loss_total'].item():.4f}")

    # Backward gradient verification
    losses["loss_total"].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    print(f"[OK] Backward pass successful! Total gradient norm: {grad_norm.item():.4f}")

    print("\n" + "=" * 70)
    print("[STEP 5/5] Testing End-to-End Inference Pipeline...")
    print("=" * 70)

    pipeline = AgeProgressionPipeline(model=model, config=config, device="cpu")
    sample_child_tensor = dummy_child[0]
    aged_image, meta = pipeline.predict(
        child=sample_child_tensor,
        target_age=30,
        source_age=6
    )

    print(f"[OK] Inference Pipeline executed successfully:")
    print(f"   - Synthesized PIL Image size: {aged_image.size} (Width x Height)")
    print(f"   - Target age: {meta['target_age']}")
    print(f"   - Synthesized Kinship norm: {meta['v_kin_norm']:.4f}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"[DONE] All systems verified successfully in {elapsed:.2f}s!")
    print("=" * 70)


if __name__ == "__main__":
    run_verification()

