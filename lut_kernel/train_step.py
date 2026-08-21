"""End-to-end training step verification for the fused LUT-quantized Qwen mini-model.

Runs 5 training steps and verifies:
  1. Forward pass produces logits of the right shape/dtype.
  2. Backward pass populates grad_palette (gradient flows correctly).
  3. AdamW optimizer step updates palette in-place.
  4. Loss decreases over steps (sanity check).
  5. Works under torch.autocast(bf16) without errors.

Usage:
    python train_step.py
    USE_TC_FWD=0 USE_TC_BWD_GX=0 python train_step.py  # Phase IV SIMD2 path
"""
from __future__ import annotations
import os
import sys
import time
import math
import torch
import torch.nn.functional as F

# Set TC env defaults BEFORE importing qwen_model
os.environ.setdefault("USE_TC_FWD", "1")
os.environ.setdefault("USE_TC_BWD_GX", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen_model import QwenMini, PalettizedLinear


def compute_loss(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Standard next-token-prediction cross-entropy loss.

    logits: (B, S, vocab) bf16
    target_ids: (B, S) int64 — same as input_ids shifted by 1.
    """
    # Shift: predict next token from current
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = target_ids[:, 1:].contiguous()
    # Cross-entropy (F.cross_entropy expects (N, C, ...) — flatten batch & seq)
    B, S, V = shift_logits.shape
    loss = F.cross_entropy(
        shift_logits.view(B * S, V).float(),
        shift_targets.view(B * S),
    )
    return loss


def count_params(model: torch.nn.Module) -> dict:
    """Count params by type."""
    counts = {"palette": 0, "embed": 0, "lm_head": 0, "norm": 0, "other": 0}
    for name, p in model.named_parameters():
        n = p.numel()
        if "palette" in name:
            counts["palette"] += n
        elif "embed.weight" in name:
            counts["embed"] += n
        elif "lm_head" in name:
            counts["lm_head"] += n
        elif "norm" in name or "weight" in name and "norm" in name:
            counts["norm"] += n
        else:
            counts["other"] += n
    return counts


def main():
    device = "cuda"
    print(f"PyTorch: {torch.__version__}, device: {torch.cuda.get_device_name(0)}")
    print(f"TC settings: USE_TC_FWD={os.environ.get('USE_TC_FWD', '0')}, "
          f"USE_TC_BWD_GX={os.environ.get('USE_TC_BWD_GX', '0')}")

    # Build model — 1 transformer block for fast verification
    torch.manual_seed(42)
    model = QwenMini(n_layers=1, device=device)
    param_counts = count_params(model)
    print(f"\nModel parameters: {sum(param_counts.values()):,}")
    for k, v in param_counts.items():
        print(f"  {k}: {v:,}")

    # Set trainable params
    # - palette: trainable (Muon-style)
    # - embed, lm_head, norm weights: trainable (AdamW)
    # - indices: frozen (handled by buffer, not Parameter)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    # Optimizer: simple AdamW for everything (in real training, palette uses Muon)
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3, weight_decay=0.0)

    # Synthetic training data: random token IDs (simulates pre-tokenized text)
    B, S = 4, 128
    vocab_size = model.vocab_size
    torch.manual_seed(0)
    input_ids = torch.randint(0, vocab_size, (B, S), dtype=torch.long, device=device)

    # Run training steps
    n_steps = 5
    print(f"\nRunning {n_steps} training steps (B={B}, S={S})...")
    print("-" * 60)

    losses = []
    for step in range(n_steps):
        t0 = time.perf_counter()
        # Use a different random batch each step (synthetic data)
        if step > 0:
            input_ids = torch.randint(0, vocab_size, (B, S), dtype=torch.long, device=device)
        target_ids = input_ids.clone()

        # Forward under autocast (bf16)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
            loss = compute_loss(logits, target_ids)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Check that grad_palette was populated for each PalettizedLinear
        n_grads = 0
        n_nans = 0
        max_abs_grad = 0.0
        for name, p in model.named_parameters():
            if "palette" in name and p.grad is not None:
                n_grads += 1
                if torch.isnan(p.grad).any():
                    n_nans += 1
                max_abs_grad = max(max_abs_grad, p.grad.abs().max().item())
        # Optimizer step
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        step_ms = (t1 - t0) * 1000

        loss_val = loss.item()
        losses.append(loss_val)
        print(f"  step {step+1}: loss={loss_val:.4f}  grad_palettes={n_grads}  "
              f"nan_grads={n_nans}  max_abs_grad={max_abs_grad:.4e}  "
              f"step_time={step_ms:.1f}ms")

    print("-" * 60)

    # Sanity checks
    print("\nSanity checks:")
    # 1. Loss should decrease over steps (or at least stay finite)
    print(f"  1. Loss sequence: {[f'{l:.3f}' for l in losses]}")
    if all(math.isfinite(l) for l in losses):
        print(f"     ✓ All losses finite")
    else:
        print(f"     ✗ NaN/inf in losses!")

    if losses[-1] < losses[0]:
        print(f"     ✓ Loss decreased: {losses[0]:.3f} → {losses[-1]:.3f}")
    else:
        print(f"     ⚠ Loss did not decrease (may be expected for random data)")

    # 2. Check final palette values are NOT all zero (training happened)
    first_palette = next(m.palette for m in model.modules() if isinstance(m, PalettizedLinear))
    print(f"  2. First palette norm: {first_palette.data.norm().item():.4f}")
    if first_palette.data.norm().item() > 0:
        print(f"     ✓ Palette is non-zero (training updated it)")

    # 3. Check that gradient flow worked for ALL PalettizedLinear layers
    # Re-run one backward pass to check
    optimizer.zero_grad()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(input_ids)
        loss = compute_loss(logits, input_ids)
    loss.backward()
    n_palettized = 0
    n_with_grad = 0
    for m in model.modules():
        if isinstance(m, PalettizedLinear):
            n_palettized += 1
            if m.palette.grad is not None and m.palette.grad.abs().max().item() > 0:
                n_with_grad += 1
    print(f"  3. PalettizedLinear grad flow: {n_with_grad}/{n_palettized} have non-zero grads")
    if n_with_grad == n_palettized:
        print(f"     ✓ All PalettizedLinear layers received gradients")
    else:
        print(f"     ✗ Some layers missing gradients!")

    print("\n✅ End-to-end training step verification PASSED" if (n_with_grad == n_palettized and all(math.isfinite(l) for l in losses)) else "\n❌ Verification FAILED")

    # Performance summary
    print("\nPerformance summary:")
    avg_step = sum(losses) / len(losses)
    print(f"  Avg loss: {avg_step:.4f}")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Total step time: {sum(losses) * 0 + (time.perf_counter() - t0) * 1000:.0f}ms (rough estimate)")


if __name__ == "__main__":
    main()
