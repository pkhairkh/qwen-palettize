# TASKS: quality-recipe

## Branch
`agent/quality-recipe`

## Overview
You improve training quality (cos) via loss config, gradient clipping, re-quantization, and deterministic-ST. Your changes are orthogonal to the kernel fusion work.

## Patch Inventory
| # | Patch | Wave | Effort | Status |
|---|-------|------|--------|--------|
| 23 | Loss config switch (1-cos+norm_mse, 80/20) | 1 | 0.1 day | ⬜ |
| 24 | Per-group gradient clipping | 1 | 0.5 day | ⬜ |
| 26 | Deterministic-ST (remove Gumbel noise) | 2 | 0.5 day | ⬜ |
| 25 | LUT-Q re-quantization (step 2000, 4000) | 3 | 1 day | ⬜ |

---

## WAVE 1

### Sub-task 23a: Loss config switch
**Research:** `research-palettes-training/05_loss_function.md`
**Paper:** `docs/papers/2210.17323_GPTQ_Frantar2023.pdf`, `docs/papers/2305.14314_QLoRA_Dettmers2023.pdf`

**File:** `scripts/train_qwen.py:96-102` (DEFAULT_HYPERPARAMS)

**What:** Change `loss_type` from `"norm_mse"` to `"1-cos+norm_mse"` and `loss_weights` from `{"cos": 0.0, "mse": 1.0}` to `{"cos": 0.8, "mse": 0.2}`.

**Why:** The current pure `norm_mse` conflates magnitude and direction errors. The 80/20 split balances gradient contributions: `cos` explicitly optimizes the direction metric we care about, `mse` provides stable magnitude gradient at training start.

**Commit:** `Patch 23: loss config 1-cos+norm_mse (cos=0.8, mse=0.2) — was norm_mse only`

### Sub-task 24a: Per-group gradient clipping
**Research:** `research-kernel-accuracy/08_recommendations.md` Fix 4
**Paper:** `docs/papers/1711.05101_AdamW_Loshchilov2019.pdf` (gradient clipping best practices)

**File:** `scripts/train_qwen.py:1130-1145` (clip_grad_norm_ section)

**What:** Replace global `clip_grad_norm_(model.parameters(), 0.3)` with per-group clipping:
- `indices_params` (1.78B index_logits, clip 1.0) — separate because their tiny Gumbel grads would be zeroed by the global norm.
- `other_params` (palettes + lora + layernorms, clip 0.3) — standard clip.

**Why:** The global clip scales the tiny palette gradient by ~1/45 (because 1.78B index_logits dominate the global norm), effectively zeroing palette updates. Per-group clip lets each group's gradient flow at its natural scale.

**Commit:** `Patch 24: per-group gradient clipping (indices clip 1.0, others clip 0.3) — was global 0.3`

### Sub-task 24b: Send messages + push
- Update PROGRESS.md.
- Push.

**Commit:** `Wave 1 closeout: PROGRESS.md update`

---

## WAVE 2

### Sub-task 26a: Deterministic-ST (coordinate with triton-kernels)
**Research:** `research-indices-training/07_recommendations.md` Fix 2
**Paper:** `docs/papers/LLT_Wang_CVPR2022.pdf` (deterministic-ST pattern), `docs/papers/2203.11086_QAT_Oscillations_Nagel2022.pdf` (oscillation prevention)

**File:** `scripts/triton_soft_forward.py` (compute_P_W_ste_kernel) — BUT this is owned by triton-kernels!

**What:** Remove the 4 Gumbel noise samples from the softmax computation. Instead of `(logits + gumbel) / tau → softmax → P`, use `logits / tau → softmax → P`. This eliminates the LCG-based Gumbel sampler (statistically weak) and makes the forward deterministic. The STE still works: forward = hard argmax, backward = soft gradient. This prevents the index oscillation pathology documented by Nagel et al. 2022.

**Communication:**
- Send to triton-kernels: "REQUEST: remove Gumbel noise from compute_P_W_ste_kernel. The forward should be `logits / tau → softmax → P` (no Gumbel sampling). Remove the `_gumbel_sample` function and the 4 Gumbel noise loads in the kernel. Also remove the `step_seed` parameter (it becomes unused). See `research-indices-training/07_recommendations.md` Fix 2 and `docs/papers/LLT_Wang_CVPR2022.pdf`."
- Wait for triton-kernels to confirm the kernel change is done.
- Then update `train_qwen.py` to remove the `step_seed` parameter from the `triton_soft_linear()` call.

**Commit:** `Patch 26: deterministic-ST — coordinate with triton-kernels to remove Gumbel noise`

### Sub-task 26b: Send messages + push
- Update PROGRESS.md.
- Push.

**Commit:** `Wave 2 closeout: PROGRESS.md update`

---

## WAVE 3

### Sub-task 25a: LUT-Q re-quantization
**Research:** `research-palettes-training/06_staged_training.md` Schedule C
**Paper:** `docs/papers/1811.05355_LUTQ_Cardinaux2018.pdf`, `docs/papers/2203.11086_QAT_Oscillations_Nagel2022.pdf`

**File:** NEW `scripts/re_quantize.py`, `scripts/train_qwen.py` (call site at step 2000, 4000)

**What:** Re-run k-means on the current `W_recon` per group every 2000 steps. This escapes the k-means local optimum that the gradient descent is stuck in. After re-quantization, re-initialize `index_logits` as one-hot from the new indices (±3 gap instead of ±10 for better gradient flow — see `research-indices-training/01_gumbel_softmax_audit.md` Finding 12).

**Implementation:**
```python
# scripts/re_quantize.py
def re_quantize_indices(model, sb_idx):
    """Re-run k-means on current W_recon, update indices + index_logits."""
    for name, mod in model.named_modules():
        if hasattr(mod, 'palette') and hasattr(mod, 'index_logits'):
            # 1. Reconstruct W from current palette + argmax(index_logits)
            argmax_idx = mod.index_logits.argmax(dim=0)  # (K, N)
            g_idx = torch.arange(N, device=mod.palette.device) // mod.group_size
            W_recon = mod.palette[g_idx, argmax_idx]  # (K, N) bf16

            # 2. Re-run k-means per group
            from palettize_pytorch import kmeans1d_weighted
            for g in range(mod.n_groups):
                start = g * mod.group_size
                end = start + mod.group_size
                W_group = W_recon[:, start:end].float()  # (K, GS)
                # Hessian weighting (uniform for now — or load from calib)
                hess = torch.ones(W_group.shape[1], device=W_group.device)
                indices_new, lut_new, _ = kmeans1d_weighted(
                    W_group, hess, mod.palette_size, mod.group_size
                )
                # 3. Update palette + indices + index_logits
                mod.palette.data[g] = lut_new.to(torch.bfloat16)
                mod.indices_int8[:, start:end] = indices_new.to(torch.int8)
                # 4. Re-init logits as ±3 one-hot (not ±10 — better gradient flow)
                for k in range(4):
                    mask = (indices_new == k)
                    mod.index_logits.data[k, :, start:end] = torch.where(
                        mask, 3.0, -3.0
                    ).to(torch.float16)
```

**Call site in train_qwen.py:**
```python
# In training loop, after step 2000 and 4000:
if global_step in [2000, 4000] and use_soft_indices:
    from re_quantize import re_quantize_indices
    print(f"  [step {global_step}] LUT-Q re-quantization...", flush=True)
    re_quantize_indices(student, sb_idx)
```

**Commit:** `Patch 25: LUT-Q re-quantization at step 2000 + 4000 (escapes k-means local optimum)`

### Sub-task 25b: Send messages + push
- Update PROGRESS.md.
- Push.

**Commit:** `Wave 3 closeout: PROGRESS.md update`

---

## DoD
- [ ] All syntax checks pass
- [ ] Import check: `python3 -c "import sys; sys.path.insert(0,'scripts'); import re_quantize"` passes
- [ ] Loss config switched (Patch 23)
- [ ] Per-group clipping implemented (Patch 24)
- [ ] Deterministic-ST coordinated with triton-kernels (Patch 26)
- [ ] LUT-Q re-quantization script created (Patch 25)
- [ ] Branch pushed to origin
