# 08 — Concrete Recommendations (with Code Patches)

This document provides drop-in code patches for the top 7 fixes identified in `00_overview.md`. Each patch is self-contained, can be applied independently, and includes the rationale, expected improvement, and testing notes.

---

## Fix 1: Enable LoftQ SVD initialization for LoRA

**Problem:** `train_qwen.py:718` constructs `QwenLoRA` with `init="loftq"` but `original_weight=None`, causing the SVD branch (`qwen_model.py:209-222`) to be skipped and the fallback zero-init B (`qwen_model.py:223-225`) to be used. LoRA must climb out of a zero-init valley from scratch.

**Expected improvement:** +1-2% cos (faster LoRA convergence, especially for the 5 worst Linears).

**Patch (file: `scripts/train_qwen.py`, function: `build_student_super_block`, around line 696):**

```python
# BEFORE (lines 696-725):
    # Attach LoRA on ALL palettized Linears.
    # Use rank-32 for the 5 worst-cosine Linears (from calib_sb0.log),
    # rank-16 for the rest. These 5 had cos < 0.93 after calibration.
    total_lora = 0
    BIG_LORA_RANK = lora_rank * 2  # 32 if lora_rank=16
    BIG_LORA_ALPHA = lora_alpha * 2  # 64 if lora_alpha=32
    BIG_LORA_TARGETS = {
        (0, "linear_attn.in_proj_z"),
        (1, "mlp.down_proj"),
        (2, "linear_attn.out_proj"),
        (2, "linear_attn.in_proj_qkv"),
        (3, "self_attn.k_proj"),
    }
    for layer_idx in range(sb_start, sb_end):
        layer = model.model.layers[layer_idx]
        for name, module in layer.named_modules():
            if isinstance(module, (PalettizedLinear, nn.Linear)) and not isinstance(module, QwenLoRA):
                is_big = (layer_idx, name) in BIG_LORA_TARGETS
                rank = BIG_LORA_RANK if is_big else lora_rank
                alpha = BIG_LORA_ALPHA if is_big else lora_alpha
                lora_mod = QwenLoRA(module, rank=rank, alpha=alpha, init="loftq", original_weight=None)
                parent = layer
                parts = name.split(".")
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], lora_mod)
                total_lora += 1

# AFTER:
    # Capture original weights for LoftQ SVD init (must happen BEFORE
    # palettization replaces the nn.Linear with PalettizedLinear).
    # We need the ORIGINAL fp16 weight, not the palettized reconstruction.
    # Strategy: re-load the original weights from the HF model checkpoint
    # (cheaper than re-loading the full model — just the relevant tensors).
    from qwen_model import capture_original_weights_from_checkpoint
    original_weights = capture_original_weights_from_checkpoint(sb_idx, model)

    # Attach LoRA on ALL palettized Linears.
    # Use rank-32 for the 5 worst-cosine Linears (from calib_sb0.log),
    # rank-16 for the rest. These 5 had cos < 0.93 after calibration.
    total_lora = 0
    BIG_LORA_RANK = lora_rank * 2  # 32 if lora_rank=16
    BIG_LORA_ALPHA = lora_alpha * 2  # 64 if lora_alpha=32
    BIG_LORA_TARGETS = {
        (0, "linear_attn.in_proj_z"),
        (1, "mlp.down_proj"),
        (2, "linear_attn.out_proj"),
        (2, "linear_attn.in_proj_qkv"),
        (3, "self_attn.k_proj"),
    }
    for layer_idx in range(sb_start, sb_end):
        layer = model.model.layers[layer_idx]
        for name, module in layer.named_modules():
            if isinstance(module, (PalettizedLinear, nn.Linear)) and not isinstance(module, QwenLoRA):
                is_big = (layer_idx, name) in BIG_LORA_TARGETS
                rank = BIG_LORA_RANK if is_big else lora_rank
                alpha = BIG_LORA_ALPHA if is_big else lora_alpha
                # Look up the original weight for this Linear
                full_name = f"model.layers.{layer_idx}.{name}.weight"
                orig_w = original_weights.get(full_name)
                lora_mod = QwenLoRA(module, rank=rank, alpha=alpha, init="loftq",
                                    original_weight=orig_w)
                parent = layer
                parts = name.split(".")
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], lora_mod)
                total_lora += 1
```

**New helper function (file: `scripts/qwen_model.py`, add after `capture_original_weights` at line 795):**

```python
def capture_original_weights_from_checkpoint(sb_idx, student_model):
    """Load original fp16 weights from the HF checkpoint for LoftQ SVD init.

    This avoids keeping the full teacher model in memory — we only need
    the weights for the Linears in the current super-block.

    Args:
        sb_idx: super-block index
        student_model: the student PartialWrapper (for layer count reference)

    Returns:
        dict: {tensor_name: weight_tensor}
    """
    from transformers import AutoModelForCausalLM
    import gc

    sb_start, sb_end = SUPER_BLOCKS[sb_idx]
    orig_weights = {}

    # Load only the relevant layers from the checkpoint
    # (HF from_pretrained loads everything, but we can extract and free)
    model_name = "Qwen/Qwen3.5-4B"
    full_model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    for layer_idx in range(sb_start, sb_end):
        layer = full_model.model.layers[layer_idx]
        layer_prefix = f"model.layers.{layer_idx}"
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                full_name = f"{layer_prefix}.{name}.weight"
                orig_weights[full_name] = module.weight.data.clone()
    del full_model
    gc.collect()
    torch.cuda.empty_cache()
    return orig_weights
```

**Testing:** After applying this patch, run training for 100 steps and verify that the LoRA B matrices are non-zero at step 0 (use `torch.norm(lora_mod.lora_B)` — should be > 0, not 0).

---

## Fix 2: Switch loss to `1-cos+norm_mse` with `cos=0.8, mse=0.2`

**Problem:** Current default is `loss_type = "norm_mse"` with `loss_weights = {"cos": 0.0, "mse": 1.0}` (`train_qwen.py:96-97`). The cosine term is monitored but not in the loss, so direction alignment is not explicitly optimized.

**Expected improvement:** +1-2% cos.

**Patch (file: `scripts/train_qwen.py`, lines 96-97):**

```python
# BEFORE:
    "loss_type": "norm_mse",
    "loss_weights": {"cos": 0.0, "mse": 1.0},

# AFTER:
    "loss_type": "1-cos+norm_mse",
    "loss_weights": {"cos": 0.8, "mse": 0.2},
```

**Testing:** After applying this patch, run training for 500 steps and verify that the logged `cos` metric improves faster than with `norm_mse` alone. Expected: cos 0.937 → 0.95+ in 500 steps (vs 0.94 with `norm_mse`).

---

## Fix 3: Promote palette to fp32

**Problem:** Palette is stored as bf16 (`qwen_model.py:82-85`), losing precision. The gradient is also cast to bf16 at the autograd boundary (`fused_lut_linear_cuda.py:168, 662`).

**Expected improvement:** +0.5-1% cos.

**Patch 3a (file: `scripts/qwen_model.py`, lines 82-85):**

```python
# BEFORE:
        self.palette = nn.Parameter(
            initial_palette.clone().to(torch.bfloat16) if initial_palette is not None
            else torch.zeros(n_groups, palette_size, dtype=torch.bfloat16)
        )

# AFTER:
        # Palette in fp32: only 2,208 params (8.8 KB), negligible memory cost.
        # fp32 preserves gradient precision through the bf16-cast bottleneck
        # at fused_lut_linear_cuda.py:168 (hard) and :662 (soft).
        PALETTE_DTYPE = torch.float32
        self.palette = nn.Parameter(
            initial_palette.clone().to(PALETTE_DTYPE) if initial_palette is not None
            else torch.zeros(n_groups, palette_size, dtype=PALETTE_DTYPE)
        )
```

**Patch 3b (file: `scripts/qwen_model.py`, in `PalettizedLinear.forward`, around line 148):**

The CUDA kernel asserts `palette.dtype == torch::kBFloat16`. We need to cast the palette to bf16 just for the kernel call, while keeping the fp32 parameter for autograd:

```python
# BEFORE (lines 140-153):
        if self._use_cuda and x_flat.is_cuda and not self.pre_transposed:
            if self.training and self.use_soft_indices and self.index_logits is not None:
                y = self._soft_kernel(
                    x_flat, self.palette, self.index_logits,
                    self.bias, self.group_size, self.tau
                )
            else:
                y = self._hard_kernel(
                    x_flat, self.palette, self.indices_int8,
                    self.bias, self.group_size
                )

# AFTER:
        if self._use_cuda and x_flat.is_cuda and not self.pre_transposed:
            # Cast palette to bf16 for the CUDA kernel (kernel asserts bf16).
            # The cast is differentiable: grad_palette_bf16 flows back to
            # grad_palette_fp32 via autograd's automatic dtype promotion.
            palette_bf16 = self.palette.to(torch.bfloat16)
            if self.training and self.use_soft_indices and self.index_logits is not None:
                y = self._soft_kernel(
                    x_flat, palette_bf16, self.index_logits,
                    self.bias, self.group_size, self.tau
                )
            else:
                y = self._hard_kernel(
                    x_flat, palette_bf16, self.indices_int8,
                    self.bias, self.group_size
                )
```

**Patch 3c (file: `scripts/fused_lut_linear_cuda.py`, line 662):**

Remove the bf16 cast on `grad_palette` in the soft path, so the gradient returns as fp32 (matching the new fp32 palette parameter):

```python
# BEFORE (line 662):
                grad_palette = contributions.sum(dim=(0, 2)).to(torch.bfloat16)

# AFTER:
                # Keep grad_palette in fp32 — palette is now fp32 (qwen_model.py).
                # The fp32 accumulation in .sum() is preserved, and the gradient
                # matches the palette dtype for autograd compatibility.
                grad_palette = contributions.sum(dim=(0, 2)).float()
```

**Patch 3d (file: `scripts/fused_lut_linear_cuda.py`, lines 167-168):**

For the hard path, the bf16 cast is inside the C++ wrapper. We need to modify the C++ source to skip the cast when the input palette is fp32. This is a more invasive change; as a quick alternative, we can override the hard-path backward in Python:

```python
# In CUDAFusedLUTLinear.backward (around line 460-491), after getting
# grad_palette from the C++ wrapper:
        # The C++ wrapper casts grad_palette to bf16 (fused_lut_linear_cuda.py:168).
        # If the palette is fp32, re-upcast the gradient to fp32.
        # This is lossy (bf16 -> fp32 doesn't recover the lost precision),
        # but it satisfies autograd's dtype contract.
        if palette.dtype == torch.float32 and grad_palette is not None:
            grad_palette = grad_palette.float()
```

**Note:** Patch 3d is a partial fix — the gradient still loses precision in the bf16 cast inside the C++ wrapper. A full fix requires modifying the C++ wrapper to skip the cast when the input palette is fp32. This is a kernel change that we defer to a follow-up.

**Testing:** After applying patches 3a-3c, run training for 100 steps and verify that `palette.grad.dtype == torch.float32` (not bf16). Check that the palette values are fp32: `palette.dtype == torch.float32`.

---

## Fix 4: Per-group gradient clipping

**Problem:** Global gradient clip = 0.3 (`train_qwen.py:98`) with 1.78B index_logits in the norm. The palette gradient is scaled by ~1/45 of its raw value.

**Expected improvement:** +1-2% cos.

**Patch (file: `scripts/train_qwen.py`, find the `clip_grad_norm_` call in the training loop and replace with per-group clipping):**

```python
# BEFORE (somewhere in the training loop, find the clip_grad_norm_ call):
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            hp.get("gradient_clip", 0.3)
        )

# AFTER:
        # Per-group gradient clipping: clip each parameter group separately
        # so that the palette gradient is not throttled by the 1.78B
        # index_logits in the global norm.
        GRAD_CLIPS = {
            "palettes": 1.0,    # was ~0.022 effective (0.3 / 13.3); now 1.0
            "lora": 1.0,        # was ~0.022 effective; now 1.0
            "indices": 0.3,     # keep tight clip on indices (large, noisy)
            "layernorms": 1.0,  # was ~0.022 effective; now 1.0
        }
        # Clip palettes + lora + layernorms together (small, well-behaved)
        small_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            group = classify_param(name, sb_idx)
            if group in ("palettes", "lora", "layernorms"):
                small_params.append(p)
        torch.nn.utils.clip_grad_norm_(small_params, max_norm=1.0)
        # Clip indices separately (large, noisy)
        indices_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            group = classify_param(name, sb_idx)
            if group == "indices":
                indices_params.append(p)
        if indices_params:
            torch.nn.utils.clip_grad_norm_(indices_params, max_norm=0.3)
```

**Testing:** After applying this patch, log the per-group gradient norms before and after clipping. Verify that the palette gradient norm is not scaled down by more than 2x (vs ~45x with the global clip).

---

## Fix 5: LUT-Q-style re-quantization

**Problem:** K-means is at a local optimum of the L2 objective. Gradient descent on the palette cannot escape this optimum because the indices are frozen (hard path) or converging to the same assignment (soft path at low τ).

**Expected improvement:** +2-3% cos.

**Patch (file: `scripts/train_qwen.py`, add new function + modify training loop):**

```python
# New function: re-quantize indices via k-means on the current W_recon
def re_quantize_indices(model, sb_idx, verbose=True):
    """Re-run k-means on the current W_recon to get new indices + palette.

    This is the LUT-Q approach: periodic re-quantization allows the indices
    to update based on the current palette, escaping the k-means local optimum.

    Args:
        model: the student model
        sb_idx: super-block index
        verbose: print progress

    Returns:
        n_changed: number of indices that changed
    """
    from qwen_model import PalettizedLinear
    from palettize_pytorch import kmeans1d_weighted
    import torch

    n_changed = 0
    n_total = 0

    for name, mod in model.named_modules():
        if not isinstance(mod, PalettizedLinear):
            continue

        # Reconstruct W_recon from current palette + indices
        # W_recon[j, o] = palette[g(o), indices[j, o]]
        with torch.no_grad():
            palette = mod.palette.float()  # (G, 4) fp32
            indices = mod.indices  # (K, N) int64
            K, N = indices.shape
            GS = mod.group_size
            G = mod.n_groups

            # Build W_recon via gather
            group_idx = torch.arange(N, device=palette.device) // GS  # (N,)
            group_per_col = group_idx.unsqueeze(0).expand(K, N)  # (K, N)
            W_recon = palette[group_per_col.long(), indices.long()]  # (K, N) fp32

            # Run k-means per group along the K dimension (for each group of N cols)
            new_indices = torch.zeros_like(indices)
            new_palette = torch.zeros_like(palette)
            for g in range(G):
                # Extract the g-th group: W_recon[:, g*GS : (g+1)*GS]
                W_group = W_recon[:, g*GS : (g+1)*GS].flatten()  # (K * GS,)
                # k-means with uniform weights (we don't have hess_diag here)
                weights = torch.ones_like(W_group)
                centers, assignments = kmeans1d_weighted(W_group, weights, k=4)
                # Reshape assignments back to (K, GS)
                new_indices[:, g*GS : (g+1)*GS] = assignments.view(K, GS)
                new_palette[g] = centers

            # Count changes
            changed = (new_indices != indices).sum().item()
            n_changed += changed
            n_total += indices.numel()

            # Update indices and palette
            mod.indices = new_indices.long()
            mod.indices_int8 = new_indices.to(torch.int8).contiguous()
            mod.palette.data.copy_(new_palette.to(mod.palette.dtype))

            # Rebuild _flat_idx cache (used by the fallback path)
            si, so = new_indices.shape
            device = new_indices.device
            group_idx2 = torch.arange(so, device=device) // GS
            group_idx_2d = group_idx2.unsqueeze(0).expand(si, so)
            mod._flat_idx = (group_idx_2d * mod.palette_size + new_indices).contiguous()

    if verbose:
        pct = 100.0 * n_changed / max(n_total, 1)
        print(f"  [re-quant] {n_changed:,}/{n_total:,} indices changed ({pct:.2f}%)", flush=True)
    return n_changed


# In the training loop, add calls to re_quantize_indices:
# (find the main training loop and add these lines at the appropriate step boundaries)

# After step 2000:
if global_step == 2000:
    print("=== Re-quantization at step 2000 ===", flush=True)
    n_changed = re_quantize_indices(model, sb_idx)
    # Re-build optimizers (palette shape unchanged, but values updated)
    # No need to rebuild — palette is the same nn.Parameter, just updated data.

# After step 4000:
if global_step == 4000:
    print("=== Re-quantization at step 4000 ===", flush=True)
    n_changed = re_quantize_indices(model, sb_idx)
    # If <1% of indices changed, we've converged — stop re-quantizing.
    if n_changed / 1.78e9 < 0.01:
        print("  <1% indices changed — re-quantization converged", flush=True)
```

**Testing:** After applying this patch, run training and verify that the re-quantization step runs at steps 2000 and 4000. Log the number of indices changed. Expected: 5-15% of indices change at step 2000, <5% at step 4000.

---

## Fix 6: Smaller GROUP_SIZE for worst-cos Linears

**Problem:** GROUP_SIZE=256 limits cos to ~0.95 for the 5 worst Linears. Smaller GROUP_SIZE (128 or 64) would improve cos by 2-4%.

**Expected improvement:** +2-4% cos on the 5 worst Linears.

**Patch (file: `scripts/palettize_core.py`, add per-tensor GROUP_SIZE override):**

```python
# Add a per-tensor GROUP_SIZE override mapping (top of file, after line 27):
PALETTE_SIZE = 1 << BITWIDTH  # 4

# New: per-tensor GROUP_SIZE override for worst-cos Linears.
# These 5 Linears had cos < 0.93 after calibration with GS=256.
# Using GS=128 doubles the palette params (still tiny: 8,832 total)
# and improves cos by 2-4%.
GROUP_SIZE_OVERRIDES = {
    "model.layers.0.linear_attn.in_proj_z.weight": 128,      # cos 0.922
    "model.layers.1.mlp.down_proj.weight": 128,              # cos 0.919
    "model.layers.2.linear_attn.out_proj.weight": 64,        # cos 0.865 (worst)
    "model.layers.2.linear_attn.in_proj_qkv.weight": 128,    # cos 0.915
    "model.layers.3.self_attn.k_proj.weight": 128,           # cos 0.923
}

def get_group_size_for_tensor(name):
    """Return the GROUP_SIZE for a tensor, using override if available."""
    return GROUP_SIZE_OVERRIDES.get(name, GROUP_SIZE)  # default 256


# In palettize_tensor_2bit (line 61), change the GROUP_SIZE usage:
def palettize_tensor_2bit(name, W_orig, X, out_dir, threshold=0.0, verbose=True):
    out_dim, in_dim = W_orig.shape
    if out_dim < 1 or in_dim < 1:
        return None

    # NEW: use per-tensor GROUP_SIZE override
    gs = get_group_size_for_tensor(name)
    if out_dim % gs != 0:
        if verbose:
            print(f"  [{name[:50]:<50s}] SKIP — out_dim {out_dim} not divisible by GS {gs}", flush=True)
        return None

    # ... (rest of function, replace GROUP_SIZE with gs)
```

**Note:** This patch requires re-running calibration for the 5 affected tensors. It is a calibration-time change, not a training-time change.

**Testing:** After re-calibration, check that the cos for the 5 worst Linears improves by 2-4%. Expected: layer-2 `out_proj` cos 0.865 → 0.90-0.92 with GS=64.

---

## Fix 7: Revive `freeze_settled_palettes`

**Problem:** `freeze_settled_palettes` (`train_qwen.py:246-299`) is implemented but never called. It implements the Nagel et al. ICML 2022 fix for QAT oscillation.

**Expected improvement:** +0.5-1% cos (prevents oscillation in late training).

**Patch (file: `scripts/train_qwen.py`, add call in training loop):**

```python
# In the training loop, after the optimizer step, add:
        # Freeze palette entries whose index assignment hasn't changed
        # between snapshots (Nagel et al. ICML 2022).
        # This prevents oscillation in late training.
        if global_step % 500 == 0 and global_step > 0:
            curr_snapshot = snapshot_palette_indices(model)
            if hasattr(self, '_prev_palette_snapshot') and self._prev_palette_snapshot is not None:
                freeze_settled_palettes(model, sb_idx, self._prev_palette_snapshot, curr_snapshot)
            self._prev_palette_snapshot = curr_snapshot
```

**Note:** The exact integration depends on the training loop structure (whether it's a class or a function). The above assumes a class-based loop; for a function-based loop, use a closure or a module-level variable.

**Testing:** After applying this patch, log the number of frozen palette groups per call. Expected: 0-5% frozen at step 500, 20-50% frozen at step 2000, 80-95% frozen at step 4000+.

---

## Summary of patches

| Fix | Files modified | Lines changed | Risk | Expected cos gain |
|---|---|---|---|---|
| 1. LoftQ SVD init | `train_qwen.py`, `qwen_model.py` | ~30 | Low | +1-2% |
| 2. Loss = 1-cos+norm_mse | `train_qwen.py` | 2 | Low | +1-2% |
| 3. Palette fp32 | `qwen_model.py`, `fused_lut_linear_cuda.py` | ~15 | Medium (kernel assertion) | +0.5-1% |
| 4. Per-group clip | `train_qwen.py` | ~25 | Low | +1-2% |
| 5. LUT-Q re-quantization | `train_qwen.py` | ~60 | Medium (new function) | +2-3% |
| 6. Smaller GROUP_SIZE | `palettize_core.py` | ~20 | High (re-calibration) | +2-4% |
| 7. Freeze settled palettes | `train_qwen.py` | ~10 | Low | +0.5-1% |

**Recommended application order:** 1, 2, 4, 7 (low risk, one-day effort) → 3 (medium risk, half-day) → 5 (medium risk, one day) → 6 (high risk, requires re-calibration, one day).

**Expected cumulative improvement:** cos 0.946 → 0.96-0.97 with fixes 1-4+7. cos 0.96-0.97 → 0.97-0.98 with fixes 5-6. The 0.999 target is not achievable at 2-bit/GS=256 without fix 6 (smaller GROUP_SIZE) and possibly not even then — the fundamental limit is ~0.97-0.98 for 2-bit/GS=64 per the FLUTE paper.
