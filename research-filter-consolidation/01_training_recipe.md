# 01 — Training Recipe Patches (1-4)

> **Wave 2 deliverable.** Full detail for the 4 training recipe enhancements. Code patches included as REFERENCE ONLY — not yet applied.

---

## Patch 1: Polynomial τ Schedule with Floor at 0.5

**Source:** [Agent 3 (indices-training), `00_overview.md` Fix 1; Agent 3, `04_tau_schedule.md`; Agent 4 (palettes-training), `00_overview.md`; Agent 1 (kernel-accuracy), `03_ste_analysis.md`]

### Problem

Current τ schedule: linear decay `2.0 → 0.1` over 4000 steps. At τ ≤ 0.1, Gumbel-Softmax gradient vanishes (proven mathematically in `research-indices-training/03_gradient_flow_analysis.md`):
- `P[argmax] ≈ 1.0`, `P[non-argmax] ≈ 0`
- `grad_logits = grad_W * P * (palette - W)` → 0 for all k

**Empirical verification:** At τ=0.1 with logits=±10, ALL 25 Linears had `grad_logits = 0.0`.

### Fix

Replace linear decay with piecewise schedule:
1. **Warmup** (500 steps): hold τ at `tau_init` (2.0) — high exploration
2. **Polynomial decay** (6000 steps): `τ = tau_init * (1 - progress)^2` — quadratic, front-loads high-τ regime
3. **Hold** (remaining steps): τ stays at `tau_final = 0.5` — gradient stays alive

**Rationale (Agent 3, `04_tau_schedule.md`):** Linear schedule spends 25% of the window at τ < 0.5 where `P[loser] < 0.016` and gradients are <9% of peak. Polynomial decay (α=2) front-loads the high-τ regime, keeping gradients strong for 66% of training. **62% boost in cumulative gradient signal.**

### Expected Impact

- Indices remain trainable throughout the entire run (not just first 2000 steps)
- `gn=[indices=...]` stays non-zero past step 4000
- **+0.005-0.01 cos** from unfreezing index training in the second half

### Code Patch (REFERENCE ONLY — not yet applied)

**File:** `scripts/train_qwen.py`
**Lines:** ~1034-1040 (τ anneal logic) + ~1241-1246 (CLI defaults)

```python
# BEFORE (current, train_qwen.py:1034-1040):
        if use_soft_indices:
            tau = max(tau_final, tau_init * (1.0 - global_step / tau_anneal_steps))
            for name, mod in student.named_modules():
                if hasattr(mod, 'tau'):
                    mod.tau = tau

# AFTER (proposed):
        if use_soft_indices:
            T_WARMUP = 500
            T_ANNEAL = 6000
            if global_step < T_WARMUP:
                tau = tau_init  # 2.0 — warmup at high tau
            elif global_step < T_WARMUP + T_ANNEAL:
                progress = (global_step - T_WARMUP) / T_ANNEAL
                tau = max(tau_final, tau_init * (1.0 - progress) ** 2)  # alpha=2 quadratic
            else:
                tau = tau_final  # 0.5 — hold (NOT 0.1, which zero gradients)
            for name, mod in student.named_modules():
                if hasattr(mod, 'tau'):
                    mod.tau = tau
```

```python
# CLI defaults (train_qwen.py:1241-1246):
# BEFORE:
    ap.add_argument("--tau_init", type=float, default=0.1)
    ap.add_argument("--tau_final", type=float, default=0.01)
    ap.add_argument("--tau_anneal_steps", type=int, default=4000)

# AFTER:
    ap.add_argument("--tau_init", type=float, default=2.0,
                    help="Initial Gumbel-Softmax temperature. Default 2.0.")
    ap.add_argument("--tau_final", type=float, default=0.5,
                    help="Final Gumbel-Softmax temperature. Default 0.5 (FLOOR: below 0.5 gradients vanish).")
    ap.add_argument("--tau_anneal_steps", type=int, default=6000,
                    help="Steps over which to anneal temperature from tau_init to tau_final.")
```

### Verification

After applying, train 1000 steps and check log:
- τ should be `2.000` for steps 0-500 (warmup)
- τ should decrease quadratically from `2.0` to `0.5` over steps 500-6500
- τ should hold at `0.500` for steps 6500+
- `gn=[indices=X.XX]` should remain non-zero (target: > 1e-3) throughout

### Dependencies

- None (standalone change to τ schedule)
- Compatible with STE (Patch in `fused_lut_linear_cuda.py` already applied)

### Risks

- τ=0.5 is "softer" than τ=0.1 — W_hard may differ slightly from W_soft at convergence
- Mitigation: add final argmax extraction step (see Patch in `research-kernel-accuracy/03_ste_analysis.md`)

---

## Patch 2: LoftQ SVD Initialization for LoRA

**Source:** [Agent 1 (kernel-accuracy), `00_overview.md` Fix 1, `08_recommendations.md`; Agent 4 (palettes-training), `01_palette_audit.md`]

### Problem

`train_qwen.py:718` constructs `QwenLoRA` with `init="loftq"` but `original_weight=None`, causing the SVD branch (`qwen_model.py:209-222`) to be skipped. LoRA falls back to zero-init B (`qwen_model.py:223-225`), meaning LoRA must climb out of a zero-init valley from scratch — wasting the first ~1000 steps.

### Fix

Capture the original fp16 weights from the HF model checkpoint BEFORE palettization replaces the `nn.Linear` with `PalettizedLinear`. Pass these as `original_weight` to `QwenLoRA`, enabling the LoftQ SVD warm start.

**LoftQ (Li et al. 2023):** `LoRA_A, LoRA_B = SVD(W_orig - W_quantized)` — the LoRA initialization directly compensates the leading quantization error directions.

### Expected Impact

- **+1-2% cos** from faster LoRA convergence
- Especially impactful for the 5 worst-cosine Linears (rank-32 LoRA)
- LoRA starts at a meaningful initialization instead of zero

### Code Patch (REFERENCE ONLY — not yet applied)

**File:** `scripts/train_qwen.py`
**Lines:** ~696-725 (build_student_super_block)

```python
# BEFORE (lines 696-725):
    total_lora = 0
    BIG_LORA_RANK = lora_rank * 2
    BIG_LORA_ALPHA = lora_alpha * 2
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
                lora_mod = QwenLoRA(module, rank=rank, alpha=alpha, init="loftq",
                                    original_weight=None)  # ← BUG: None!
                # ... setattr ...

# AFTER:
    # Capture original weights for LoftQ SVD init (must happen BEFORE
    # palettization replaces the nn.Linear with PalettizedLinear).
    from qwen_model import capture_original_weights_from_checkpoint
    original_weights = capture_original_weights_from_checkpoint(sb_idx, model)

    total_lora = 0
    BIG_LORA_RANK = lora_rank * 2
    BIG_LORA_ALPHA = lora_alpha * 2
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
                                    original_weight=orig_w)  # ← FIX: pass original
                # ... setattr ...
```

**New helper needed in `scripts/qwen_model.py`:**

```python
def capture_original_weights_from_checkpoint(sb_idx, model):
    """Load original fp16 weights from HF checkpoint before palettization.
    Returns dict {tensor_name: weight_tensor}."""
    from transformers import AutoModelForCausalLM
    full_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    sb_start, sb_end = SUPER_BLOCKS[sb_idx]
    weights = {}
    for layer_idx in range(sb_start, sb_end):
        layer = full_model.model.layers[layer_idx]
        for name, param in layer.named_parameters():
            if name.endswith(".weight"):
                weights[f"model.layers.{layer_idx}.{name}"] = param.data.clone()
    del full_model
    import gc; gc.collect(); torch.cuda.empty_cache()
    return weights
```

### Verification

After applying:
- LoRA B should be NON-zero at init (SVD values present)
- First 100 steps should show faster cos improvement than zero-init
- `gn=[palettes+lora+norms=...]` should be larger in early steps

### Dependencies

- Requires loading the full HF model temporarily (extra ~4GB VRAM during build)
- Must happen BEFORE `palettize_linear()` replaces `nn.Linear`

### Risks

- Extra VRAM during build (4GB for full model) — freed after capture
- LoftQ SVD on large Linears (3072×3072) takes ~2-5 seconds per Linear

---

## Patch 3: Adaptive Logit Clamp ±5τ

**Source:** [Agent 6 (literature-review), `00_executive_summary.md` Rec 1.2; Agent 3 (indices-training), `00_overview.md` Fix 3; Agent 4 (palettes-training), `08_recommendations.md`]

### Problem

Current logit clamp: `par.data.clamp_(-20.0, 20.0)` after `opt_indices.step()`.

At τ=0.1, `softmax(±20/0.1) = softmax(±200)` overflows to `[1, 0]` in fp32 — gradient is exactly zero. The ±20 clamp was a band-aid for fp16 overflow, but it's too loose at low τ.

### Fix

Make the clamp adaptive to temperature: `±5τ` instead of `±20`.

- At τ=2.0: clamp to ±10 (loose, exploration)
- At τ=0.5: clamp to ±2.5 (tight, commitment)
- `softmax(±5) ≈ [0.993, 0.007]` — still essentially hard, but with finite-precision gradient

**Rationale (Agent 6):** BNN's tight clip (`[-1, 1]` for the FP shadow) is the canonical fix. For our logit-space parameterization, `±5τ` is the right scale.

### Expected Impact

- **+0.005-0.015 cos** from preventing logit saturation
- Keeps `P[loser]` non-zero throughout training
- Indices can still flip argmax in late training

### Code Patch (REFERENCE ONLY — not yet applied)

**File:** `scripts/train_qwen.py`
**Lines:** ~1153 (after `opt_indices.step()`)

```python
# BEFORE (current, train_qwen.py:1153):
        if opt_indices:
            with torch.no_grad():
                for name, par in student.named_parameters():
                    if "index_logits" in name:
                        par.data.clamp_(-20.0, 20.0)

# AFTER (proposed):
        if opt_indices:
            with torch.no_grad():
                for name, par in student.named_parameters():
                    if "index_logits" in name:
                        # Adaptive clamp: ±5τ (was ±20)
                        # At tau=2.0: clamp to ±10 (loose, exploration)
                        # At tau=0.5: clamp to ±2.5 (tight, commitment)
                        par.data.clamp_(-5.0 * tau, 5.0 * tau)
```

### Verification

After applying:
- Logits should stay within `±5τ` range after each step
- No NaN (clamp prevents overflow)
- `gn=[indices=...]` should be non-zero (clamp prevents saturation)

### Dependencies

- Requires `tau` variable to be in scope (already is — set at top of training loop)
- Compatible with Patch 1 (polynomial τ schedule)

### Risks

- Very tight clamp (±2.5 at τ=0.5) might prevent indices from fully committing
- Mitigation: if flip rate drops to 0, slightly loosen to ±7τ

---

## Patch 4: Group Size 256→128

**Source:** [Agent 6 (literature-review), `00_executive_summary.md` Rec 1.1; Agent 1 (kernel-accuracy), `00_overview.md`; Agent 4 (palettes-training), `02_gradient_correctness.md`]

### Problem

Current `GROUP_SIZE=256` — the largest group size in the entire 20-method literature survey. GPTQ/AWQ default is 128; aggressive methods (SqueezeLLM, AffineQuant) use 64.

Larger group = more weight diversity within the group = 4-entry k-means codebook fits worse. Halving the group size roughly halves within-group diversity, so the 4-entry codebook fits better.

### Fix

Change `GROUP_SIZE = 256` to `GROUP_SIZE = 128` in `palettize_core.py`.

**Note:** This requires RE-CALIBRATION (run `calib_qwen.py` again). The kernel already parameterizes on `group_size`, so no kernel changes needed.

### Expected Impact

- **+0.005-0.01 cos** at calibration time (before any training)
- Per-Linear cos improvement (from `research-literature-review/00_executive_summary.md`):
  - Before (GS=256): mean cos 0.937, min 0.865, max 0.989
  - After (GS=128): expect mean cos 0.945-0.950, min 0.885, max 0.992

**Memory cost:** Palette storage doubles (8 bytes/group → 16 bytes/group for same Linear). But total palette storage was ~75KB per super-block, so this is negligible.

### Code Patch (REFERENCE ONLY — not yet applied)

**File:** `scripts/palettize_core.py`
**Line:** 26

```python
# BEFORE:
GROUP_SIZE = 256

# AFTER:
GROUP_SIZE = 128  # Halved from 256 for better k-means fit (GPTQ/AWQ standard)
```

### Verification

After applying + re-calibrating:
- Run `calib_qwen.py --sb_idx 0`
- Check `logs/calib_sb0.log` for per-Linear cos values
- Mean cos should improve from ~0.937 to ~0.945-0.950

### Dependencies

- **Requires re-calibration** (run `calib_qwen.py`)
- Kernel already parameterizes on `group_size` — no kernel change needed
- Existing checkpoints (GS=256) are incompatible — must re-calibrate from scratch

### Risks

- Existing trained checkpoint (cos=0.9464) was calibrated with GS=256 — incompatible after change
- Must either: (a) re-calibrate + retrain from scratch, or (b) keep GS=256 for super-block 0 and use GS=128 for super-blocks 1-7
- Option (b) is safer — preserves existing work

### Alternative: Per-Tensor Override

Instead of global GS=128, override only for the 5 worst-cosine Linears (from `research-kernel-accuracy/00_overview.md`):

```python
# In palettize_core.py:
GROUP_SIZE = 256  # Default
GROUP_SIZE_OVERRIDES = {
    "model.layers.2.linear_attn.out_proj.weight": 128,
    "model.layers.2.linear_attn.in_proj_qkv.weight": 128,
    "model.layers.1.mlp.down_proj.weight": 128,
    "model.layers.0.linear_attn.in_proj_z.weight": 128,
    "model.layers.3.self_attn.k_proj.weight": 128,
}
def get_group_size_for_tensor(tensor_name):
    return GROUP_SIZE_OVERRIDES.get(tensor_name, GROUP_SIZE)
```

This preserves the existing checkpoint for the other 20 Linears while improving the 5 worst.

---

## Summary of Training Recipe Patches

| # | Patch | Expected cos impact | Risk | Dependencies |
|---|-------|--------------------|------|---------------|
| 1 | Polynomial τ schedule (floor 0.5) | +0.005-0.01 | Low | None |
| 2 | LoftQ SVD init for LoRA | +0.01-0.02 | Medium (extra VRAM during build) | Capture weights before palettization |
| 3 | Adaptive logit clamp ±5τ | +0.005-0.015 | Low | Requires `tau` in scope |
| 4 | Group size 256→128 | +0.005-0.01 | Medium (requires re-calibration) | Re-run calib_qwen.py |
| **Total** | | **+0.025-0.055** | | |

All 4 patches preserve our approach (Gumbel-Softmax + STE + k-means + LoRA). None replace any core component.
