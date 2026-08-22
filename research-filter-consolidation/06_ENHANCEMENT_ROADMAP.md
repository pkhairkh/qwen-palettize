# 06 — Master Enhancement Roadmap

> **Wave 4 deliverable.** The definitive reference for the 9 ANE-aligned enhancements. This is what the implementation agent will follow.

---

## 1. Executive Summary

This roadmap contains **9 patches** that enhance our current approach (2-bit LUT + Gumbel-Softmax + STE + k-means + LoRA) WITHOUT replacing any core component.

**Expected combined impact:**
- **cos:** +0.025-0.055 (from 0.9530 → 0.978-0.985+)
- **throughput:** ~530ms → ~200ms/step (2.5× faster, from kernel efficiency)
- **VRAM:** ~14GB saved (from fused AdamW 8-bit)
- **Unlocks:** torch.compile (1.5-2×), gradient checkpointing (batch=128+)

**No approach changes:**
- ✅ 2-bit per-group palettization (GS=256 or 128) — PRESERVED
- ✅ 1d-kmeans calibration — PRESERVED (enhanced, not replaced)
- ✅ Gumbel-Softmax trainable indices + STE — PRESERVED (enhanced, not replaced)
- ✅ Trainable palettes via AdamW — PRESERVED (optimizer optimized, not replaced)
- ✅ LoRA rank-16/32 — PRESERVED (LoftQ init added)
- ✅ PartialWrapper → nn.Module — SPEED enhancement (unlocks framework features)

---

## 2. Training Recipe Patches (4)

### Patch 1: Polynomial τ Schedule with Floor at 0.5
- **File:** `scripts/train_qwen.py` (~line 1034, ~1241)
- **Change:** Linear `2.0→0.1` → piecewise warmup (500 steps at τ=2.0) + quadratic decay to τ=0.5 (6000 steps) + hold
- **Expected:** +0.005-0.01 cos (indices remain trainable throughout)
- **Detail:** `01_training_recipe.md` § Patch 1

### Patch 2: LoftQ SVD Initialization for LoRA
- **File:** `scripts/train_qwen.py` (~line 696), `scripts/qwen_model.py` (new helper)
- **Change:** Capture original fp16 weights before palettization, pass to QwenLoRA for SVD warm start
- **Expected:** +0.01-0.02 cos (faster LoRA convergence)
- **Detail:** `01_training_recipe.md` § Patch 2

### Patch 3: Adaptive Logit Clamp ±5τ
- **File:** `scripts/train_qwen.py` (~line 1153)
- **Change:** `par.data.clamp_(-20.0, 20.0)` → `par.data.clamp_(-5.0 * tau, 5.0 * tau)`
- **Expected:** +0.005-0.015 cos (prevents logit saturation, keeps gradients flowing)
- **Detail:** `01_training_recipe.md` § Patch 3

### Patch 4: Group Size 256→128
- **File:** `scripts/palettize_core.py` (line 26)
- **Change:** `GROUP_SIZE = 256` → `GROUP_SIZE = 128`
- **Expected:** +0.005-0.01 cos at calibration (better k-means fit)
- **Note:** Requires re-calibration. Alternative: per-tensor override for 5 worst Linears only.
- **Detail:** `01_training_recipe.md` § Patch 4

---

## 3. Kernel Efficiency Patches (3)

### Patch 5: Fused Backward Kernel with AoS P Layout
- **Files:** `scripts/fused_lut_kernel.cu`, `scripts/fused_lut_linear_cuda.py`
- **Change:** P storage `(4,K,N)` SoA → `(K,N,4)` AoS. Re-enable fused bwd kernel (was 10× slow due to strided access).
- **Expected:** backward 260ms → ~60ms (saves 200ms), eliminates ~7GB intermediates
- **Detail:** `02_kernel_efficiency.md` § Patch 5

### Patch 6: Stream Double-Buffering
- **File:** `scripts/train_qwen.py` (~line 1058)
- **Change:** Persistent stream + `h_out_buf[2]` + CUDA events. Teacher prepares batch N+1 while student trains on batch N.
- **Expected:** saves ~69ms/step (teacher hidden behind student backward)
- **Detail:** `02_kernel_efficiency.md` § Patch 6

### Patch 7: Batched compute_P_W (25 Launches → 1)
- **Files:** `scripts/fused_lut_kernel.cu`, `scripts/fused_lut_linear_cuda.py`
- **Change:** Single kernel launch with `PalettizedLayerDesc[25]` array + `blockIdx.z` as layer index
- **Expected:** saves ~36ms/step (18ms forward + 18ms backward)
- **Alternative:** CUDA Graphs (if descriptor batching too invasive)
- **Detail:** `02_kernel_efficiency.md` § Patch 7

---

## 4. Speed Enhancement Patches (2)

### Patch 8: Fused AdamW for Indices
- **File:** `scripts/train_qwen.py` (~line 597)
- **Change:** `FP32MasterAdamW` → `bitsandbytes.optim.AdamW8bit` (8-bit state, handles fp16 params)
- **Expected:** optimizer 113ms → ~20ms (saves 93ms), VRAM 21.4GB → 7.1GB (saves 14GB)
- **Detail:** `03_optimizer_speedup.md` § Patch 8

### Patch 9: PartialWrapper → nn.Module
- **File:** `scripts/qwen_model.py` (~line 438-600)
- **Change:** `PartialModel` and `PartialWrapper` inherit from `nn.Module`, use `nn.ModuleList`, add `forward()`
- **Expected:** Unlocks torch.compile (1.5-2×), gradient checkpointing (batch=128+), state_dict
- **Detail:** `03_optimizer_speedup.md` § Patch 9

---

## 5. Dependencies + Ordering

```
Patch 9 (nn.Module) ────→ FIRST (unlocks everything)
  ↓
Patch 1 (τ schedule) ───→ standalone
Patch 2 (LoftQ init) ───→ standalone (needs capture helper)
Patch 3 (logit clamp) ──→ depends on Patch 1 (needs τ)
Patch 4 (group size) ───→ standalone (needs re-calibration)
  ↓
Patch 5 (fused bwd AoS) → standalone (kernel change)
Patch 7 (batched P_W) ──→ depends on Patch 5 (same P layout)
Patch 6 (stream double) → depends on data prefetch
  ↓
Patch 8 (fused AdamW) ──→ LAST (after nn.Module, needs bitsandbytes)
```

**Recommended implementation order:**

1. **Patch 9** (nn.Module) — foundation, unlocks torch.compile + checkpointing
2. **Patch 1** (τ schedule) — standalone, immediate cos impact
3. **Patch 2** (LoftQ init) — standalone, needs capture helper
4. **Patch 3** (logit clamp) — depends on Patch 1
5. **Patch 4** (group size) — standalone, needs re-calibration
6. **Patch 5** (fused bwd AoS) — kernel change, test correctness
7. **Patch 7** (batched P_W) — depends on Patch 5
8. **Patch 6** (stream double) — depends on data prefetch
9. **Patch 8** (fused AdamW) — after nn.Module, needs bitsandbytes

---

## 6. Testing Plan (How to Verify Each Patch — Do NOT Run Yet)

### Patch 1 (τ schedule)
- Train 1000 steps, check log:
  - τ = 2.000 for steps 0-500
  - τ decreases quadratically 2.0→0.5 over steps 500-6500
  - τ = 0.500 for steps 6500+
  - `gn=[indices=...]` stays non-zero (target: > 1e-3)

### Patch 2 (LoftQ init)
- Check LoRA B is NON-zero at init (SVD values present)
- First 100 steps: faster cos improvement than zero-init

### Patch 3 (logit clamp)
- Logits stay within ±5τ range after each step
- No NaN (clamp prevents overflow)
- `gn=[indices=...]` non-zero

### Patch 4 (group size)
- Re-run `calib_qwen.py --sb_idx 0`
- Check `logs/calib_sb0.log`: mean cos 0.937 → 0.945-0.950

### Patch 5 (fused bwd AoS)
- Run `test_fused_bwd.py`: `max_err < 1e-3` vs Python reference
- Profile: backward 260ms → ~60ms
- VRAM: ~7GB freed

### Patch 6 (stream double-buffer)
- GPU util stays >95% (no gaps)
- Power stays >400W (no idle)
- `nsys profile` shows stream overlap

### Patch 7 (batched compute_P_W)
- `nsys profile`: 25 launches → 1 launch
- Output matches per-Linear: `max_err < 1e-4`
- Forward time: ~18ms saved

### Patch 8 (fused AdamW)
- No NaN (8-bit state stable)
- Optimizer step: 113ms → ~20ms
- VRAM: 21.4GB → 7.1GB

### Patch 9 (nn.Module)
- `isinstance(student, nn.Module)` → True
- `student.parameters()` works without custom impl
- `torch.compile(student)` succeeds
- `student.state_dict()` returns single dict

---

## 7. Risks + Rollback Notes

### High-Risk Patches (test carefully)
- **Patch 5** (fused bwd AoS): Layout change affects all P reads. If fused kernel still slower than Python, keep Python path. Test: `max_err < 1e-3`.
- **Patch 6** (stream double-buffer): Event sync complexity. Deadlocks possible. Test: run 500 steps without stall.
- **Patch 9** (nn.Module): Behavioral change. `nn.Module.to(dtype)` casts ALL buffers — verify `indices_int8` stays int8 (it should — PyTorch skips non-float buffers).

### Medium-Risk Patches
- **Patch 2** (LoftQ): Extra VRAM during build (4GB for full model). Freed after capture.
- **Patch 4** (group size): Existing checkpoint incompatible. Use per-tensor override to preserve existing work.
- **Patch 8** (bitsandbytes 8-bit): 8-bit state may lose precision for small gradients. Test for NaN.

### Low-Risk Patches
- **Patch 1** (τ schedule): Standalone, easy to revert.
- **Patch 3** (logit clamp): One-line change.
- **Patch 7** (batched compute_P_W): Alternative CUDA Graphs available.

### Rollback Strategy
- Each patch is independently revertable via `git revert <commit>`
- Patches 5+7 are coupled (same P layout) — revert together
- Patch 9 (nn.Module) affects checkpoint format — keep migration script handy

---

## 8. Notes

- **Training server is OFFLINE** — all patches are research-only. Do not run training until server is back.
- **1d-kmeans is our calibration** — already tested as best non-GPTQ approach. Patches enhance it (group size tuning), not replace it.
- **PartialWrapper→nn.Module (Patch 9)** is a speed enhancement that unlocks torch.compile and gradient checkpointing. It does NOT change the training approach.
- See original research folders (`research-kernel-accuracy/`, `research-kernel-efficiency/`, etc.) for deeper analysis of each patch.
