# TASKS: lora-fusion

## Branch
`agent/lora-fusion`

## Overview
You fuse the LoRA backward into Triton, eliminating the 167ms `aten::add_` and 40ms `aten::mul` overhead from 31 LoRA modules.

## Patch Inventory
| # | Patch | Wave | Effort | Status |
|---|-------|------|--------|--------|
| 19 | Fused LoRA backward | 3 | 2 days | ⬜ |
| 20 | Fused LoRA + PalettizedLinear backward | 4 | 2 days | ⬜ |

---

## WAVE 3

### Sub-task 19a: Fused LoRA backward
**Research:** NEW (profiler finding: 167ms `aten::add_`, 31 LoRA modules, 3 matmuls each)
**Paper:** `docs/papers/2305.14314_QLoRA_Dettmers2023.pdf` (LoRA backward math)

**File:** NEW `scripts/triton_lora.py`, `scripts/qwen_model.py:184-265` (QwenLoRA class)

**Problem:** The current LoRA forward is `lora_out = (x @ A) @ B.T * scaling`. The backward produces:
- `grad_A = x.T @ (grad_y * scaling @ B)` — (in_dim, rank) matmul
- `grad_B = (grad_y * scaling).T @ (x @ A)` — (out_dim, rank) matmul
- `grad_x_lora = (grad_y * scaling @ B.T) @ A` — flows to grad_x
- Plus the `* scaling` elementwise (31 `aten::mul` calls)

These are all separate PyTorch ops with kernel launch overhead.

**Fix:** Fuse `grad_A` and `grad_B` into a single Triton kernel with shared `x @ A` intermediate. The scaling is fused into the `grad_y` load (multiply by `scaling` on load).

**Implementation:**
```python
@triton.jit
def fused_lora_bwd_kernel(
    x_ptr, grad_y_ptr, A_ptr, B_ptr,
    grad_A_ptr, grad_B_ptr,
    M, in_dim, out_dim, rank,
    scaling,
    stride_xm, stride_xin,
    stride_gym, stride_gyout,
    stride_ain, stride_ar,
    stride_bout, stride_br,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    # 1. Load grad_y tile, multiply by scaling (fused)
    # 2. Compute xA = x @ A (intermediate, in shared mem)
    # 3. grad_B = grad_y_scaled.T @ xA  (Triton TC matmul)
    # 4. Compute grad_y_B = grad_y_scaled @ B (intermediate, in shared mem)
    # 5. grad_A = x.T @ grad_y_B  (Triton TC matmul)
    # 6. Store grad_A, grad_B
    ...
```

**Dependency:** Wait for triton-kernels' "triton_soft_backward.py API stable" message (Patch 18 done).

**Commit:** `Patch 19: fused LoRA backward (grad_A + grad_B + scaling in one kernel)`

### Sub-task 19b: Wire fused LoRA into QwenLoRA
**File:** `scripts/qwen_model.py:184-265` (QwenLoRA class)

**What:** Replace the current `lora_out = (x_flat @ self.lora_A) @ self.lora_B.T * scaling` with a call to the fused Triton kernel. The `QwenLoRA.forward` should call `triton_lora.triton_lora_forward(x, lora_A, lora_B, scaling)` and `QwenLoRA` should use `torch.autograd.Function` to wire the fused backward.

**Commit:** `Patch 19b: wire fused LoRA kernel into QwenLoRA class`

### Sub-task 19c: Send messages + push
- Send to cuda-graphs: "LoRA backward fused — triton_lora.py ready for CUDA Graph capture"
- Update PROGRESS.md.
- Push.

**Commit:** `Wave 3 closeout: PROGRESS.md + inbox msg to cuda-graphs`

---

## WAVE 4

### Sub-task 20a: Fused LoRA + PalettizedLinear backward
**Research:** NEW (profiler finding: 120ms `aten::copy_` for gradient accumulation)
**Paper:** `docs/papers/2305.14314_QLoRA_Dettmers2023.pdf` (fused LoRA + base weight backward)

**File:** `scripts/triton_lora.py`

**Problem:** The current PalettizedLinear backward computes `grad_x_base = grad_y @ W_ste.T`, and the LoRA backward computes `grad_x_lora = grad_y @ (lora_B @ lora_A * scaling).T`. These are accumulated via `aten::add_`: `grad_x = grad_x_base + grad_x_lora` (31 `aten::add_` calls, 167ms total).

**Fix:** Fuse the LoRA backward with the PalettizedLinear backward. The combined kernel computes:
- `grad_x = grad_y @ (W_ste + lora_B @ lora_A * scaling).T` — single matmul with combined weight
- `grad_palette`/`grad_logits` (from W_ste path) — existing triton-kernels elementwise
- `grad_lora_A`/`grad_lora_B` (from LoRA path) — from Patch 19

**Implementation:**
```python
@triton.jit
def fused_pl_lora_bwd_grad_x_kernel(
    grad_y_ptr, W_ste_ptr, lora_A_ptr, lora_B_ptr,
    grad_x_ptr,
    M, N, K, rank, scaling,
    ...
):
    # 1. Load W_ste tile (K, N) bf16
    # 2. Load lora_A (in_dim, rank), lora_B (out_dim, rank)
    # 3. Compute lora_weight = lora_B @ lora_A * scaling  (K, N) — in shared mem
    # 4. combined_weight = W_ste + lora_weight
    # 5. grad_x = grad_y @ combined_weight.T  (Triton TC matmul)
    ...
```

**Commit:** `Patch 20: fused LoRA + PalettizedLinear backward (eliminates 31 aten::add_ for grad_x accumulation)`

### Sub-task 20b: Send messages + push
- Send to cuda-graphs: "All LoRA + PL backward fused — ready for CUDA Graph capture"
- Update PROGRESS.md.
- Push.

**Commit:** `Wave 4 closeout: PROGRESS.md + inbox msg to cuda-graphs`

---

## DoD
- [ ] All syntax checks pass
- [ ] Import check: `python3 -c "import sys; sys.path.insert(0,'scripts'); import triton_lora"` passes
- [ ] Fused LoRA backward (Patch 19)
- [ ] Fused LoRA + PL backward (Patch 20)
- [ ] QwenLoRA.forward uses Triton fused kernel
- [ ] Branch pushed to origin
