# TASKS: triton-kernels

## Branch
`agent/triton-kernels`

## Overview
You optimize the per-PalettizedLinear Triton kernels. The existing kernels are fast (187ms for 25 layers) but have 4 improvements available.

## Patch Inventory
| # | Patch | Wave | Effort | Status |
|---|-------|------|--------|--------|
| 16 | Eliminate redundant matmul in STE forward | 1 | 0.5 day | ⬜ |
| 17 | Buffer pooling for P_aos + W_ste | 1 | 0.5 day | ⬜ |
| 15 | Batched compute_P_W (Triton, 25→1) | 2 | 2 days | ⬜ |
| 18 | Chunked reduction for elementwise backward | 2 | 1 day | ⬜ |

---

## WAVE 1

### Sub-task 16a: Eliminate redundant matmul in STE forward
**Research:** `research-indices-training/01_gumbel_softmax_audit.md` Finding 8
**Paper:** `docs/papers/1611.01144_Gumbel-Softmax_Jang2017.pdf` (STE implementation)

**File:** `scripts/triton_soft_forward.py` (TritonSoftLinear.forward, ~line 330)

**Problem:** The current forward computes `y_soft = x @ W_soft` inside the compute_P_W kernel (line 576 of the old CUDA path), then DISCARDS it (line 596), and recomputes `y = x @ W_ste` (line 595). This wastes one full matmul per layer × 25 layers = 50ms wasted.

**Fix:** The Triton `compute_P_W_ste_kernel` should ONLY compute `P_aos` + `W_ste` (NOT `y_soft`). The `fused_soft_matmul_kernel` computes `y = x @ W_ste` directly. `W_soft` is NOT needed in forward — it's only needed in backward, where it's recomputed on-the-fly from `P_aos + palette` (already done by `fused_soft_bwd_elementwise_kernel`).

**Verification:**
- Check that `compute_P_W_ste_triton` does NOT compute or return `y_soft`
- Check that `W_soft` is only used as a debug output (optional)
- Verify the `fused_soft_matmul_triton` only uses `W_ste`

**Commit:** `Patch 16: eliminate redundant matmul — compute_P_W_ste only returns P_aos + W_ste`

### Sub-task 17a: Buffer pooling for P_aos + W_ste
**Research:** `research-kernel-efficiency/05_memory_optimization.md` §3.2
**Paper:** (memory optimization, no specific paper)

**File:** `scripts/triton_soft_forward.py`, `scripts/triton_soft_backward.py`

**What:** Add `PalettizedLinear._P_POOL = {}` keyed on `(K, N)` shape. The forward writes `P_aos` + `W_ste` into pooled buffers (reused across steps) instead of allocating new `torch.empty` tensors every call. This eliminates ~50 `torch.empty` calls per step + associated CUDA memory management.

**Implementation:**
```python
# In triton_soft_forward.py
_P_POOL = {}  # keyed on (K, N, device) → {"P_aos": tensor, "W_ste": tensor}

def compute_P_W_ste_triton(logits, palette, group_size, tau, step_seed):
    ...
    key = (K, N, logits.device.index)
    if key not in _P_POOL:
        _P_POOL[key] = {
            "P_aos": torch.empty((K, N, 4), dtype=torch.float16, device=logits.device),
            "W_ste": torch.empty((K, N), dtype=torch.bfloat16, device=logits.device),
        }
    P_aos = _P_POOL[key]["P_aos"]
    W_ste = _P_POOL[key]["W_ste"]
    # ... kernel writes into P_aos, W_ste ...
```

**Commit:** `Patch 17: buffer pooling for P_aos + W_ste (eliminates 50 torch.empty per step)`

### Sub-task 18a (Wave 2): Send messages + push
- Send to layer-fusion: "triton_soft_forward.py API stable — compute_P_W_ste_triton returns (P_aos, W_ste) only. You can call it from your fused layer kernels."
- Send to lora-fusion: "triton_soft_backward.py API stable — fused_soft_bwd_grad_x/grad_W/elementwise_triton are ready for LoRA fusion."
- Update PROGRESS.md.
- Push.

**Commit:** `Wave 1 closeout: PROGRESS.md + inbox msgs to layer-fusion + lora-fusion`

---

## WAVE 2

### Sub-task 15a: Batched compute_P_W (Triton, 25→1)
**Research:** `research-kernel-efficiency/03_batched_compute_pw.md`
**Paper:** (kernel optimization, no specific paper)

**File:** `scripts/triton_soft_forward.py`

**What:** Replace 25 separate `compute_P_W_ste_triton` calls with a single batched kernel. Grid = `(cdiv(max_K, BM), cdiv(max_N, BN), 25)`, `blockIdx.z = layer_idx`. Each layer uses `seed_offset = layer_idx` for Gumbel noise decorrelation. A `PalettizedLayerDesc[25]` tensor (shape [25, 4]: K, N, palette_ptr, logits_ptr) is passed to the kernel.

**Implementation:**
```python
@triton.jit
def compute_P_W_ste_batched_kernel(
    layer_descs_ptr,  # (25, 4) int64 — [K, N, palette_ptr, logits_ptr] per layer
    P_aos_ptrs_ptr,   # (25,) int64 — output P_aos pointer per layer
    W_ste_ptrs_ptr,   # (25,) int64 — output W_ste pointer per layer
    max_K, max_N, n_layers,
    group_size: tl.constexpr,
    tau, base_seed,
    BM: tl.constexpr, BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    layer_idx = tl.program_id(2)
    # Load layer descriptor
    desc = tl.load(layer_descs_ptr + layer_idx * 4 + tl.arange(0, 4))
    K, N, palette_ptr, logits_ptr = desc[0], desc[1], desc[2], desc[3]
    # ... same as single-layer kernel but with per-layer K, N, pointers ...
    step_seed = base_seed + layer_idx  # decorrelate Gumbel noise
    # ... compute P_aos, W_ste, write to layer-specific output ...
```

**Commit:** `Patch 15: batched compute_P_W (25 layers → 1 kernel launch)`

### Sub-task 18a: Chunked reduction for elementwise backward
**Research:** `research-kernel-efficiency/05_memory_optimization.md` §3.3
**Paper:** (kernel optimization, no specific paper)

**File:** `scripts/triton_soft_backward.py`

**What:** The current elementwise backward materializes `grad_W` (K×N fp32) as an HBM intermediate. The chunked kernel holds `(K_CHUNK, N_TILE, 4)` in shared memory, accumulating `grad_palette` and `grad_logits` without the HBM intermediate. K_CHUNK=64, N_TILE=32.

**Commit:** `Patch 18: chunked reduction — eliminate grad_W HBM intermediate`

### Sub-task 18b: Send messages + push
- Send to lora-fusion: "Patch 18 done — elementwise backward now uses chunked reduction. You can fuse LoRA backward with the chunked grad_W."
- Update PROGRESS.md.
- Push.

**Commit:** `Wave 2 closeout: PROGRESS.md + inbox msg to lora-fusion`

---

## DoD
- [ ] All syntax checks pass
- [ ] `python3 -c "import sys; sys.path.insert(0,'scripts'); import triton_soft_forward, triton_soft_backward"` passes
- [ ] No redundant matmul in forward (Patch 16)
- [ ] Buffer pooling implemented (Patch 17)
- [ ] Batched compute_P_W implemented (Patch 15)
- [ ] Chunked reduction implemented (Patch 18)
- [ ] Branch pushed to origin
