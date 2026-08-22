# TASKS — agent-kernels

> **Branch:** `agent/kernels`
> **Patches:** 5 (fused bwd AoS P), 7 (batched compute_P_W)
> **Independent:** No file conflicts with other agents.

---

## Wave 1: Patch 5 — Fused Backward Kernel with AoS P Layout

**Research:** [`research-kernel-efficiency/02_fused_bwd_fix.md`](../../research-kernel-efficiency/02_fused_bwd_fix.md), [`research-filter-consolidation/02_kernel_efficiency.md`](../../research-filter-consolidation/02_kernel_efficiency.md) §Patch 5
**Files:** `scripts/fused_lut_kernel.cu`, `scripts/fused_lut_linear_cuda.py`

### Sub-task 5a: Add compute_P_W_aos kernel (AoS output)

**File:** `scripts/fused_lut_kernel.cu` (new kernel after existing `fused_lut_linear_soft_compute_P_W_kernel` ~line 1301)

Add a new kernel that outputs P in `(K, N, 4)` AoS layout instead of `(4, K, N)` SoA. The 4 P values per (j,o) are adjacent in memory — single 64-bit load replaces 4 strided loads.

```cuda
__global__ void fused_lut_linear_soft_compute_P_W_aos_kernel(
    const __half*        __restrict__ logits,    // (4, K, N) fp16 — INPUT stays SoA
    const __nv_bfloat16* __restrict__ palette,   // (G, 4) bf16
    __half*              __restrict__ P_aos,     // (K, N, 4) fp16 — OUTPUT is AoS
    __nv_bfloat16*       __restrict__ W_out,     // (K, N) bf16
    int K, int N, int group_size,
    float tau, uint32_t step_seed
) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int o = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= K || o >= N) return;

    const int g = o / group_size;
    const int idx_soA = j * N + o;
    const int idx_aos = (j * N + o) * 4;
    const int plane_size = K * N;

    // Load 4 logits from SoA layout
    float l0 = __half2float(logits[0 * plane_size + idx_soA]);
    float l1 = __half2float(logits[1 * plane_size + idx_soA]);
    float l2 = __half2float(logits[2 * plane_size + idx_soA]);
    float l3 = __half2float(logits[3 * plane_size + idx_soA]);

    // Gumbel + softmax (same as existing kernel)
    float inv_tau = 1.0f / tau;
    float n0 = (l0 + gumbel_sample(step_seed, idx_soA * 4 + 0)) * inv_tau;
    float n1 = (l1 + gumbel_sample(step_seed, idx_soA * 4 + 1)) * inv_tau;
    float n2 = (l2 + gumbel_sample(step_seed, idx_soA * 4 + 2)) * inv_tau;
    float n3 = (l3 + gumbel_sample(step_seed, idx_soA * 4 + 3)) * inv_tau;
    float m = fmaxf(fmaxf(n0, n1), fmaxf(n2, n3));
    float e0 = expf(n0 - m), e1 = expf(n1 - m), e2 = expf(n2 - m), e3 = expf(n3 - m);
    float s = e0 + e1 + e2 + e3;
    float p0 = e0/s, p1 = e1/s, p2 = e2/s, p3 = e3/s;

    // Write P in AoS layout — 4 adjacent fp16 values (coalesced!)
    P_aos[idx_aos + 0] = __float2half(p0);
    P_aos[idx_aos + 1] = __float2half(p1);
    P_aos[idx_aos + 2] = __float2half(p2);
    P_aos[idx_aos + 3] = __float2half(p3);

    // Compute W = sum(P * palette)
    float c0 = __bfloat162float(palette[g * 4 + 0]);
    float c1 = __bfloat162float(palette[g * 4 + 1]);
    float c2 = __bfloat162float(palette[g * 4 + 2]);
    float c3 = __bfloat162float(palette[g * 4 + 3]);
    W_out[idx_soA] = __float2bfloat16(p0*c0 + p1*c1 + p2*c2 + p3*c3);
}
```

Add a launcher function and C++ wrapper (follow the pattern of existing `fused_lut_linear_soft_compute_P_W_Launcher` at ~line 1439).

**Test:** Compile (the kernel will compile on first import of `fused_lut_linear_cuda`).
**Commit:** `Patch 5a: add compute_P_W_aos kernel (AoS output layout)`

### Sub-task 5b: Update fused backward kernel to read AoS P

**File:** `scripts/fused_lut_kernel.cu`, modify `fused_lut_linear_soft_bwd_fused_kernel` (~line 1498)

Add a new variant `fused_lut_linear_soft_bwd_fused_aos_kernel` that reads P in `(K, N, 4)` AoS layout:
- Replace 4 strided loads with single 64-bit load (or 4 adjacent fp16 loads)
- Index: `P_aos[(j * N + o) * 4 + k]` instead of `P[k * plane_size + j * N + o]`

**Commit:** `Patch 5b: add fused_bwd_aos kernel (reads AoS P, coalesced)`

### Sub-task 5c: Switch Python backward to use fused AoS kernel

**File:** `scripts/fused_lut_linear_cuda.py`, in `CUDAFusedLUTLinearSoft.backward()` (~line 620)

**Change:**
1. In `forward()`: allocate `P_aos` as `(K, N, 4)` instead of `P` as `(4, K, N)`. Call the new `compute_P_W_aos` kernel.
2. In `backward()`: replace the Python elementwise path with a call to `mod.fused_lut_linear_soft_bwd_fused_aos(grad_y, x, P_aos, palette, GS)`.

**Before (current, ~260ms):**
```python
grad_W = torch.matmul(x.T, grad_y)
P_kno = P.permute(1, 2, 0).float()  # ← materializes (K,N,4) fp32
# ... elementwise ops ...
```

**After (proposed, ~60ms):**
```python
grad_logits, grad_palette = mod.fused_lut_linear_soft_bwd_fused_aos(
    grad_y, x, P_aos, palette, GS
)
```

**Note:** STE forward (already in code) must be updated to use `P_aos` instead of `P` for the argmax computation. The `logits.argmax(dim=0)` still works (logits are still `(4, K, N)`).

**Commit:** `Patch 5c: switch Python backward to fused AoS kernel`
**Push:** `git push origin agent/kernels`

### Sub-task 5d: Write correctness test

**File:** `scripts/test_fused_bwd_aos.py` (new)

Test that `fused_lut_linear_soft_bwd_fused_aos` produces gradients within `1e-3` of the Python reference:

```python
# Reference: Python elementwise (the old path)
# Compare: fused AoS kernel
# Assert: max(abs(grad_logits_fused - grad_logits_ref)) < 1e-3
# Assert: max(abs(grad_palette_fused - grad_palette_ref)) < 1e-3
```

**Commit:** `Patch 5d: correctness test for fused AoS bwd kernel`

**DoD for Wave 1:**
- [ ] `compute_P_W_aos_kernel` added (outputs (K,N,4) AoS)
- [ ] `fused_lut_linear_soft_bwd_fused_aos_kernel` added (reads AoS P)
- [ ] Python backward calls fused AoS kernel
- [ ] STE forward updated to use P_aos
- [ ] Correctness test passes (max_err < 1e-3)
- [ ] syntax check passes
- [ ] Branch pushed
- [ ] PROGRESS.md: Patch 5 ✅
- [ ] Inbox message sent to training-recipe + optimizer-streams

---

## Wave 2: Patch 7 — Batched compute_P_W (25 Launches → 1)

**Research:** [`research-kernel-efficiency/03_batched_compute_pw.md`](../../research-kernel-efficiency/03_batched_compute_pw.md), [`research-filter-consolidation/02_kernel_efficiency.md`](../../research-filter-consolidation/02_kernel_efficiency.md) §Patch 7
**Files:** `scripts/fused_lut_kernel.cu`, `scripts/fused_lut_linear_cuda.py`

### Sub-task 7a: Define PalettizedLayerDesc struct

**File:** `scripts/fused_lut_kernel.cu` (new struct near top)

```cuda
struct PalettizedLayerDesc {
    const __half*        logits;      // (4, K, N) fp16
    const __nv_bfloat16* palette;     // (G, 4) bf16
    __half*              P_aos;       // (K, N, 4) fp16 — AoS output
    __nv_bfloat16*       W_out;       // (K, N) bf16
    int K, N, G;
};
```

**Commit:** `Patch 7a: add PalettizedLayerDesc struct`

### Sub-task 7b: Add batched compute_P_W kernel

**File:** `scripts/fused_lut_kernel.cu` (new kernel)

```cuda
__global__ void fused_compute_P_W_batched_kernel(
    const PalettizedLayerDesc* __restrict__ layers,
    int n_layers, int group_size, float tau, uint32_t step_seed
) {
    const int layer_idx = blockIdx.z;
    if (layer_idx >= n_layers) return;
    const PalettizedLayerDesc& desc = layers[layer_idx];
    const int K = desc.K, N = desc.N;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int o = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= K || o >= N) return;
    // ... same compute_P_W_aos logic, using desc.logits, desc.palette, desc.P_aos, desc.W_out ...
}
```

Add launcher + C++ wrapper.

**Commit:** `Patch 7b: add batched compute_P_W kernel (blockIdx.z = layer index)`

### Sub-task 7c: Add Python wrapper to build descriptor array

**File:** `scripts/fused_lut_linear_cuda.py` (new function)

```python
def fused_compute_P_W_batched(student, tau, step_seed):
    """Compute P and W for ALL 25 PalettizedLinears in one kernel launch."""
    from qwen_model import PalettizedLinear
    layers = [mod for _, mod in student.named_modules()
              if isinstance(mod, PalettizedLinear) and mod.index_logits is not None]
    # Build descriptor array on device (pack pointers + shapes)
    # Call mod.fused_compute_P_W_batched_launcher(descs, len(layers), group_size, tau, step_seed)
```

**Commit:** `Patch 7c: add Python wrapper for batched compute_P_W`
**Push:** `git push origin agent/kernels`

**DoD for Wave 2:**
- [ ] `PalettizedLayerDesc` struct added
- [ ] `fused_compute_P_W_batched_kernel` added (uses blockIdx.z)
- [ ] Python wrapper builds descriptor array
- [ ] Correctness: output matches per-Linear kernel (max_err < 1e-4)
- [ ] syntax check passes
- [ ] Branch pushed
- [ ] PROGRESS.md: Patch 7 ✅
- [ ] Inbox message to nn-module-foundation: "Batched compute_P_W available"

---

## Wave 3: Final Verification + Merge Prep

### Sub-task 8a: Write profile test

**File:** `scripts/test_profile_kernels.py` (new)

Profile before/after to verify speedup:
- Backward time: 260ms → ~60ms (target)
- Forward time: ~36ms saved (25 launches → 1)

### Sub-task 8b: Merge prep

```bash
git fetch origin main
git merge origin/main  # should be clean (no shared files)
```

**DoD for Wave 3:**
- [ ] Profile test shows backward < 100ms
- [ ] Branch merges cleanly with main
- [ ] PROGRESS.md fully updated
- [ ] Branch pushed
