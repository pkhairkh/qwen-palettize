# Work Instruction: Gumbel-Softmax Soft Forward Kernel for 2-bit LUT Palettization

## Context

You are working on a 2-bit LUT-quantized LLM training system (Qwen3.5-4B). You previously wrote a fused CUDA kernel for the **hard** forward/backward (`/root/lut_kernel/fused_lut_kernel.cu`). That kernel uses frozen int8 indices to gather weights from a trainable palette.

**The problem**: Training plateaus at cos~0.93 because the int8 indices are frozen — weights can't be reassigned to better palette entries as training progresses.

**The solution**: Add a **soft forward path** using Gumbel-Softmax relaxation. During training, indices are replaced by a `(K, N, 4)` probability tensor, and weights are computed as a differentiable weighted sum of all 4 palette entries. This lets gradients flow back to index logits, enabling the model to reassign weights to better clusters.

---

## Your Task

Write a new CUDA kernel `fused_lut_linear_soft` that computes the soft forward + backward. This kernel runs **alongside** the existing hard kernel — they share the same file and Python wrapper. The `PalettizedLinear` module switches between them based on `model.training`.

---

## What Already Exists (DO NOT modify)

The following files are on the remote at `/root/lut_kernel/`:

| File | Description |
|---|---|
| `fused_lut_kernel.cu` | Your existing hard kernel (4 kernels: fwd, bwd_grad_x, bwd_grad_palette, bwd_grad_bias) + TC variants |
| `fused_lut_linear_cuda.py` | Python wrapper with `CUDAFusedLUTLinear` autograd Function |
| `test_correctness.py` | 41 correctness tests (all passing) |
| `benchmark.py` | Performance benchmark |
| `reference_pytorch.py` | PyTorch reference (correctness oracle) |
| `triton_lut_linear.py` | Triton reference kernel |

**Add your new kernel to the same files.** Do NOT break the existing hard kernel.

---

## Architecture: Hard vs Soft

### Hard (existing) — eval/inference mode

```
Inputs: x (M, K) bf16, palette (G, 4) bf16, indices (K, N) int8
W[j, o] = palette[o // GS, indices[j, o]]           // gather — 1 lookup
y = x @ W                                              // matmul with TC
```

### Soft (NEW) — training mode

```
Inputs: x (M, K) bf16, palette (G, 4) bf16, logits (K, N, 4) fp16, tau (float)
P[j, o, k] = softmax((logits[j, o, k] + gumbel_noise) / tau)   // (K, N, 4) probs
W[j, o] = Σ_k P[j, o, k] * palette[o // GS, k]                 // weighted sum — NO gather
y = x @ W                                                        // matmul with TC
```

**Key difference**: `W[j, o]` is a 4-element dot product `(P[j,o,:], palette[g,:])` instead of a gather. No index needed.

---

## Soft Forward Kernel: `fused_lut_linear_soft_fwd`

### Inputs

| Name | Shape | Dtype | Description |
|---|---|---|---|
| `x` | `(M, K)` | bf16 | Input activations, row-major contiguous |
| `palette` | `(G, 4)` | bf16 | LUT entries per group, contiguous |
| `logits` | `(K, N, 4)` | fp16 | Index logits — **trainable**. Stored as 4 contiguous `(K, N)` planes: `logits[k] = logits[:, :, k]` |
| `tau` | scalar | fp32 | Temperature for Gumbel-Softmax (annealed from 1.0 → 0.01) |
| `group_size` | scalar | int | 256 (always) |

### Output

| Name | Shape | Dtype | Description |
|---|---|---|---|
| `y` | `(M, N)` | bf16 | Output, row-major contiguous |
| `P` | `(K, N, 4)` | fp16 | Saved softmax probabilities (for backward). Stored as 4 `(K, N)` planes. |

### Math

```
// Per weight element (j, o):
g = o // group_size                                           // group index

// Gumbel-Softmax:
//   Sample gumbel noise per (j, o, k): g ~ Gumbel(0, 1)
//   Gumbel(0,1) = -log(-log(U)) where U ~ Uniform(0,1)
//   Use a simple LCG per thread for U (no need for curand)

for k in 0..3:
    noisy_logit[j, o, k] = (logits[j, o, k] + gumbel[j, o, k]) / tau

P[j, o, :] = softmax(noisy_logit[j, o, :])                   // (4,) probabilities

// Construct W via weighted sum (NOT gather):
W[j, o] = P[j, o, 0] * palette[g, 0]
        + P[j, o, 1] * palette[g, 1]
        + P[j, o, 2] * palette[g, 2]
        + P[j, o, 3] * palette[g, 3]

// Standard matmul:
y[i, o] = Σ_j x[i, j] * W[j, o] + bias[o]
```

### Tiling Strategy

**Reuse the existing TC forward tiling** (BM=64, BN=64, BK=32, `mma.sync m16n8k16`).

The only change is the **W construction step** inside each K-tile loop iteration:

```cuda
// ── Hard kernel (existing): load W via gather ──
// Load indices tile: sidx[BK][BN] (uint8)
// Load 1 palette entry per weight: sw = spalette[group][sidx[k][n]]
// Store sw into sW[BK][BN]

// ── Soft kernel (NEW): load W via weighted sum ──
// Load 4 logit planes: slogits_k[BK][BN] for k=0,1,2,3 (fp16)
// Load palette: spalette[group][0..3] (4 bf16 values, from smem)
// Compute softmax per (k_tile, n_tile) element
// Compute W = Σ_k P[k] * palette[k]
// Store W into sW[BK][BN] (same buffer as hard kernel)
```

### Shared Memory Layout

```
__shared__ __nv_bfloat16 sx[BM][BK];       // x tile (same as hard)
__shared__ __nv_bfloat16 sW[BK][BN];        // W tile (same as hard — now filled by soft path)
__shared__ __nv_bfloat16 spalette[2][4];    // palette for 2 groups (same as hard)
__shared__ __nv_bfloat16 sP[BK][BN];        // NEW: probabilities (computed from logits)

// Logits are loaded from global memory per K-chunk, 4 planes:
//   slogits_0[BK][BN], slogits_1[BK][BN], slogits_2[BK][BN], slogits_3[BK][BN]
// Total logits smem: 4 * BK * BN * 2 bytes = 4 * 32 * 64 * 2 = 16 KB
// Total smem: 4KB (x) + 4KB (W) + 16B (palette) + 16KB (logits) = ~24 KB
// L4 sm_89 has 128 KB per SM — fits easily.
```

**IMPORTANT**: The logits `(K, N, 4)` must be stored as 4 separate `(K, N)` fp16 planes in global memory for coalesced access. The Python wrapper handles this layout.

### Gumbel Noise

Use a simple LCG (Linear Congruential Generator) per thread — no need for `curand`:

```cuda
__device__ float gumbel_sample(uint32_t* state) {
    // LCG: state = state * 1103515245 + 12345
    *state = *state * 1103515245u + 12345u;
    uint32_t x = *state;
    // Convert to float in (0, 1)
    float u = (float)x / 4294967296.0f;  // [0, 1)
    u = fmaxf(u, 1e-7f);  // avoid log(0)
    // Gumbel(0,1) = -log(-log(U))
    return -logf(-logf(u));
}
```

Each thread initializes its LCG state from `(blockIdx, threadIdx, step_number)` — deterministic per training step, so results are reproducible.

### Temperature Annealing

`tau` is passed as a scalar argument. The Python training loop anneals it:

```python
# Schedule: start at 1.0, decay to 0.01 over training
tau = max(0.01, 1.0 * (1.0 - global_step / anneal_steps))
```

As τ→0, the softmax becomes hard argmax, and the soft kernel converges to the hard kernel. At eval time, switch to the hard kernel entirely.

---

## Soft Backward: `fused_lut_linear_soft_bwd`

### Inputs

| Name | Shape | Dtype | Description |
|---|---|---|---|
| `grad_y` | `(M, N)` | bf16 | Gradient from upstream |
| `x` | `(M, K)` | bf16 | Saved from forward |
| `P` | `(K, N, 4)` | fp16 | Saved softmax probabilities from forward |
| `palette` | `(G, 4)` | bf16 | LUT entries |

### Outputs

| Name | Shape | Dtype | Description |
|---|---|---|---|
| `grad_x` | `(M, K)` | bf16 | Gradient w.r.t. x |
| `grad_logits` | `(K, N, 4)` | fp16 | Gradient w.r.t. logits — **the key new output** |
| `grad_palette` | `(G, 4)` | fp32 | Gradient w.r.t. palette |
| `grad_bias` | `(N,)` | bf16 | Gradient w.r.t. bias |

### Math

```
// ── Step 1: Compute grad_W = x.T @ grad_y ──
// Same as hard kernel: standard matmul, reuse TC tiles
// grad_W: (K, N) fp32

// ── Step 2: Compute grad_x = grad_y @ W.T ──
// W is reconstructed from P and palette (same as forward)
// Same TC matmul as hard kernel's bwd_grad_x
// grad_x: (M, K) bf16

// ── Step 3: Compute grad_logits (NEW — softmax Jacobian) ──
// For each (j, o):
//   dL/dlogits[j, o, k] = dL/dW[j, o] * dW/dlogits[j, o, k]
//   where dW/dlogits[j, o, k] = palette[g, k] * P[j, o, k] * (delta_{k,k'} - P[j, o, k'])
//                               = palette[g, k] * P[j, o, k] * (I[k==k'] - P[j, o, k'])
//
// Simplified (softmax Jacobian-vector product):
//   Let d = grad_W[j, o]  (scalar, gradient w.r.t. this weight)
//   Let p = P[j, o, :]    (4-vector, probabilities)
//   Let c = palette[g, :] (4-vector, LUT entries)
//
//   dW/dlogits_k = c[k] * p[k] * (1 - p[k])            // diagonal term
//                - c[k'] * p[k'] * p[k]  for k' != k   // off-diagonal
//
//   dL/dlogits_k = d * c[k] * p[k] * (1 - p[k])
//                - Σ_{k'} d * c[k'] * p[k'] * p[k]
//              = d * p[k] * (c[k] - Σ_{k'} c[k'] * p[k'])
//              = d * p[k] * (c[k] - W[j, o])             // W[j,o] = Σ c[k']*p[k']
//
// So: grad_logits[j, o, k] = grad_W[j, o] * P[j, o, k] * (palette[g, k] - W[j, o])
//
// This is BEAUTIFUL — only needs grad_W, P, palette, and W (which we already have).

// ── Step 4: Compute grad_palette (weighted scatter_add) ──
// grad_palette[g, k] = Σ_{j, o in group g} grad_W[j, o] * P[j, o, k]
// This is similar to the hard kernel's scatter_add, but weighted by P[j, o, k]
// instead of a 0/1 indicator. Use fp32 atomicAdd.
```

### Kernel Decomposition

Reuse existing kernels where possible:

| Sub-kernel | Reuse from hard kernel? | Changes |
|---|---|---|
| `bwd_grad_x` | ✅ Yes (TC variant) | Reconstruct W from P+palette instead of indices+palette |
| `bwd_grad_W` (= x.T @ grad_y) | ✅ Yes (TC matmul) | None — pure matmul |
| `bwd_grad_logits` | ❌ NEW | Softmax Jacobian-vector product per element |
| `bwd_grad_palette` | ⚠️ Modified | Weighted scatter_add (P-weighted instead of 0/1) |
| `bwd_grad_bias` | ✅ Yes | None — same reduction |

### `bwd_grad_logits` Kernel Design

```
Grid: (cdiv(K, BK), cdiv(N, BN))
Block: (16, 16) = 256 threads

Per thread (tk, tn):
  j = blockIdx.x * BK + tk
  o = blockIdx.y * BN + tn
  if j >= K or o >= N: return

  g = o // group_size

  // Load grad_W[j, o] (computed by bwd_grad_W kernel, stored in global mem)
  float dW = grad_W[j * N + o];

  // Load P[j, o, 0..3] from 4 planes
  float p0 = P_plane0[j * N + o];
  float p1 = P_plane1[j * N + o];
  float p2 = P_plane2[j * N + o];
  float p3 = P_plane3[j * N + o];

  // Load palette[g, 0..3]
  float c0 = palette[g * 4 + 0];
  float c1 = palette[g * 4 + 1];
  float c2 = palette[g * 4 + 2];
  float c3 = palette[g * 4 + 3];

  // W[j, o] = Σ c[k] * p[k]
  float W_val = c0*p0 + c1*p1 + c2*p2 + c3*p3;

  // grad_logits[j, o, k] = dW * p[k] * (c[k] - W_val)
  grad_logits_plane0[j * N + o] = __float2half(dW * p0 * (c0 - W_val));
  grad_logits_plane1[j * N + o] = __float2half(dW * p1 * (c1 - W_val));
  grad_logits_plane2[j * N + o] = __float2half(dW * p2 * (c2 - W_val));
  grad_logits_plane3[j * N + o] = __float2half(dW * p3 * (c3 - W_val));
```

This kernel is **memory-bound** (loads P + grad_W, stores grad_logits) but trivially parallel — each element is independent. Expected ~0.5ms for the largest (9216×2560) shape.

### `bwd_grad_palette` (Modified)

Same structure as the hard kernel's `bwd_grad_palette`, but the scatter is weighted:

```
// Hard (existing):
//   grad_palette[g, indices[j, o]] += dW[j, o]
//   (0/1 indicator — weight goes to exactly 1 palette entry)

// Soft (NEW):
//   grad_palette[g, k] += dW[j, o] * P[j, o, k]
//   (weighted — each weight contributes to ALL 4 palette entries, weighted by probability)
```

Implementation: for each (j, o) in tile, compute 4 atomic adds (one per k) instead of 1. The atomic adds are to fp32 (same as hard kernel).

---

## Python Wrapper Changes

### New autograd Function: `CUDAFusedLUTLinearSoft`

```python
class CUDAFusedLUTLinearSoft(torch.autograd.Function):
    """Soft forward with Gumbel-Softmax relaxation.

    Forward: W = Σ_k P[k] * palette[k]  (soft, differentiable)
    Backward: grad_x, grad_logits, grad_palette, grad_bias
    """
    @staticmethod
    def forward(ctx, x, palette, logits, bias, group_size, tau, hard_eval=False):
        # x: (M, K) bf16
        # palette: (G, 4) bf16
        # logits: (K, N, 4) fp16 — stored as 4 (K, N) planes
        # tau: float — temperature
        # hard_eval: bool — if True, use straight-through (hard forward, soft backward)

        if hard_eval:
            # Straight-through Gumbel-Softmax:
            # Forward: hard argmax (discrete indices)
            # Backward: use soft gradients (STE)
            indices = logits.argmax(dim=-1).to(torch.int8)
            y = fused_lut_linear_hard_fwd(x, palette, indices, bias, group_size)
            # Save P for backward (compute soft P even in hard forward)
            P = softmax((logits + gumbel_noise) / tau)
            ctx.save_for_backward(x, palette, P)
        else:
            # Full soft forward:
            y, P = fused_lut_linear_soft_fwd(x, palette, logits, bias, group_size, tau)
            ctx.save_for_backward(x, palette, P)

        ctx.group_size = group_size
        ctx.tau = tau
        ctx.has_bias = bias is not None
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, palette, P = ctx.saved_tensors
        grad_x, grad_logits, grad_palette, grad_bias = fused_lut_linear_soft_bwd(
            grad_y, x, palette, P, ctx.group_size
        )
        return grad_x, grad_palette, grad_logits, grad_bias if ctx.has_bias else None, None, None, None
```

### PalettizedLinear Changes

```python
class PalettizedLinear(nn.Module):
    def __init__(self, ...):
        # Existing: palette, indices (frozen), bias
        # NEW: index_logits (trainable)
        self.index_logits = nn.Parameter(
            torch.zeros(K, N, 4, dtype=torch.float16, device=device)
        )
        # Initialize: one-hot from current indices
        with torch.no_grad():
            for k in range(4):
                self.index_logits.data[:, :, k] = -10.0
            # Set current index to logit=10
            self.index_logits.data.scatter_(-1,
                self.indices.long().unsqueeze(-1), 10.0)

    def forward(self, x):
        if self.training and self.use_soft:
            # Soft forward with Gumbel-Softmax
            y = CUDAFusedLUTLinearSoft.apply(
                x_flat, self.palette, self.index_logits,
                self.bias, self.group_size, self.tau,
                hard_eval=(self.tau < 0.01)  # STE when temperature is very low
            )
        else:
            # Hard forward (eval or when soft is disabled)
            y = CUDAFusedLUTLinear.apply(
                x_flat, self.palette, self.indices_int8,
                self.bias, self.group_size
            )
        return y
```

---

## Memory Layout: Logits Storage

The `(K, N, 4)` logits tensor must be stored as **4 contiguous `(K, N)` planes** for efficient GPU access:

```python
# Instead of: logits shape (K, N, 4) — bad for coalescing (stride-4 access)
# Use: logits shape (4, K, N) — 4 contiguous planes, each (K, N)
#      OR: 4 separate Parameters: logits_0, logits_1, logits_2, logits_3

# Recommended: store as (4, K, N) and access logits[k] for plane k
self.index_logits = nn.Parameter(
    torch.zeros(4, K, N, dtype=torch.float16, device=device)
)
```

This way, the CUDA kernel loads each plane with coalesced 16-byte `int4` loads — same pattern as the existing x tile loading.

### Memory Budget

| Component | Shape | Dtype | Size (largest Linear: 9216×2560) | ×43 Linears |
|---|---|---|---|---|
| `index_logits` | (4, K, N) | fp16 | 4 × 9216 × 2560 × 2 = 189 MB | 8.1 GB |
| `P` (saved during forward) | (4, K, N) | fp16 | 189 MB | 8.1 GB (transient — freed after backward) |
| `grad_logits` | (4, K, N) | fp16 | 189 MB | 8.1 GB (transient) |

**Total persistent extra: ~8.1 GB** (logits parameter).
**Peak transient extra: ~16.2 GB** (P + grad_logits during backward).

On 24GB L4 with ~18 GB currently used, this is **too tight**. Mitigations:

1. **Only use soft indices for the largest Linears** (gate_proj, up_proj, down_proj — 6 Linears per super-block instead of all 25). This reduces to ~2 GB extra.
2. **Use int8 logits with per-group scale** (quantize logits to int8). Halves memory to ~4 GB.
3. **Gradient checkpointing** on logits — recompute P during backward instead of saving it.
4. **Reduce batch_size from 8 to 4** — halves activation memory, freeing room for logits.

---

## Test Plan

### Correctness Tests

Add tests to `test_correctness.py`:

```python
def test_soft_forward_vs_reference():
    """Soft forward must match PyTorch reference (soft W construction)."""
    # Reference:
    P = softmax((logits + gumbel) / tau)
    W_ref = sum(P[k] * palette[k] for k in range(4))
    y_ref = x @ W_ref

    # CUDA:
    y_cuda, P_cuda = fused_lut_linear_soft_fwd(x, palette, logits, bias, GS, tau)
    assert torch.allclose(y_ref, y_cuda, atol=1e-3, rtol=1e-3)

def test_grad_logits_vs_finite_diff():
    """grad_logits must match finite-difference estimate."""
    # Perturb one logit element, measure loss change, compare to analytic gradient
    eps = 1e-3
    logits_plus = logits.clone(); logits_plus[j, o, k] += eps
    logits_minus = logits.clone(); logits_minus[j, o, k] -= eps
    loss_plus = compute_loss(fwd(x, logits_plus))
    loss_minus = compute_loss(fwd(x, logits_minus))
    fd_grad = (loss_plus - loss_minus) / (2 * eps)
    analytic_grad = grad_logits[j, o, k]
    assert abs(fd_grad - analytic_grad) / max(abs(fd_grad), 1e-8) < 1e-2

def test_grad_palette_soft_vs_hard():
    """When P is one-hot, soft grad_palette should match hard grad_palette."""
    # Set logits so that P is exactly one-hot (tau → 0)
    logits_hard = torch.full((K, N, 4), -1e6)
    logits_hard[:, :, current_indices] = 1e6
    P = softmax(logits_hard / 0.001)  # approximately one-hot
    grad_palette_soft = soft_bwd(..., P, ...)
    grad_palette_hard = hard_bwd(..., indices, ...)
    assert torch.allclose(grad_palette_soft, grad_palette_hard, atol=1e-4)
```

### Performance Tests

Benchmark soft vs hard vs reference:

```python
# Shapes to test (same as existing benchmark):
# (2560, 2560), (2560, 8192), (2560, 9216), (9216, 2560)
# M = 1024 (batch_seq = 8 × 128)

# Expected: soft kernel ~5-10% slower than hard kernel
# (due to loading 4 logit planes instead of 1 index plane)
# But still 5-10× faster than PyTorch eager soft path
```

---

## Implementation Order

1. **Write `fused_lut_linear_soft_fwd_kernel`** — soft forward with Gumbel-Softmax
   - Reuse TC tiling from existing fwd kernel
   - Add Gumbel noise LCG
   - Add softmax computation (4 elements, trivial)
   - Add W construction via weighted sum
   - Output: y + P (saved for backward)

2. **Write `fused_lut_linear_soft_bwd_grad_x_kernel`** — backward grad_x
   - Reuse TC tiling from existing bwd_grad_x
   - Reconstruct W from P+palette (instead of indices+palette)

3. **Write `fused_lut_linear_soft_bwd_grad_logits_kernel`** — backward grad_logits
   - NEW kernel: softmax Jacobian-vector product
   - `grad_logits[j, o, k] = grad_W[j, o] * P[j, o, k] * (palette[g, k] - W[j, o])`
   - Memory-bound, embarrassingly parallel

4. **Write `fused_lut_linear_soft_bwd_grad_palette_kernel`** — modified grad_palette
   - Same structure as hard kernel's scatter_add
   - But weighted by P[j, o, k] instead of 0/1 indicator

5. **Reuse `fused_lut_linear_bwd_grad_bias_kernel`** — no changes needed

6. **Add launchers + C++ wrapper** — same pattern as existing

7. **Add `CUDAFusedLUTLinearSoft` autograd Function** in Python

8. **Test correctness** against PyTorch soft reference

9. **Benchmark** — verify <10% overhead vs hard kernel

---

## Key Design Decisions

1. **Logits stored as `(4, K, N)` fp16 planes** — coalesced access, each plane loaded as contiguous tile
2. **Gumbel noise via LCG** (not curand) — deterministic per (thread, step), no library dependency
3. **Straight-through mode when tau < 0.01** — hard forward + soft backward (STE). This is the transition point where the soft kernel converges to the hard kernel.
4. **P saved for backward** — must store `(4, K, N)` fp16 during forward. Use gradient checkpointing if memory is tight.
5. **grad_W computed as intermediate** — `x.T @ grad_y` via TC matmul. Stored in global memory temporarily, then consumed by grad_logits and grad_palette kernels.
6. **Dual-mode PalettizedLinear** — `self.training` flag switches between soft (train) and hard (eval) kernels. The hard kernel is used for eval to avoid Gumbel noise affecting metrics.

---

## Expected Performance

| Component | Hard kernel | Soft kernel (expected) | Overhead |
|---|---|---|---|
| FWD (per layer, 9216×2560) | 4.0 ms | ~4.5 ms | +12% (4 logit planes vs 1 index) |
| BWD grad_x | 2.0 ms | ~2.0 ms | 0% (same TC matmul) |
| BWD grad_W (x.T @ grad_y) | (fused in grad_palette) | ~2.0 ms | NEW (needed for grad_logits) |
| BWD grad_logits | N/A | ~0.5 ms | NEW (memory-bound, parallel) |
| BWD grad_palette | 4.4 ms | ~5.0 ms | +14% (4 weighted atomics vs 1) |
| **Total per layer** | **10.4 ms** | **~14.0 ms** | **+35%** |
| **×43 layers** | **447 ms** | **~602 ms** | **+35%** |

At 1.6 tps (hard kernel), soft kernel would give ~1.2 tps. Still 3× faster than the original PyTorch path (0.4 tps).

---

## Deliverables

1. Updated `fused_lut_kernel.cu` with 5 new kernels:
   - `fused_lut_linear_soft_fwd_kernel`
   - `fused_lut_linear_soft_bwd_grad_x_kernel` (modified from existing)
   - `fused_lut_linear_soft_bwd_grad_logits_kernel` (NEW)
   - `fused_lut_linear_soft_bwd_grad_palette_kernel` (modified from existing)
   - (reuse existing `fused_lut_linear_bwd_grad_bias_kernel`)

2. Updated `fused_lut_linear_cuda.py` with:
   - `CUDAFusedLUTLinearSoft` autograd Function
   - `fused_lut_linear_soft()` functional interface

3. Updated `test_correctness.py` with soft kernel tests

4. Updated `benchmark.py` with soft kernel benchmarks

5. Updated `qwen_model.py` with dual-mode `PalettizedLinear`:
   - Training mode: soft kernel + trainable logits
   - Eval mode: hard kernel + frozen indices (argmax of logits)

---

## References

- GSQ (Gumbel-Softmax for LLM quant): arXiv:2604.18556, https://github.com/IST-DASLab/GSQ
- LLT (Learnable Lookup Table): CVPR 2022, https://github.com/SYSU-SAIL/LLT
- Gumbel-Softmax: arXiv:1611.01144 (Jang et al.)
- FLUTE (our current hard kernel base): arXiv:2407.10960
- Knowledge base: `/home/z/my-project/download/trainable_indices_kb.md`
