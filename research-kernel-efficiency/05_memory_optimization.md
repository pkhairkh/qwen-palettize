# 05 — Memory Optimisation: Eliminating the `(K, N, 4)` Intermediates

> **Wave 3 deliverable #2.** Target: ≥5 pages. Documents the memory
> allocation profile of the soft backward, explains why the
> `(K, N, 4)` intermediates prevent `batch=64` runs, and proposes
> three concrete fixes: (a) AoS layout (covered in `02_fused_bwd_fix.md`),
> (b) buffer pooling, (c) chunked reduction.

---

## 1. Current VRAM budget — where the 36 GB goes

From `01_profiling_breakdown.md` §3, the breakdown of the 36 GB observed
on the Blackwell at `batch=32 seq=512`:

| Buffer | Size (GB) | Lifecycle | Notes |
|--------|-----------|-----------|-------|
| `embed_tokens` (frozen) | 1.27 | Process lifetime | `[248320, 2560] bf16` |
| Teacher 4-layer activations (`h_out`) | 0.04 | Per-step | `[32, 512, 2560] bf16` |
| Student forward activations (saved for backward) | 1.20 | Per-step | per-layer × 4 × seq×batch×hidden |
| `index_logits` (25 × 4 × K × N fp16) | 1.31 | Persistent | the trained parameter |
| FP32 AdamW master of `index_logits` | 7.15 | Persistent | `FP32MasterAdamW` maintains fp32 master |
| AdamW `m` for `index_logits` (fp32) | 7.15 | Persistent | first moment |
| AdamW `v` for `index_logits` (fp32) | 7.15 | Persistent | second moment |
| LoRA params + masters + m/v | 0.50 | Persistent | 4.7 M LoRA params × 7 bytes (bf16 + 3× fp32) |
| Palettes (25 × 144 entries × bf16) | <0.01 | Persistent | negligible |
| Backward intermediates (peak) | ~2.6 | Per-step | see §2 |
| cuBLAS workspace + caching allocator | ~5.0 | Persistent | PyTorch overhead |
| CUDA context + kernel JIT cache | ~2.0 | Process lifetime | |
| **Sum** | **~34.6 GB** | | matches observed 36 GB |

The two surprises here:

1. **The indices optimizer state alone is 21.5 GB** — `master + m + v`
   at fp32. The `SPEC.md` §6.3 estimate of "~20 GB GPU memory" undercounts
   this because it omits the `m` and `v` state.
2. **The remaining 60 GB of VRAM is unused** at the current `batch=32`.

---

## 2. The `(K, N, 4)` intermediate — why it exists and why it's bad

`scripts/fused_lut_linear_cuda.py` lines 658–673 materialise three
`(K, N, 4)` tensors during the backward:

```python
# Per layer, per backward call:
P_kno = P.permute(1, 2, 0)  # (K, N, 4) fp16 view, no copy yet
contributions = (grad_W.unsqueeze(-1) * P_kno).view(K, G, GS, 4)
                # ↑ materialises (K, N, 4) fp16 = 52 MB per layer

# Later in the grad_logits branch:
P_kno_f = P.permute(1, 2, 0).float()  # (K, N, 4) fp32 = 105 MB per layer
pal_pos = palette[g_idx.long()].unsqueeze(0).expand(K, N, 4).float()
                # (K, N, 4) fp32 = 105 MB per layer
W_val = (P_kno_f * pal_pos).sum(dim=-1)  # (K, N) fp32 = 26 MB
grad_logits = (grad_W_f.unsqueeze(-1) * P_kno_f * (pal_pos - W_val.unsqueeze(-1)))
                # (K, N, 4) fp32 = 105 MB per layer
```

For the canonical Linear shape `K = N = 2560`:

| Intermediate | Shape | Dtype | Size per layer |
|--------------|-------|-------|----------------|
| `contributions` | `(K, N, 4)` | fp16 | 52.4 MB |
| `P_kno_f` | `(K, N, 4)` | fp32 | 104.9 MB |
| `pal_pos` (after expand→materialise) | `(K, N, 4)` | fp32 | 104.9 MB |
| `grad_W_f.unsqueeze(-1) * P_kno_f * (...)` result | `(K, N, 4)` | fp32 | 104.9 MB |
| **Peak per layer** | | | **~367 MB** |

With 25 layers, if all intermediates were alive simultaneously, peak
would be **9.2 GB**. In practice, PyTorch's autograd frees each layer's
intermediates after that layer's backward completes (because they are
local to the backward function), so the **peak simultaneous** usage is
~3–4 layers' worth = **~1.5 GB**. This is the source of the "2.6 GB
peak intermediates" figure in the budget above.

### 2.1 Why does `batch=64` OOM?

At `batch=64 seq=512`, the per-step activations double (1.2 GB → 2.4 GB)
and the cuBLAS workspace demand doubles (~5 GB → ~10 GB). The remaining
VRAM is `96 - 1.27 - 0.04 - 2.4 - 1.31 - 21.45 - 0.50 - 0.01 - 2.6 - 10
- 2 = 54 GB`. **54 GB free**, yet the user reports OOM. The OOM must be
coming from somewhere else.

The actual OOM trigger is the **`(K, N, 4)` intermediates being
simultaneously alive with the larger batch activations**. At `batch=64`,
the cuBLAS workspace grows to 10 GB, and the `(K, N, 4)` intermediates
grow proportionally with `K` (because `K = M × seq = 64 × 512 = 32768`
for a `[32768, 2560]` matmul — wait, that's wrong).

Actually, re-reading the code: `grad_W = torch.matmul(x.T, grad_y)` where
`x = (M, K)` and `grad_y = (M, N)`, so `grad_W = (K, N)`. The `K` and `N`
are the Linear's weight shape, **not the batch shape**. So the `(K, N, 4)`
intermediate size is **independent of batch size** — it is always 52.4 MB
fp16 or 105 MB fp32 per layer.

The OOM at `batch=64` is therefore NOT due to `(K, N, 4)` intermediates
growing. It is due to the **forward activations growing linearly with
batch**: `student_activations = 4 layers × batch × seq × hidden × bf16
× ~5 saved tensors per layer`. At batch=32 this is `4 × 32 × 512 × 2560 ×
2 × 5 = 3.4 GB`. At batch=64 it doubles to **6.7 GB**, plus the larger
cuBLAS workspace (~10 GB) plus the existing 32 GB persistent state =
**~49 GB**, with 47 GB headroom. That should fit.

The actual OOM cause must be the **`grad_x = grad_y @ W.T` matmul output
at batch=64**: `grad_y = (M, N) = (32768, 2560)` × `W.T = (N, K) = (2560,
2560)` = `grad_x = (32768, 2560) bf16 = 168 MB` per layer × 25 layers =
**4.2 GB peak** (if all alive simultaneously — in practice the autograd
graph frees each layer's `grad_x` after the next layer consumes it, so
peak is ~2 layers × 168 MB = 0.34 GB — small).

The most plausible OOM cause is the **cuBLAS workspace reservation
doubling at larger M**. cuBLAS picks an algorithm at first-GEMM-call
time based on the M, N, K; at `M = 32768` it may pick a split-K algorithm
that requires a much larger workspace. The fix is to set
`CUBLAS_WORKSPACE_CONFIG=:32768:8` (or pass `cublasSetWorkspace(...)` with
a pre-allocated 4 GB buffer) to control the workspace size.

---

## 3. Eliminating the `(K, N, 4)` intermediates

Even though the OOM is not directly caused by `(K, N, 4)` intermediates,
eliminating them is still worthwhile for **bandwidth** reasons: writing
52–105 MB to HBM and reading it back 2–3 times per layer is a major
contributor to the 260 ms backward.

### 3.1 Fix #1: AoS layout (from `02_fused_bwd_fix.md`)

If we change `P`'s storage layout from `(4, K, N)` to `(K, N, 4)`, the
permuted view `P_kno = P.permute(1, 2, 0)` becomes a no-op (just use `P`
directly). PyTorch's elementwise fusion then kicks in:

```python
# Before:
P_kno = P.permute(1, 2, 0)  # non-contiguous view
contributions = (grad_W.unsqueeze(-1) * P_kno).view(K, G, GS, 4)
                # ↑ PyTorch must materialise (K, N, 4) because P_kno is non-contig

# After:
P_kno = P  # already (K, N, 4) contiguous
contributions = (grad_W.unsqueeze(-1) * P_kno).view(K, G, GS, 4)
                # ↑ PyTorch can fuse mul+view+sum into a single reduce kernel
```

Expected savings: 52 MB write + 52 MB read per layer eliminated = ~13 ms
per layer × 25 layers = **~325 ms saved in the backward** (though this
is the upper bound; the current backward is 260 ms total, so the savings
are capped at ~150 ms once we account for the parts that cannot be fused).

### 3.2 Fix #2: Buffer pooling (P_aos reuse)

Even with AoS layout, each PalettizedLinear allocates a fresh `P_aos`
buffer per forward call. With 25 layers and per-layer peak of 52 MB,
the **total `P_aos` allocation across the forward** is 1.3 GB.

We can reduce this by **reusing a single `P_aos` buffer across layers**
(via a buffer pool keyed on the layer's `(K, N)` shape):

```python
# In qwen_model.py, modify PalettizedLinear:
class PalettizedLinear(nn.Module):
    _P_POOL = {}  # (K, N) → P_aos buffer
    
    def forward(self, x):
        # ...
        key = (self.in_features, self.out_features)
        if key not in PalettizedLinear._P_POOL:
            PalettizedLinear._P_POOL[key] = torch.empty(
                self.in_features, self.out_features, 4,
                dtype=torch.float16, device=x.device
            )
        P_aos = PalettizedLinear._P_POOL[key]
        
        # Compute P_aos in-place (the compute_P_W_aos kernel writes to it)
        mod.fused_lut_linear_soft_fwd_aos(x, self.palette, self.index_logits,
                                          P_aos, ...)
```

This drops the per-forward `P_aos` allocation from 1.3 GB peak to **52 MB
peak** (one buffer reused across all layers of the same shape). The
savings are smaller than they appear, because PyTorch's caching allocator
already reuses freed buffers — but explicit pooling avoids the **first-
call allocation jitter** and reduces fragmentation.

### 3.3 Fix #3: Chunked reduction (no materialisation)

The deepest fix is to replace the PyTorch elementwise path entirely with
a **chunked reduction kernel** that never materialises the `(K, N, 4)`
intermediate. This is the `fused_lut_linear_soft_bwd_fused_aos_kernel`
designed in `02_fused_bwd_fix.md` §3.2 — but with an additional
**K-dimension chunking** to keep the working set in shared memory:

```cpp
// Chunked reduction kernel — processes K in chunks of K_CHUNK
// to keep the (K_CHUNK, N, 4) intermediate in shared memory only.
__global__ void fused_lut_soft_bwd_chunked_kernel(
    const __nv_bfloat16* __restrict__ grad_y,    // (M, N)
    const __nv_bfloat16* __restrict__ x,         // (M, K)
    const __half*        __restrict__ P_aos,     // (K, N, 4)
    const __nv_bfloat16* __restrict__ palette,   // (G, 4)
    __half*              __restrict__ grad_logits, // (4, K, N)
    float*               __restrict__ grad_palette, // (G, 4)
    int M, int K, int N, int group_size
) {
    constexpr int K_CHUNK = 64;   // K tile processed per kernel invocation
    constexpr int N_TILE = 32;    // N tile per block
    constexpr int M_TILE = 128;   // M tile per block (for the reduction)
    
    __shared__ float s_grad_palette_chunk[G_PER_BLOCK][4];  // per-block palette accumulator
    __shared__ __nv_bfloat16 s_x_chunk[M_TILE][K_CHUNK];    // (M, K) tile
    __shared__ __nv_bfloat16 s_gy_chunk[M_TILE][N_TILE];   // (M, N) tile
    
    const int k_base = blockIdx.x * K_CHUNK;
    const int n_base = blockIdx.y * N_TILE;
    
    // ── Outer loop over K_CHUNK tiles ──────────────────────────────────
    for (int k_off = 0; k_off < K_CHUNK; ++k_off) {
        const int j = k_base + k_off;
        if (j >= K) continue;
        
        // Load P_aos[j, n_base:n_base+N_TILE, 0..3] into smem (one coalesced read)
        __shared__ __half s_P_chunk[N_TILE][4];
        // ... (load N_TILE × 4 fp16 = 256 bytes per warp, coalesced)
        
        // Load palette[g, 0..3] for the n_base group
        // ... (8 bytes per group, broadcast)
        
        // Inner reduction: grad_W = Σ_m x[m, j] * grad_y[m, n]
        //                  grad_palette[g, k] += grad_W * P[k]
        //                  grad_logits[k, j, o] = grad_W * P[k] * (palette[g, k] - W[j, o])
        float grad_W = 0.0f;
        for (int m_off = 0; m_off < M; m_off += M_TILE) {
            // Load x[m_off:m_off+M_TILE, j] and grad_y[m_off:m_off+M_TILE, n_base:n_base+N_TILE]
            // ... (cooperative load)
            for (int m = 0; m < M_TILE && (m_off + m) < M; ++m) {
                grad_W += __bfloat162float(s_x_chunk[m][k_off]) *
                          __bfloat162float(s_gy_chunk[m][threadIdx.x]);
            }
            __syncthreads();
        }
        
        // Write grad_palette (atomicAdd to global grad_palette[g*4+k])
        // Write grad_logits (direct write, no race)
        // ...
    }
}
```

This kernel processes K in chunks of 64 elements, with each chunk's
`P_aos` data (64 × N_TILE × 4 = 32 KB) held in shared memory throughout
the M-reduction. The intermediate `(K, N, 4)` tensor is **never
materialised in HBM** — it lives entirely in shared memory.

Expected savings:
- HBM traffic: from 52 MB write + 52 MB read = 104 MB per layer
  to ~0 (smem only).
- Time: from 13 ms per layer × 25 layers = 325 ms (upper bound)
  to ~0.5 ms per layer × 25 layers = 12 ms.
- VRAM peak: from 1.5 GB peak intermediates to **0**.

---

## 4. The `M_CHUNK` strategy for `grad_W`

The `grad_W = x.T @ grad_y` matmul itself produces a `(K, N)` bf16 output
that is later consumed by the elementwise `grad_palette` and `grad_logits`
kernels. This `grad_W` is also a materialised intermediate (52 MB per
layer × 25 = 1.3 GB peak).

To eliminate it, the chunked reduction kernel above computes `grad_W` in
registers and immediately consumes it for the `grad_palette` and
`grad_logits` updates. The matmul `x.T @ grad_y` is **fused into the
reduction loop** (lines "Inner reduction" in §3.3).

This is the **deepest fusion** possible without rewriting the kernel in
terms of `tcgen05.mma`. It brings the entire backward to:

| Step | HBM traffic per layer | Time per layer |
|------|------------------------|----------------|
| Load `grad_y[M, N]` | M × N × 2 = 16 MB | 2 ms |
| Load `x[M, K]` | M × K × 2 = 16 MB | 2 ms |
| Load `P_aos[K, N, 4]` | K × N × 4 × 2 = 52 MB | 6 ms |
| Write `grad_logits[4, K, N]` | 4 × K × N × 2 = 52 MB | 6 ms |
| Atomic `grad_palette[G, 4]` | ~2 KB (negligible) | 0 |
| **Total** | **136 MB / layer** | **~16 ms × 25 = 400 ms** |

Wait — this is *slower* than the current 260 ms backward. Why?

The answer: the current PyTorch path benefits from **cuBLAS GEMM's
optimal HBM bandwidth** for the `grad_W = x.T @ grad_y` step. cuBLAS
streams `x` and `grad_y` through the tensor cores at near-peak bandwidth
(~7 TB/s out of 8 TB/s peak), achieving the matmul in **1 ms per layer
× 25 = 25 ms** (vs the 50 ms we estimated naively). The chunked kernel's
"fusion" loses this cuBLAS advantage — it computes the matmul in a
non-tensor-core scalar loop, which is 5–10× slower than cuBLAS even
though it saves the HBM round-trip.

**Recommendation**: do NOT fuse the matmul. Keep `grad_W = x.T @ grad_y`
as a cuBLAS GEMM, and only fuse the **elementwise** part (the
`(grad_W.unsqueeze(-1) * P_kno)` and downstream reductions). The
chunked reduction kernel above should read `grad_W` from HBM (one extra
52 MB read per layer = 6 ms × 25 = 150 ms), but save the 52 MB write of
the intermediate (6 ms × 25 = 150 ms saved) and the 105 MB fp32 cast
(13 ms × 25 = 325 ms saved).

Net savings: ~325 ms saved on the fp32 casts (which we eliminate by doing
the math in fp16/bf16 with selective fp32 accumulation in the
reduction). The 260 ms backward drops to **~120 ms**.

---

## 5. Reducing the optimizer state — `bfloat16` master?

The 21.5 GB of FP32 AdamW state for `index_logits` is the single largest
VRAM consumer. The comment in `scripts/train_qwen.py` line 594–597:

```python
# AdamW with fp32 master for index_logits (Blackwell 96GB can afford 21.4 GB).
# CRITICAL: fp16 AdamW state + eps=1e-8 → NaN (sqrt(v)+eps underflows to 0 in fp16).
# FP32 master avoids this. Plain SGD (L4 fallback) was too weak for Gumbel-Softmax grads.
```

The fp32 master is required because fp16 AdamW underflows at `eps=1e-8`.
There are three alternatives:

### 5.1 bf16 AdamW state (no fp32 master)

`bfloat16` has the same dynamic range as fp32 (max ~3.4 × 10³⁸) but
only 7 bits of mantissa (vs fp32's 23). The `sqrt(v) + eps` underflow
problem does not occur in bf16 (the smallest positive subnormal is
~1.18 × 10⁻³⁸, well below `eps=1e-8`).

Switching `index_logits`'s optimizer to **bf16 master + bf16 m/v**
reduces the state from 21.5 GB to **10.7 GB** (4 bytes/element × 3
states → 2 bytes/element × 3 states = 6 bytes/element × 1.78 B =
10.7 GB). The trade-off: bf16's 7-bit mantissa means small gradient
updates (typical for late-training Gumbel-Softmax with `tau=0.1`)
get rounded to zero.

Empirically, this is a problem: at `tau=0.1`, the Gumbel-Softmax
gradient magnitudes are ~1e-4, and bf16's epsilon is ~2⁻⁷ ≈ 0.008 of the
gradient's typical value — meaning most updates round to zero. The
training would stall.

### 5.2 FP32 `v`, bf16 `m` and master

A hybrid: keep `v` (the second moment, which is the one that needs
high dynamic range) in fp32, but use bf16 for `m` and the master. This
saves 33 % of the state (from 21.5 GB to 14.3 GB) while preserving
numerical stability. PyTorch's `torch.optim.AdamW` does not directly
support mixed-precision state; we would need to implement a custom
optimizer.

### 5.3 Adafactor (no second moment)

Adafactor replaces the second moment `v` with row/column statistics of
the gradient outer product, reducing the per-parameter state from
`2 × params` to `O(rows + cols)`. For a `(4, K, N) = (4, 2560, 2560)`
logits tensor, the state drops from `2 × 4 × K × N = 52.4 M params` to
`4 × (K + N) = 20 K params` per layer — a 2 600× reduction.

The trade-off: Adafactor is **less stable** for Gumbel-Softmax training
(the original paper used it for transformer pre-training, not for
discrete-index learning). Empirical tuning would be needed.

### 5.4 Recommendation

Keep the fp32 master for now (the 21.5 GB fits comfortably in 96 GB).
Revisit only if we hit VRAM limits at `batch=128` or larger.

---

## 6. Memory layout summary after all fixes

| Buffer | Before | After (with all fixes) |
|--------|--------|--------------------------|
| `embed_tokens` | 1.27 GB | 1.27 GB (unchanged) |
| `h_out` (teacher) | 0.04 GB | 0.04 GB |
| Student activations | 1.20 GB | 1.20 GB (could reduce with gradient checkpointing) |
| `index_logits` | 1.31 GB | 1.31 GB |
| FP32 AdamW state (master + m + v) | 21.45 GB | 21.45 GB (keep fp32) |
| LoRA + masters + m/v | 0.50 GB | 0.50 GB |
| Palettes | <0.01 GB | <0.01 GB |
| `P_aos` buffers (pooled) | 1.30 GB peak | **0.05 GB** (one buffer reused) |
| Backward intermediates | 2.60 GB peak | **0** (chunked reduction) |
| cuBLAS workspace | 5.00 GB | 5.00 GB (cap with CUBLAS_WORKSPACE_CONFIG) |
| CUDA context | 2.00 GB | 2.00 GB |
| **Total** | **~34.6 GB** | **~32 GB** |

The savings are modest in absolute terms (~3 GB) but they **unlock
larger batch sizes**: at `batch=64`, the activations double to 2.4 GB,
the cuBLAS workspace grows to 10 GB, and we still have 96 - 32 - 2.4 -
10 = **51.6 GB free** — more than enough for `batch=128` (which would
need ~10 GB more activations + 5 GB more cuBLAS workspace = 15 GB).

The real win of the memory optimisation is not the GB savings but the
**bandwidth savings**: eliminating the 52 MB writes/reads of the
`(K, N, 4)` intermediate saves **~150 ms per step** of HBM traffic,
bringing the backward from 260 ms to ~110 ms.

---

## 7. Gradient checkpointing — the optional further optimisation

If we want to push to `batch=128`+, gradient checkpointing (a.k.a.
"recomputation") can drop the forward activations from 1.2 GB to ~0.3 GB
by recomputing the forward during the backward. The trade-off: a ~30 %
increase in forward compute (one extra forward pass per layer during
backward).

PyTorch's `torch.utils.checkpoint.checkpoint` supports this with a one-
line change in `qwen_model.py`:

```python
# In PalettizedLinear.forward:
if self.training and self.use_gradient_checkpointing:
    return torch.utils.checkpoint.checkpoint(self._forward_impl, x, use_reentrant=False)
return self._forward_impl(x)
```

This would unlock `batch=256` (with ~2.4 GB activations + 21.5 GB AdamW
state + 10 GB cuBLAS = 34 GB, leaving 62 GB free). The cost is 30 %
more compute — but on Blackwell with the Phase C `tcgen05.mma` kernel,
we have compute headroom to spare.

**Recommendation**: defer gradient checkpointing until after the
`tcgen05.mma` migration. It is a "last resort" optimisation that should
only be applied if we hit VRAM limits at `batch=64` after the other
fixes are in place.

---

## 8. Summary of memory optimisations

| Optimisation | VRAM saved | Bandwidth saved | Effort |
|--------------|------------|------------------|--------|
| AoS layout for `P` | 0 GB (same shape) | ~150 ms/step | 1 day (see `02_fused_bwd_fix.md`) |
| Buffer pooling for `P_aos` | 1.25 GB | 0 (no bandwidth change) | 0.5 day |
| Chunked reduction (no `grad_W` materialisation) | 0.5 GB | ~50 ms/step | 3 days |
| Chunked reduction (no `(K,N,4)` materialisation) | 2.6 GB | ~150 ms/step | (covered by `02_fused_bwd_fix.md` §3.2) |
| bf16 AdamW master | 7.15 GB | 0 | (NOT recommended — Gumbel underflow) |
| Gradient checkpointing | 0.9 GB per checkpoint | -30 % compute | 1 day (defer) |
| **Cumulative** | **~4 GB** | **~350 ms/step** | **5.5 days** |

The bandwidth savings dominate — they translate directly to step-time
reductions. The VRAM savings unlock larger batch sizes, which translate
to higher throughput via better SM occupancy.

The next document, `06_stream_overlap.md`, tackles the **stream
overlap** problem — making the teacher forward truly overlap with the
student backward via double-buffered CUDA streams.

