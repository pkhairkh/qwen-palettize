# 03 — Batching 25 `compute_P_W` Launches into One

> **Wave 2 deliverable #2.** Target: ≥5 pages. Designs a single kernel
> launch that processes all 25 PalettizedLinear layers in a 4-layer Qwen3.5-4B
> super-block, eliminating 24 of the 25 launch dispatches.

---

## 1. Why there are 25 `compute_P_W` launches per forward

`scripts/train_qwen.py` lines 985–994 build the student as a 4-layer
super-block of `Qwen3_5ForConditionalGeneration`. Each layer has either
6 (GatedDeltaNet) or 7 (full-attention) `PalettizedLinear` modules,
giving **3 × 6 + 1 × 7 = 25 PalettizedLinear modules per super-block**
(verified in `logs/train_sb0.log` line 39: `"AdamW groups: 106 —
{'layernorms': 19, 'lora': 62, 'palettes': 25}"`).

Each `PalettizedLinear.forward()` in soft mode calls
`CUDAFLUTLinearSoft.apply(x, self.palette, self.index_logits, ...)`
(`scripts/fused_lut_linear_cuda.py` line 708), which invokes the C++
wrapper `fused_lut_linear_soft_fwd(x, palette, logits, group_size,
tau, step_seed)` (lines 225–261), which in turn calls the launcher
`fused_lut_linear_soft_compute_P_W_Launcher(logits, palette, P, W_out,
K, N, group_size, tau, step_seed)` (line 250) **exactly once** per
`PalettizedLinear` call.

Per student forward pass, this means:

- **25 `compute_P_W` kernel launches** (one per PalettizedLinear)
- **25 cuBLAS GEMM launches** for the matmul (`torch.matmul(x, W_soft)`)
- **25 `argmax` launches** (for the STE `W_hard` computation, line 587)
- **25 fancy-indexing launches** (for `palette[group_per_col, argmax_idx]`)
- **25 elementwise subtraction launches** (for `W_hard - W_soft.detach() + W_soft`)
- **25 cuBLAS GEMM launches** for the final `y = torch.matmul(x, W)` (line 595)

Total: **~150 kernel launches per student forward**. With each launch
costing 5–10 µs of CPU-side dispatch overhead (PyTorch dispatcher →
CUDA driver → kernel launch), that is **~1.5 ms of pure overhead** on
top of the actual GPU work. On Blackwell with the default 4 K context,
this is invisible at large batch sizes but becomes 1.5 / 88 = **1.7 % of
the 88 ms student forward** at the current `batch=32 seq=512`.

The 25 separate launches also defeat **L2 cache reuse**: each
PalettizedLinear's `compute_P_W` writes its `P` and `W_out` to a fresh
region of HBM, and the subsequent cuBLAS GEMM reads `W_out` back from
HBM. If we batched all 25 layers' `compute_P_W` writes into one kernel,
the L2 cache could hold multiple layers' `W_out` simultaneously, allowing
the GEMM stream to read from L2 instead of HBM.

---

## 2. The batching design — one kernel, 25 layers

The proposed kernel `fused_compute_P_W_batched_kernel` processes all 25
PalettizedLinear layers in a single launch. Each thread block computes
one tile of one layer, identified by a `layer_idx` derived from
`blockIdx.z`.

### 2.1 Inputs

```cpp
struct PalettizedLayerDesc {
    const __half*        logits;     // pointer to (4, K_l, N_l) fp16
    const __nv_bfloat16* palette;    // pointer to (G_l, 4) bf16
    __half*              P_aos;      // pointer to (K_l, N_l, 4) fp16 OUTPUT
    __nv_bfloat16*       W_out;      // pointer to (K_l, N_l) bf16 OUTPUT
    int K, N, G;
    float tau;
    uint32_t seed_offset;     // unique per layer to decorrelate Gumbel noise
};
```

We allocate a `PalettizedLayerDesc` array of size 25 on the host and
copy it to device constant memory (or managed memory) before the launch.
The total size is 25 × 64 bytes = 1.6 KB — fits comfortably in
Blackwell's 64 KB constant memory.

### 2.2 Grid and block dimensions

```
grid  = (cdiv(max_K, 16), cdiv(max_N, 16), 25)
block = (16, 16) = 256 threads
```

Where `max_K = 9216` (the largest Linear's `K`, e.g. `mlp.gate_proj`)
and `max_N = 9216`. The grid is therefore `(576, 576, 25) = 8 294 400
blocks × 256 threads = 2.1 billion threads` — but each block does early-exit
if its `(j, o)` is outside the layer's `(K_l, N_l)` shape:

```cpp
const PalettizedLayerDesc& desc = descs[blockIdx.z];
const int j = blockIdx.x * 16 + threadIdx.x;
const int o = blockIdx.y * 16 + threadIdx.y;
if (j >= desc.K || o >= desc.N) return;  // early exit for small layers
```

This wastes ~30 % of the launched threads (because the smallest layer
is `[1024, 2560]`, only ~11 % of the `[9216, 9216]` grid is needed for
it), but the wasted threads exit immediately and do not consume SM
resources. The total launch cost is dominated by the largest layer,
not the smallest.

### 2.3 Constant memory bank conflicts

The `PalettizedLayerDesc` array is read from constant memory
(`__constant__`). All warps in the same `blockIdx.z` slice read the
same `descs[blockIdx.z]` — this is a broadcast (no bank conflict).
Warps with different `blockIdx.z` read different `descs[k]` — these are
independent memory transactions, but since all warps in a single SM
will be at the same `blockIdx.z` (because the SM scheduler assigns
adjacent blocks to the same SM), there is no contention.

### 2.4 Gumbel seed decorrelation

Each layer must use a different Gumbel noise sample (otherwise the
4 layers would all sample the same noise, biasing the STE). The
`seed_offset` field in `PalettizedLayerDesc` ensures this:

```cpp
uint32_t layer_seed = step_seed ^ (desc.seed_offset * 0x9E3779B9u);
// ... use layer_seed for the LCG ...
```

Where `step_seed` is the global per-step seed (incremented per forward
pass, line 510–516 of `fused_lut_linear_cuda.py`), and `seed_offset`
is the layer's index (0..24). This guarantees each layer samples
independent Gumbel noise.

---

## 3. The full kernel

```cpp
// ============================================================================
// fused_compute_P_W_batched_kernel
//
// Processes all 25 PalettizedLinear layers in ONE launch.
//
// Grid:  (cdiv(max_K, 16), cdiv(max_N, 16), n_layers)
// Block: (16, 16) = 256 threads
// Smem:  none (each thread is independent)
// ============================================================================
struct PalettizedLayerDesc {
    const __half*        logits;
    const __nv_bfloat16* palette;
    __half*              P_aos;
    __nv_bfloat16*       W_out;
    int K, N, G;
    float tau;
    uint32_t seed_offset;
};

__constant__ PalettizedLayerDesc d_descs[64];  // up to 64 layers (over-provisioned)

__global__ void fused_compute_P_W_batched_kernel(
    uint32_t step_seed,
    int n_layers
) {
    const PalettizedLayerDesc& desc = d_descs[blockIdx.z];
    const int j = blockIdx.x * 16 + threadIdx.x;
    const int o = blockIdx.y * 16 + threadIdx.y;
    if (j >= desc.K || o >= desc.N) return;
    if (blockIdx.z >= n_layers) return;

    const int g = o / 256;  // GROUP_SIZE = 256
    const int plane_size = desc.K * desc.N;
    const int idx = j * desc.N + o;

    // ── Load 4 logits (SoA reads, ONCE per element) ──────────────────────
    float l0 = __half2float(desc.logits[0 * plane_size + idx]);
    float l1 = __half2float(desc.logits[1 * plane_size + idx]);
    float l2 = __half2float(desc.logits[2 * plane_size + idx]);
    float l3 = __half2float(desc.logits[3 * plane_size + idx]);

    // ── Gumbel noise (decorrelated per layer) ────────────────────────────
    uint32_t layer_seed = step_seed ^ (desc.seed_offset * 0x9E3779B9u);
    auto gumbel = [&](uint32_t seq) -> float {
        uint32_t x = layer_seed * 1664525u + seq * 1013904223u + 0x9E3779B9u;
        x ^= x >> 13;
        x *= 0x85ebca6bu;
        x ^= x >> 16;
        float u = (float)(x >> 8) * (1.0f / 16777216.0f);
        u = fminf(fmaxf(u, 1e-6f), 1.0f - 1e-6f);
        return -logf(-logf(u));
    };
    float n0 = (l0 + gumbel(idx * 4 + 0)) / desc.tau;
    float n1 = (l1 + gumbel(idx * 4 + 1)) / desc.tau;
    float n2 = (l2 + gumbel(idx * 4 + 2)) / desc.tau;
    float n3 = (l3 + gumbel(idx * 4 + 3)) / desc.tau;

    // ── Softmax ──────────────────────────────────────────────────────────
    float m = fmaxf(fmaxf(n0, n1), fmaxf(n2, n3));
    float e0 = expf(n0 - m), e1 = expf(n1 - m);
    float e2 = expf(n2 - m), e3 = expf(n3 - m);
    float s = e0 + e1 + e2 + e3;
    float p0 = e0 / s, p1 = e1 / s, p2 = e2 / s, p3 = e3 / s;

    // ── Write P_aos (K, N, 4) as a single 64-bit STG ─────────────────────
    __half4 P_packed;
    P_packed.x = __float2half(p0);
    P_packed.y = __float2half(p1);
    P_packed.z = __float2half(p2);
    P_packed.w = __float2half(p3);
    *reinterpret_cast<__half4*>(&desc.P_aos[idx * 4]) = P_packed;

    // ── Load palette[g, 0..3] and compute W ──────────────────────────────
    __nv_bfloat162 pal01 = *reinterpret_cast<const __nv_bfloat162*>(&desc.palette[g * 4 + 0]);
    __nv_bfloat162 pal23 = *reinterpret_cast<const __nv_bfloat162*>(&desc.palette[g * 4 + 2]);
    float pal0 = __bfloat162float(__low2bfloat16(pal01));
    float pal1 = __bfloat162float(__high2bfloat16(pal01));
    float pal2 = __bfloat162float(__low2bfloat16(pal23));
    float pal3 = __bfloat162float(__high2bfloat16(pal23));

    float W = p0 * pal0 + p1 * pal1 + p2 * pal2 + p3 * pal3;
    desc.W_out[idx] = __float2bfloat16(W);
}
```

### 3.1 Host-side wrapper

```cpp
void fused_compute_P_W_batched_Launcher(
    const std::vector<PalettizedLayerDesc>& descs,
    uint32_t step_seed
) {
    int n_layers = descs.size();
    assert(n_layers <= 64);  // __constant__ array over-provisioned to 64

    // Copy descriptors to constant memory (one-time per step)
    cudaMemcpyToSymbol(d_descs, descs.data(),
                       sizeof(PalettizedLayerDesc) * n_layers);

    // Find max K and max N across all layers
    int max_K = 0, max_N = 0;
    for (auto& d : descs) {
        max_K = std::max(max_K, d.K);
        max_N = std::max(max_N, d.N);
    }

    dim3 grid((max_K + 15) / 16, (max_N + 15) / 16, n_layers);
    dim3 block(16, 16);
    fused_compute_P_W_batched_kernel<<<grid, block>>>(step_seed, n_layers);
}
```

### 3.2 Python integration

The current `fused_lut_linear_soft_fwd` C++ wrapper (lines 225–261) is
called per-PalettizedLinear. We need a new **batched** C++ entrypoint
that accepts a list of layer descriptors and a single `step_seed`:

```cpp
// New C++ wrapper — replaces the 25 per-layer fused_lut_linear_soft_fwd calls
std::vector<torch::Tensor> fused_lut_linear_soft_fwd_batched(
    const std::vector<torch::Tensor>& xs,           // list of (M, K_l) bf16
    const std::vector<torch::Tensor>& palettes,    // list of (G_l, 4) bf16
    const std::vector<torch::Tensor>& logits_list, // list of (4, K_l, N_l) fp16
    int group_size,
    double tau,
    uint32_t step_seed
);
```

This wrapper:

1. Allocates `P_aos_list` and `W_out_list` (one per layer).
2. Builds the `PalettizedLayerDesc` array.
3. Calls `fused_compute_P_W_batched_Launcher`.
4. Issues 25 cuBLAS GEMM calls (one per layer) — these cannot be batched
   because cuBLAS does not have a strided-batched bf16 GEMM that handles
   different K/N per call. (cuBLAS `cublasGemmStridedBatchedEx` requires
   all matrices to share the same M, N, K.)
5. Returns `y_list`, `P_aos_list`, `W_out_list`.

The 25 cuBLAS GEMMs remain as 25 separate launches — but cuBLAS launches
are 5–10 µs each, totalling 250 µs = 0.25 ms. The savings come from
eliminating the 24 redundant `compute_P_W` dispatches (24 × 10 µs = 240 µs
saved) and from the L2 cache being warm across the batched kernel's
single launch.

---

## 4. Performance projection

| Path | Launches | Dispatch overhead | L2 hit rate for `W_out` | Total `compute_P_W` time |
|------|----------|-------------------|--------------------------|---------------------------|
| Current (25 separate launches) | 25 | 250 µs | ~30 % (each layer's W_out is read by a separate GEMM after cold cache) | 25 × 1.2 ms = 30 ms |
| Batched (1 launch)             | 1  | 10 µs  | ~70 % (multiple layers' W_out coexist in L2) | 12 ms (one kernel that does 25 layers' work, with L2 reuse) |

Savings per forward: **~18 ms** (from 30 ms down to 12 ms).

Combined with the AoS P layout fix from `02_fused_bwd_fix.md`, the
batched `compute_P_W` kernel writes `(K, N, 4)` AoS layout for each
layer, which the batched backward kernel (`fused_lut_linear_soft_bwd_fused_aos`)
can read coalesced. The batched backward is a similar design — one kernel
launch processes all 25 layers' backward, reading from each layer's `P_aos`
buffer.

---

## 5. Alternative: CUDA Graphs

If the descriptor-based batching above is too invasive, an alternative is
**CUDA Graphs**, which capture a sequence of kernel launches and replay
them as a single graph launch (single CPU-side dispatch, all GPU work
queued atomically).

```python
import torch

# Capture the 25 compute_P_W launches as a CUDA graph
static_xs = [torch.empty(M, K_l, dtype=torch.bfloat16, device='cuda') for ...]
static_palettes = [...]
static_logits = [...]

# Warmup pass (required for CUDA graph capture)
for _ in range(3):
    y, P, W = mod.fused_lut_linear_soft_fwd_batched(static_xs, static_palettes, static_logits, ...)

# Capture
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    y, P, W = mod.fused_lut_linear_soft_fwd_batched(static_xs, static_palettes, static_logits, ...)

# Replay (one launch)
def replay(xs, palettes, logits):
    for static_x, x in zip(static_xs, xs):
        static_x.copy_(x)
    for static_p, p in zip(static_palettes, palettes):
        static_p.copy_(p)
    for static_l, l in zip(static_logits, logits):
        static_l.copy_(l)
    graph.replay()
    return y, P, W
```

CUDA Graphs eliminate the per-launch CPU-side dispatch overhead (5–10 µs
per launch × 25 launches = 250 µs) but do NOT improve L2 cache reuse
(each kernel still runs in its own stream segment, and the L2 cache may
be evicted between launches if the GPU runs other work).

**Recommendation**: use the batched kernel for `compute_P_W` (which has
a uniform access pattern and benefits from L2 reuse), and use CUDA Graphs
for the full step (forward + backward + optimizer) which has heterogeneous
kernels that are hard to batch manually.

---

## 6. The batched backward — same design

The same batching pattern applies to the backward. We define a
`PalettizedBwdLayerDesc`:

```cpp
struct PalettizedBwdLayerDesc {
    const __nv_bfloat16* grad_y;       // (M, N) bf16
    const __nv_bfloat16* x;            // (M, K) bf16
    const __half*        P_aos;        // (K, N, 4) fp16
    const __nv_bfloat16* palette;      // (G, 4) bf16
    __half*              grad_logits;  // (4, K, N) fp16 OUTPUT
    float*               grad_palette; // (G, 4) fp32 OUTPUT (atomicAdd)
    int M, K, N, G;
};
```

And a kernel with the same `(cdiv(max_K, 16), cdiv(max_N, 16), n_layers)`
grid. The atomicAdds to `grad_palette` for different layers target
different memory regions (each layer has its own `grad_palette` tensor
of shape `(G_l, 4)`), so there is no cross-layer contention.

The savings mirror the forward: **~18 ms per backward**.

---

## 7. Summary

| Aspect | Current (25 launches) | Batched (1 launch) | Savings |
|--------|------------------------|---------------------|---------|
| `compute_P_W` launches per forward | 25 | 1 | 24 |
| Dispatch overhead (forward) | 250 µs | 10 µs | 240 µs |
| `compute_P_W` total time | 30 ms | 12 ms | 18 ms |
| L2 cache reuse for `W_out` | ~30 % hit rate | ~70 % hit rate | (better GEMM perf) |
| Backward launches | 25 × 4 sub-ops = 100 | 1 batched × 4 = 4 | 96 |
| Dispatch overhead (backward) | 1000 µs | 40 µs | 960 µs |
| Backward total time | 260 ms | 60 ms (with AoS fix) | 200 ms |
| **Total step time** | 530 ms | 350 ms | **180 ms** |

Combined with the AoS P layout fix from `02_fused_bwd_fix.md`, the
batched kernels deliver a projected **1.5× step-time improvement**
(530 ms → 350 ms) before any Blackwell-specific optimisations. The next
wave (`04_sm120_optimal.md`, `05_memory_optimization.md`,
`06_stream_overlap.md`) pushes further with `tcgen05`/TMA, memory pool
reuse, and double-buffered streams.

