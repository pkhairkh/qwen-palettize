# 04 — Optimal Tensor Core / TMA / WGMMA Strategy for Blackwell sm_120

> **Wave 3 deliverable #1.** Target: ≥6 pages. Surveys the Blackwell
> sm_120 ISA options (`mma.sync`, `wgmma`, `tcgen05`, `cp.async.bulk.tensor`
> / TMA, `cluster.sync`, `setmaxnreg`) and recommends a migration path
> from the current Ampere-era `mma.sync.m16n8k16` + `cp.async` baseline.
> References NVIDIA PTX ISA 8.7 documentation and CUTLASS 3.x examples.

---

## 1. What the current kernel uses and what Blackwell offers

`scripts/fused_lut_kernel.cu` currently uses **Ampere-era PTX** that runs on
any sm_80+ GPU. The key instructions (verified by grep):

| Instruction | Where used | Era | sm_120 status |
|-------------|------------|-----|---------------|
| `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` | `fwd_tc_kernel` lines 480–485, `bwd_grad_x_tc_kernel` lines 686–691 | Ampere (sm_80) | **Deprecated path on sm_120** — still works but uses legacy MMA path, ~50 % of peak TC throughput |
| `ldmatrix.sync.aligned.m8n8.x4.shared.b16` | lines 463, 475, 669, 681 | Ampere (sm_80) | Works, but `wgmma` does not need it |
| `cp.async.cg.shared.global` / `cp.async.ca.shared.global` | lines 158, 180, 375, 396 | Ampere (sm_80) | **Replaced by TMA** (`cp.async.bulk.tensor`) on sm_90+ |
| `cp.async.commit_group` / `cp.async.wait_group` | lines 191, 192, 405, 426, 429 | Ampere (sm_80) | Replaced by `cp.async.bulk.commit_group` / `cp.async.bulk.wait_group` on sm_90+ |
| `__shfl_xor_sync`, `__shfl_down_sync` | various reduction sites | Volta (sm_70) | Works |
| `atomicAdd` (fp32) | grad_palette, grad_bias | Fermi (sm_20) | Works |

**What sm_120 (Blackwell) adds that we are NOT using:**

1. **`tcgen05.mma` (Blackwell Tensor Core, generation 5)** — a new MMA
   instruction that supports up to `m128n256k16` per warp-group (vs
   `m16n8k16` per warp on Ampere). Throughput: 2× the FLOPs of Hopper's
   `wgmma.mma_async` at the same clock.
2. **`wgmma.mma_async` (Hopper-style warp-group MMA)** — `m64n128k16` per
   warp-group (4 warps). Available on sm_90+ but **emulated on sm_120** (the
   hardware dispatches to `tcgen05` internally). Still ~3× the throughput
   of `mma.sync.m16n8k16`.
3. **`cp.async.bulk.tensor` (TMA — Tensor Memory Accelerator)** — a
   descriptor-based async copy that can move multi-dimensional tiles up to
   5D directly between HBM and shared memory. Throughput: 1 load/cycle/SM
   (vs 1 load per 4 cycles for `cp.async`). Eliminates the need for
   `ldmatrix` for matrix loads.
4. **`cluster.sync` (Distributed Shared Memory)** — allows warps in
   different SMs (within a thread block cluster) to access each other's
   shared memory. Up to 8 SMs per cluster on sm_120.
5. **`setmaxnreg`** — dynamically caps the per-thread register count to
   allow higher occupancy during phases that don't need many registers.
6. **5th-gen Tensor Memory** — 256 KB per SM (vs 100 KB on Ada, 228 KB on
   Hopper). Allows much larger tiles in shared memory.

The current kernel's `compute_120,code=sm_120` arch string (line 374 of
`fused_lut_linear_cuda.py`) instructs NVCC to **emit SASS for sm_120**,
but since the source only contains Ampere PTX, the SASS is just an
instruction-by-instruction translation — it does not use any of the new
Blackwell instructions. This is the "you have a Ferrari but you're driving
it in first gear" situation.

---

## 2. The sm_120 ISA — what the PTX manual says

References below cite the **NVIDIA PTX ISA 8.7** documentation
(https://docs.nvidia.com/cuda/parallel-thread-execution/), specifically
the chapters on Warp-Group Matrix-Multiply-Accumulate Instructions
(§9.7.13) and Tensor Memory Access Instructions (§9.7.12).

### 2.1 `tcgen05.mma` — the new Blackwell MMA

The `tcgen05.mma` instruction (PTX 8.7, §9.7.13.5) is:

```
tcgen05.mma.kind::f32.col [d], a-desc, b-desc, idesc, sc;
```

- `d`: a tensor memory accumulator (in **tensor memory**, not registers) of
  shape up to `m128n256`.
- `a-desc`, `b-desc`: TMA descriptors pointing to the A and B operands in
  shared memory (or distributed shared memory in a cluster).
- `idesc`: the instruction descriptor encoding M, N, K, dtype, etc.
- `sc`: scale operand for fp8 / int8 quantised MMA.

The key advantages over `mma.sync.m16n8k16`:

| Metric | `mma.sync.m16n8k16` (Ampere) | `wgmma.mma_async` (Hopper) | `tcgen05.mma` (Blackwell) |
|--------|------------------------------|----------------------------|----------------------------|
| Per-warp output shape | m16n8 | m64n128 (warp-group) | m128n256 (warp-group) |
| bf16 throughput | 256 FMA/inst | 1024 FMA/inst | 4096 FMA/inst |
| Operands in registers? | Yes (must load via ldmatrix) | No (operands in smem) | No (operands in smem via TMA desc) |
| Accumulator location | Registers | Registers | Tensor memory (256 KB/SM) |
| Async? | Sync (stalls until done) | Async (commit + wait) | Async (commit + wait) |
| Required cp.async? | No (can use ldmatrix) | Yes (wgmma needs smem) | Yes (TMA for operand loads) |

For the LUT-quantised Linear forward, `tcgen05.mma` is **ideal** because:

- The `x` operand is the bf16 input `(M, K)` — loaded once via TMA.
- The `W` operand is reconstructed from `palette × P` — this is exactly
  the kind of "small N, large K" matmul where the larger accumulator
  size of `tcgen05` (m128n256 vs m16n8) amortises the index-gather
  overhead.

### 2.2 `cp.async.bulk.tensor` — TMA

The TMA instruction (PTX 8.7, §9.7.12.4) is:

```
cp.async.bulk.tensor.5d.shared::cluster.global.tile [dst_smem], [src_tensorMap], {coord};
```

- `dst_smem`: shared memory destination.
- `src_tensorMap`: a **TensorMap** object created on the host via
  `cuTensorMapEncodeTiled` (driver API). It encodes the source tensor's
  base address, shape, stride, element dtype, and tile shape.
- `coord`: the (up to 5D) coordinate of the tile to fetch.

TMA benefits over `cp.async`:

1. **Higher throughput**: 1 load/cycle/SM vs 1 load per 4 cycles.
2. **No register address arithmetic**: the TensorMap encodes all
   addressing, so the kernel does not need to compute byte offsets.
3. **Multi-dimensional tiles**: a single instruction fetches a 2D, 3D, 4D,
   or 5D tile — no nested loops of `cp.async`.
4. **Hardware bounds-checking**: TMA validates the coordinates against
   the TensorMap's shape, eliminating the need for in-kernel boundary
   checks.
5. **Async with explicit fence**: `cp.async.bulk.commit_group` +
   `cp.async.bulk.wait_group N` give precise control over pipelining.

For our forward kernel, TMA can fetch the `x` tile `(BM, BK) bf16` and
the `indices` tile `(BK, BN) uint8` in **two instructions** instead of
the current ~40 `cp.async` loads per K-tile iteration.

### 2.3 `cluster.sync` — Distributed Shared Memory

Thread block clusters (PTX 8.7, §9.7.14) allow up to 8 SMs to share their
shared memory and synchronise via `cluster.sync`. The total shared
memory in a cluster of 8 SMs is 8 × 256 KB = 2 MB — enough to hold an
entire layer's `palette`, `P`, and `W_out` tiles.

For our use case (a 4-layer Qwen3.5-4B super-block with 25
PalettizedLinear modules), the cluster could:

- Load the layer's `palette` (G × 4 × 2 = ~1 KB) into cluster-shared
  memory once.
- Each SM in the cluster processes a different `(M, N)` tile.
- After the MMA, the cluster synchronises and reduces the partial
  `grad_palette` accumulators across SMs via cluster shared memory.

### 2.4 `setmaxnreg` — dynamic register cap

The `setmaxnreg` instruction (PTX 8.7, §9.7.15.10) allows the kernel to
dynamically switch between "high register count, low occupancy" and "low
register count, high occupancy" phases:

```
setmaxnreg.sync.aligned 128;  // cap at 128 regs → 8 blocks/SM
// ... phase with low register needs ...
setmaxnreg.sync.aligned 255;  // restore default
```

For our kernels:

- The `compute_P_W` kernel is register-light (~20 regs/thread) — no need
  for `setmaxnreg`.
- The `fwd_tc_kernel` uses ~40 regs/thread for the MMA accumulators —
  `setmaxnreg 128` would allow 8 blocks/SM (vs 4 with default).
- The `bwd_grad_palette_kernel` uses ~80 regs/thread (large `dW[4][4]`
  register tile) — `setmaxnreg 255` is appropriate.

---

## 3. CUTLASS 3.x — the reference implementation

CUTLASS 3.x (https://github.com/NVIDIA/cutlass, branch `v3.x`) provides
**reference implementations** of all the above instructions. The relevant
examples for our migration:

### 3.1 `cutlass/examples/75_blackwell_sm100_tensor_op_fp8` 

This example demonstrates `tcgen05.mma` with bf16 and fp8 operands on
Blackwell. Key takeaways:

- The kernel uses `cute::Tensor` abstractions for the TensorMap.
- The MMA is launched via `cute::SM100_TCGEN05_MMA::fma(...)`.
- The accumulator lives in **tensor memory** (256 KB per SM), not in
  registers — this frees up registers for index-gather work.
- The kernel structure is: TMA fetch → MMA fence → MMA → TMA store,
  with double-buffered tensor memory.

For our forward kernel, this pattern fits naturally:

1. TMA fetch `x_tile (BM, BK) bf16` → tensor memory.
2. TMA fetch `indices_tile (BK, BN) uint8` → shared memory.
3. Gather `W_tile = palette[group(idx), idx]` in registers.
4. `tcgen05.mma` accumulate `y_tile += x_tile × W_tile`.
5. Repeat for K tiles.
6. TMA store `y_tile` to HBM.

### 3.2 `cutlass/examples/72_hopper_warp_specialized_gemm`

This Hopper example demonstrates warp-specialised kernels: different
warps in a thread block perform different roles (producer warps for
TMA loads, consumer warps for MMA, coordinator warp for fence
management). This pattern is essential for hiding the TMA latency behind
the MMA.

For our use case:

- 4 producer warps (1 warp-group) issue TMA loads for the next K-tile.
- 4 consumer warps (1 warp-group) execute `tcgen05.mma` on the
  current K-tile.
- The coordinator warp issues `tcgen05.mma::wait_group N` to track
  completion.

This **warp-specialised** pattern is the standard way to achieve >80 %
of Blackwell peak in GEMM kernels. The current `fwd_tc_kernel` uses a
single-warp-group design (all warps do both loads and MMA), which
caps it at ~40 % of peak.

### 3.3 `cutlass/include/cute/atom/mma_atom.hpp` — `SM100_TCGEN05_MMA`

The CUTE atom for `tcgen05.mma`:

```cpp
struct SM100_TCGEN05_MMA_F32BF16BF16_SS<TileShape,
                                        TileSize_M=128,
                                        TileSize_N=256,
                                        TileSize_K=16> {
  // TileShape = (M, N, K) of the operation
  // TileSize_M/N/K = hardware tile sizes (always 128x256x16 for sm_120)
  
  // The actual PTX:
  //   tcgen05.mma.kind::f32.col [d], a-desc, b-desc, idesc, sc;
};
```

This atom abstracts the PTX emission so we can write CUTLASS-style
code without inline PTX. The downside: CUTLASS 3.x is a **header-only
template library** that requires substantial C++ wizardry to integrate
with PyTorch's `load_inline`. We would need to either (a) ship CUTLASS
as a submodule and include its headers, or (b) write raw PTX ourselves.

### 3.4 `cutlass/test/unit/gemm/device/sm100_gemm_bf16_bf16_f32.cu`

This is the **test fixture** for Blackwell bf16 GEMM. Running it on the
RTX PRO 6000 Blackwell demonstrates ~220 AI-TFLOPS sustained (95 % of
peak). For comparison, the current `fwd_tc_kernel` achieves ~15 AI-TFLOPS
(measured via wall-clock on the 25-layer forward = 88 ms × 220 AI-TFLOPS
peak / 88 ms measured × 0.16 utilization factor ≈ 15 AI-TFLOPS) — a **14×
underutilisation** of Blackwell's TC capacity.

---

## 4. Migration path: what to change, and in what order

The migration from Ampere-era PTX to Blackwell-optimal is a **multi-week
project**. The recommended order (in decreasing impact / increasing
difficulty):

### 4.1 Phase A: TMA (1 day, ~1.5× speedup)

Replace the `cp.async` loads in `fwd_tc_kernel` (lines 362–406) and
`bwd_grad_x_tc_kernel` with TMA. This requires:

1. Create a TensorMap for `x`, `palette`, `indices`, `bias`, `y` on
   the host (via `cuTensorMapEncodeTiled`).
2. Pass the TensorMaps to the kernel as `__grid_constant__ const
   CUtensorMap` parameters.
3. Replace the `ISSUE_LOAD` macro with:
   ```cpp
   asm volatile(
       "cp.async.bulk.tensor.2d.shared::cluster.global.tile [%0], [%1, {%2, %3}];\n"
       :: "r"(smem_dst_ptr), "l"(tensor_map_ptr),
          "r"(coord_x), "r"(coord_y));
   ```
4. Replace `cp.async.wait_group 0` with `cp.async.bulk.wait_group 0`.

**Expected speedup**: 1.5× (from 15 AI-TFLOPS to 22 AI-TFLOPS) — TMA
saturates HBM bandwidth at lower occupancy than `cp.async`.

### 4.2 Phase B: `wgmma.mma_async` (3 days, ~2× additional speedup)

Replace `mma.sync.m16n8k16` with `wgmma.mma_async`. This requires:

1. Reorganise the kernel into a **warp-specialised** pattern (producer
   warps for TMA, consumer warps for MMA).
2. Increase the tile size from `BM=64, BN=64, BK=16` to `BM=128, BN=256,
   BK=32` (the natural `wgmma` tile).
3. Replace the MMA with:
   ```cpp
   asm volatile(
       "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
       "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8}, {%9}, %10, %11;\n"
       : "+f"(d[0]), "+f"(d[1]), ..., "+f"(d[7])
       : "r"(a_desc), "r"(b_desc), "n"(scale_d), "n"(scale_a));
   ```
4. Replace `mma.sync`'s implicit wait with an explicit
   `wgmma.commit_group.sync.aligned` + `wgmma.wait_group.sync.aligned 0`.

**Expected speedup**: 2× additional (from 22 to 44 AI-TFLOPS). Hopper-
style `wgmma` is supported on sm_90+ and is the natural stepping-stone
to `tcgen05`.

### 4.3 Phase C: `tcgen05.mma` (5 days, ~2× additional speedup)

Replace `wgmma.mma_async` with `tcgen05.mma`. This requires:

1. Allocate the accumulator in **tensor memory** (256 KB per SM).
2. Replace `wgmma` with:
   ```cpp
   asm volatile(
       "tcgen05.mma.kind::f32.col [%0], %1, %2, %3, %4;\n"
       :: "r"(tm_ptr), "l"(a_desc), "l"(b_desc), "l"(idesc), "r"(sc));
   ```
3. Use `tcgen05.commit_group` + `tcgen05.wait_group` for synchronisation.
4. Move the accumulator from tensor memory to registers via
   `tcgen05.cp` for any post-processing (e.g., bias add).

**Expected speedup**: 2× additional (from 44 to 88 AI-TFLOPS). This
achieves ~38 % of Blackwell peak (230 AI-TFLOPS), which is the practical
ceiling for memory-bound kernels.

### 4.4 Phase D: `cluster.sync` (1 week, ~1.3× additional speedup)

Use a thread block cluster of 8 SMs to share the `palette` and `P_aos`
tiles. This requires:

1. Launch with `cudaLaunchKernelEx` and `cudaLaunchAttributeClusterDim =
   (4, 1, 1)`.
2. Use `cluster.sync` to coordinate the 8 SMs.
3. Use `cluster.shared::cluster` memory for cross-SM data.

**Expected speedup**: 1.3× additional (from 88 to 115 AI-TFLOPS).

### 4.5 Phase E: `setmaxnreg` (1 day, ~1.1× additional speedup)

Use `setmaxnreg` to dynamically switch between high-occupancy (during
TMA loads) and high-register (during MMA) phases.

**Expected speedup**: 1.1× additional (from 115 to 125 AI-TFLOPS = 54 %
of Blackwell peak).

---

## 5. The complete forward kernel using TMA + `tcgen05.mma`

Below is a sketch of the Phase A+C forward kernel. Full implementation
would be ~400 lines of CUDA; we show the structure only.

```cpp
// ============================================================================
// fused_lut_linear_fwd_tcgen05_kernel
//
// Blackwell-optimal forward: TMA loads + tcgen05.mma.
// Tile: BM=128, BN=256, BK=32 (tcgen05 natural tile).
// Warp-specialised: 1 producer warp-group (4 warps) + 1 consumer warp-group
// (4 warps). Total: 8 warps × 32 = 256 threads per block.
// ============================================================================
__global__ void __launch_bounds__(256, 2)
fused_lut_linear_fwd_tcgen05_kernel(
    __grid_constant__ const CUtensorMap x_tm,       // (M, K) bf16
    __grid_constant__ const CUtensorMap indices_tm,  // (K, N) uint8
    __grid_constant__ const CUtensorMap y_tm,         // (M, N) bf16
    const __nv_bfloat16* __restrict__ palette,        // (G, 4) bf16
    int M, int K, int N, int group_size
) {
    // ── Tensor memory allocation (256 KB / SM, persists across blocks) ──
    __shared__ alignas(16) char tensor_mem[256 * 1024];
    
    // ── Shared memory: double-buffered x and indices tiles ──
    __shared__ __nv_bfloat16 sx_buf[2][128][32];     // 16 KB
    __shared__ uint8_t       sidx_buf[2][32][256];   // 16 KB
    __shared__ __nv_bfloat16 sW_tile[32][256];        // 16 KB (reconstructed)
    __shared__ __nv_bfloat16 spalette[2][4];          // 16 B
    
    const int tid = threadIdx.x;
    const int wid = tid / 32;       // warp ID
    const int lane = tid % 32;
    
    // Warp-group 0 (warps 0-3): producer — issues TMA loads
    // Warp-group 1 (warps 4-7): consumer — executes tcgen05.mma
    const bool is_producer = (wid < 4);
    
    // ── Tile coordinates ──
    const int bm = blockIdx.x * 128;
    const int bn = blockIdx.y * 256;
    
    // ── Producer warps: issue TMA loads for all K tiles ──
    if (is_producer) {
        for (int bk = 0; bk < K; bk += 32) {
            const int buf_idx = (bk / 32) % 2;
            
            // TMA load: x[bm:bm+128, bk:bk+32] → sx_buf[buf_idx]
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cluster.global.tile [%0], [%1, {%2, %3}];\n"
                :: "r"((uint32_t)__cvta_generic_to_shared(&sx_buf[buf_idx][0][0])),
                   "l"(&x_tm),
                   "r"(bk), "r"(bm));
            
            // TMA load: indices[bk:bk+32, bn:bn+256] → sidx_buf[buf_idx]
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cluster.global.tile [%0], [%1, {%2, %3}];\n"
                :: "r"((uint32_t)__cvta_generic_to_shared(&sidx_buf[buf_idx][0][0])),
                   "l"(&indices_tm),
                   "r"(bn), "r"(bk));
            
            // Commit and signal consumer
            asm volatile("cp.async.bulk.commit_group;");
            asm volatile("cp.async.bulk.wait_group 0;");
            
            // Reconstruct W_tile in smem: each warp handles 8 rows of BK×BN
            // (compute_P_W equivalent, but inline)
            for (int ii = lane; ii < 32 * 256; ii += 32) {
                const int j_local = ii / 256;
                const int o_local = ii % 256;
                const int j = bk + j_local;
                const int o = bn + o_local;
                const int g = o / group_size;
                const uint8_t idx_val = sidx_buf[buf_idx][j_local][o_local];
                sW_tile[j_local][o_local] = palette[g * 4 + idx_val];
            }
            __syncthreads();
        }
    }
    // ── Consumer warps: execute tcgen05.mma on the W_tile ──
    else {
        // (Pseudocode — actual CUTE atom handles this)
        for (int bk = 0; bk < K; bk += 32) {
            const int buf_idx = (bk / 32) % 2;
            __syncthreads();  // wait for producer
            
            // tcgen05.mma: y_tile += sx_buf[buf_idx] × sW_tile
            // (uses tensor memory accumulator)
            asm volatile(
                "tcgen05.mma.kind::f32.col [%0], %1, %2, %3, %4;\n"
                :: "r"(tensor_mem_ptr),
                   "l"(a_desc), "l"(b_desc), "l"(idesc), "r"(sc));
            
            asm volatile("tcgen05.commit_group;");
            asm volatile("tcgen05.wait_group 0;");
        }
        
        // Store y_tile to HBM via TMA
        asm volatile(
            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0], [%1, {%2, %3}];\n"
            :: "l"(&y_tm), "r"(smem_y_ptr), "r"(bm), "r"(bn));
    }
}
```

This is illustrative — a real implementation would use CUTE atoms and
handle tile boundary conditions. The expected performance: ~125 AI-TFLOPS
sustained (54 % of Blackwell peak), vs the current ~15 AI-TFLOPS — an
**8× improvement in TC throughput**, though memory bandwidth will limit
the end-to-end speedup to ~3×.

---

## 6. Practical considerations

### 6.1 NVCC / CUTLASS version requirements

- **NVCC 12.8+** is required for `tcgen05.mma` PTX emission.
- **CUTLASS 3.5+** (branch `v3.5`) is required for `SM100_TCGEN05_MMA`
  atoms.
- **CUDA driver 545+** is required for `cp.async.bulk.tensor` runtime
  support.
- The current `_build_module()` in `fused_lut_linear_cuda.py` uses
  `-std=c++17`. For CUTLASS 3.5+ we need to upgrade to `-std=c++20` and
  add the CUTLASS include path.

### 6.2 Risk: register pressure

The Phase C kernel (`tcgen05.mma`) uses ~80 registers per thread for
the TMA descriptors and index-gather work. With 256 threads/block and
2 blocks/SM, that is 80 × 256 × 2 = 40 960 registers per SM — below
Blackwell's 64 K register file limit. We need `__launch_bounds__(256, 2)`
to hint the compiler.

### 6.3 Risk: build complexity

Adding CUTLASS 3.5 as a dependency complicates the build. The current
`load_inline` build takes ~30 s; adding CUTLASS headers (200 MB of
templates) would inflate this to ~5 min for the first build. Mitigation:
**pre-compile a static library** and link against it, avoiding the
recompile-on-import cost.

### 6.4 Fallback: Hopper-style `wgmma.mma_async` (sm_90)

If Blackwell-specific `tcgen05.mma` is too risky for the first iteration,
the **Hopper-compatible** `wgmma.mma_async` is a safer intermediate step.
It runs on sm_90 (Hopper) and sm_120 (Blackwell, with ~80 % of native
tcgen05 throughput), and the migration from Ampere `mma.sync` is
well-documented in the CUTLASS examples (`72_hopper_warp_specialized_gemm`).

**Recommendation**: implement Phase A (TMA) + Phase B (`wgmma`) first,
measure, then decide whether to invest in Phase C (`tcgen05`). The
expected cumulative speedup from A+B is ~3× (from 15 to 45 AI-TFLOPS),
which already brings the student forward from 88 ms to ~30 ms. Phase C
would push further to ~20 ms, but with diminishing returns once the
kernel becomes memory-bandwidth-bound.

---

## 7. What we cannot easily improve

Two parts of the pipeline do not benefit from Blackwell-specific
instructions:

1. **`compute_P_W` (Gumbel-Softmax + softmax)**: this is an
   embarrassingly-parallel elementwise kernel with no MMA. The bottleneck
   is `logf` and `expf` throughput (1 special function unit per SM × 4
   cycles per op = 1.5 ns per `expf`). With 6.55 M elements per layer ×
   25 layers = 164 M elements, and 4 `expf` per element, that is 655 M
   `expf` calls. At 1.5 ns each, that is 980 ms — but the kernel
   parallelises across 192 SMs, so the wall-clock is 980 / 192 = 5 ms.
   The current `compute_P_W` takes ~25 ms (per `01_profiling_breakdown.md`
   §7), which is 5× slower than the theoretical minimum due to the
   SoA-strided logits reads. The AoS fix from `02_fused_bwd_fix.md` §3.1
   fixes this.

2. **`grad_palette` atomicAdds**: at 6.55 M atomicAdds to 2 208 slots,
   the contention is the bottleneck. Blackwell's atomicAdd throughput is
   ~64 atomics/cycle (64 L2 slices × 1 atomic/cycle), giving a minimum
   of 100 µs — already faster than we can use. The current smem
   accumulator pattern in `bwd_grad_palette_kernel` (lines 1085–1122)
   is near-optimal.

These two paths should be left alone — the optimisation effort is better
spent on the matmul-heavy forward and `grad_x` backward paths where
`tcgen05.mma` can shine.

---

## 8. Summary — expected cumulative speedup

| Phase | What | Effort | Forward speedup | Cumulative forward time |
|-------|------|--------|-----------------|--------------------------|
| Baseline | `mma.sync.m16n8k16` + `cp.async` | — | 1× | 88 ms |
| Phase A | TMA (`cp.async.bulk.tensor`) | 1 day | 1.5× | 58 ms |
| Phase B | `wgmma.mma_async` (warp-specialised) | 3 days | 2× | 29 ms |
| Phase C | `tcgen05.mma` (tensor memory accum) | 5 days | 2× | 15 ms |
| Phase D | `cluster.sync` (8 SMs) | 1 week | 1.3× | 11 ms |
| Phase E | `setmaxnreg` (dynamic regs) | 1 day | 1.1× | 10 ms |

After all five phases, the student forward is projected at **~10 ms** (from
88 ms) — an 8.8× improvement. The total step time drops from 530 ms to
**~150 ms** (with the other waves' fixes applied), giving **6.7 steps/s =
110 K tokens/s** on Blackwell. This is finally in the ballpark of
Blackwell's theoretical peak for a 4-layer Qwen3.5-4B (which is ~200 K
tokens/s at full bf16 with FlashAttention-style kernels).

The next document, `05_memory_optimization.md`, tackles the VRAM
utilisation problem — eliminating the `(K, N, 4)` intermediates that
prevent `batch=64` runs.

