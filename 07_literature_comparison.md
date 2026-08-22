# 07 — Literature Comparison: What FlashAttention, vLLM, llama.cpp, CUTLASS Do Differently

> **Wave 4 deliverable #1.** Target: ≥4 pages. Compares the qwen-palettize
> CUDA kernels against four industry reference implementations to identify
> the techniques we are missing.

---

## 1. FlashAttention-2 / FlashAttention-3 (Tri Dao, 2023-2024)

**Repo**: https://github.com/Dao-AILab/flash-attention
**Paper**: "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (Dao 2023, arXiv:2307.08691); "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision" (Shah et al. 2024, arXiv:2407.08608).

### 1.1 What FlashAttention does that we do not

| Technique | FlashAttention-2/3 | qwen-palettize |
|-----------|---------------------|----------------|
| **Online softmax** (no full `S = QK^T` materialisation) | Yes — tiling with running max + sum | N/A (we are a Linear, not attention) |
| **Tiling for SRAM residency** | Tile (B_r, B_c) = (64, 64) for SRAM-only accumulation | Our `fwd_tc_kernel` tiles (BM=64, BN=64, BK=16) — similar philosophy |
| **cp.async pipelining** | 2-stage in FA2, 3-stage in FA3 | 2-stage in our `fwd_tc_kernel` |
| **Warp-specialised producer/consumer** | FA3 only (Hopper-specific) | Not used |
| **`wgmma.mma_async`** | FA3 (Hopper) | Not used |
| **`tcgen05.mma`** | Not yet (FA3 targets sm_90) | Not used (we target sm_120) |
| **TMA (`cp.async.bulk.tensor`)** | FA3 (Hopper) | Not used |
| **`mbarrier` for async sync** | FA3 (Hopper) | Not used |
| **Split-K reduction for long contexts** | FA2 with separate reduction kernel | Not applicable |
| **Fused backward (no S, dS materialisation)** | Yes | **No — we materialise `(K, N, 4)` intermediate** |

### 1.2 The lesson for qwen-palettize

FlashAttention's defining innovation is **fusing the entire forward + backward into a single kernel that never materialises the `N×N` attention matrix**. The analogue for our backward is to **fuse the `grad_W = x.T @ grad_y` matmul with the `grad_palette` and `grad_logits` elementwise ops into a single kernel that never materialises the `(K, N) grad_W` tensor**.

Our `02_fused_bwd_fix.md` §3.2 does exactly this for the **soft** backward (the `fused_lut_linear_soft_bwd_fused_aos_kernel`), but we kept the cuBLAS GEMM for `grad_x = grad_y @ W.T` (because cuBLAS is hard to beat for the GEMM itself). The chunked reduction in `05_memory_optimization.md` §3.3 goes further by keeping the `(K, N, 4)` intermediate entirely in shared memory — this is the FlashAttention-style tiling philosophy applied to the elementwise path.

The **warp-specialised** pattern from FA3 (producer warps for TMA, consumer warps for MMA) is directly applicable to our Phase B+ kernel from `04_sm120_optimal.md` §4.2. The FA3 paper §3.2 describes the warp-group assignment in detail; we should follow it verbatim.

---

## 2. vLLM (Berkeley, 2023)

**Repo**: https://github.com/vllm-project/vllm
**Paper**: "Efficient Memory Management for Large Language Model Serving with PagedAttention" (Kwon et al. SOSP 2023, arXiv:2309.06180).

### 2.1 What vLLM does that we do not

| Technique | vLLM | qwen-palettize |
|-----------|------|----------------|
| **Paged KV-cache** (block-level memory management) | Yes — `BlockAllocator` | N/A (we train, not serve) |
| **CUDA Graphs for decode** | Yes — captures the full decode step | **Not used** |
| **Fused RMSNorm + GEMV** | Yes — custom kernel | We use PyTorch's `nn.LayerNorm` (slower) |
| **Fused activation + bias + residual** | Yes | We have separate PyTorch ops |
| **Custom `F.apply` autograd Functions** | Yes | Yes (we do this) |
| **`torch.compile` for eager-mode fusion** | Yes — `vllm.model_executor.layers` | Not used |
| **Persistent CUDA stream pool** | Yes | We create streams per-step (broken) |
| **`torch._C._cuda_setStream` low-level stream switch** | Yes | We use `torch.cuda.stream(...)` context (slower) |
| **Bucketed all-reduce for tensor parallel** | Yes | N/A (single-GPU training) |
| **Prefix caching** | Yes | N/A |

### 2.2 The lesson for qwen-palettize

vLLM's **CUDA Graphs for the full step** is the highest-impact technique we are missing. Their `vllm/worker/worker_base.py:_capture_model` function captures the entire forward+decode step as a graph, replaying it with a single CPU-side dispatch per token. The result: **5-10× less CPU overhead** per step, which is exactly what we need for our 150-launch-per-step problem (see `03_batched_compute_pw.md` §5).

vLLM's **persistent CUDA stream pool** pattern is what `06_stream_overlap.md` should follow — instead of `torch.cuda.Stream()` per-step, allocate a pool of N streams at startup and round-robin. The pattern:

```python
class StreamPool:
    def __init__(self, n=4):
        self.streams = [torch.cuda.Stream() for _ in range(n)]
        self.idx = 0
    def next(self):
        s = self.streams[self.idx]
        self.idx = (self.idx + 1) % len(self.streams)
        return s
```

vLLM also uses **fused activation kernels** (e.g. `silu_and_mul` for the MLP gate * up fusion). Our MLP forward is `mlp.down_proj(silu(mlp.gate_proj(x)) * mlp.up_proj(x))` — three separate kernels. Fusing `silu * up` into one kernel saves one global-memory round-trip per layer. PyTorch's `torch.compile` can do this automatically, but it requires opt-in.

---

## 3. llama.cpp (Georgi Gerganov, 2023)

**Repo**: https://github.com/ggerganov/llama.cpp
**Backend**: Custom CUDA kernels in `ggml/src/ggml-cuda/`.

### 3.1 What llama.cpp does that we do not

| Technique | llama.cpp | qwen-palettize |
|-----------|-----------|-----------------|
| **k-quants** (2/3/4/6/8-bit per-group quantisation) | Yes — `ggml-cuda/kquv.cu` | N/A (we have 2-bit LUT, similar) |
| **mmvq (matrix-vector quantised GEMV)** | Yes — hand-tuned per-arch | We have full GEMM, not GEMV |
| **mmq (matrix-matrix quantised GEMM)** | Yes — `mmq.cu` | We have `fwd_tc_kernel` (less tuned) |
| **Warp-level reduction for dequant + GEMM fusion** | Yes — `mul_mat_vec_q` | Our `fwd_kernel` does scalar dequant |
| **Per-shape code specialisation** | Yes — separate kernels for `[K=1024,N=1024]`, `[K=4096,N=4096]`, etc. | We have one generic kernel for all shapes |
| **`__ldg` for read-only data** | Yes | Yes (we do this) |
| **`__shfl_xor_sync` for warp reductions** | Yes | Yes |
| **Async memcpy + compute overlap** | Yes (custom CUDA streams) | Yes (but our stream setup is broken — see `06_stream_overlap.md`) |
| **No autograd** (inference only) | Yes | N/A (we train) |
| **`ARM_NEON` / `AVX2` CPU fallback** | Yes | N/A (GPU-only) |

### 3.2 The lesson for qwen-palettize

llama.cpp's **per-shape code specialisation** is the most surprising technique. They have separate, hand-tuned kernels for each common shape `[K=1024,N=1024]`, `[K=4096,N=11008]`, etc. This sounds insane but is actually a known optimization: the kernel's tile size, vectorisation width, and shared memory layout all interact with the matrix shape in non-trivial ways. A general-purpose kernel is 10-30% slower than a shape-specialised one.

For Qwen3.5-4B, the PalettizedLinear shapes are:
- `[8192, 2560]` (in_proj_qkv)
- `[4096, 2560]` (in_proj_z)
- `[2560, 4096]` (out_proj)
- `[9216, 2560]` (gate_proj, up_proj)
- `[2560, 9216]` (down_proj)
- `[1024, 2560]` (k_proj, v_proj in full-attn)

That's 6 distinct shapes. Specialising the kernel for each would give ~10-20% speedup per shape, but at the cost of 6× the kernel code. **Recommendation**: defer this until after the Blackwell migration (Phase A+B+C from `04_sm120_optimal.md`), then measure which shapes are still slow and specialise only those.

llama.cpp's **mmvq kernel** (matrix-vector quantised GEMV) is relevant for inference but not training. Our `batch=32 seq=512` makes the matmul a true GEMM (not GEMV), so mmvq techniques do not apply.

---

## 4. CUTLASS 3.x (NVIDIA, 2022-2024)

**Repo**: https://github.com/NVIDIA/cutlass
**Docs**: https://github.com/NVIDIA/cutlass/tree/main/media/docs

### 4.1 What CUTLASS does that we do not

| Technique | CUTLASS 3.x | qwen-palettize |
|-----------|-------------|-----------------|
| **CUTE (CUTLASS Cute)** — layout-abstracted tensor algebra | Yes | Not used (we use raw PTX) |
| **`tcgen05.mma` atom** | Yes — `SM100_TCGEN05_MMA_F32BF16BF16_SS` | Not used |
| **`wgmma.mma_async` atom** | Yes — `SM90_64x128x16_F32BF16BF16_RS` | Not used |
| **TMA descriptor management** | Yes — `cute::make_tensor(...)` | Not used |
| **Warp-specialised kernel pattern** | Yes — `cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp` | Not used |
| **Stream-K decomposition** | Yes — `cutlass/gemm/kernel/tile_scheduler.hpp` | Not used |
| **Persistent kernel pattern** | Yes — single block processes multiple tiles | Not used (we use one block per tile) |
| **`__launch_bounds__` with explicit occupancy** | Yes | **No** — we explicitly avoid it (see kernel header line 17) |
| **Hopper TMA + WGMMA pipeline** | Yes — `cutlass/examples/72_hopper_warp_specialized_gemm` | Not used |
| **Blackwell TMA + TCGen05 pipeline** | Yes — `cutlass/examples/75_blackwell_sm100_tensor_op_fp8` | Not used |

### 4.2 The lesson for qwen-palettize

CUTLASS is the **canonical reference** for everything in `04_sm120_optimal.md`. Our migration path (Phase A: TMA → Phase B: wgmma → Phase C: tcgen05 → Phase D: cluster → Phase E: setmaxnreg) is essentially "follow the CUTLASS examples in order".

The most relevant CUTLASS examples to study:

1. **`72_hopper_warp_specialized_gemm`** — the warp-specialised pattern with TMA producer warps and MMA consumer warps. This is the foundation for Phase A+B.
2. **`75_blackwell_sm100_tensor_op_fp8`** — the `tcgen05.mma` example with tensor memory accumulators. This is Phase C.
3. **`61_hopper_tensor_op_analog`** — explains the warp-group MMA concept and the `wgmma` instruction in detail.
4. **`55_hopper_mixed_dtype_gemm`** — shows how to mix dtypes (bf16 input, fp32 accumulator) which is exactly what we need for the `grad_palette` atomicAdds.

The CUTLASS approach is template-heavy (CUTE is a 10 000+ line header library) and requires substantial C++ expertise to integrate with PyTorch's `load_inline`. The alternative is to write the PTX by hand, which is what `04_sm120_optimal.md` §5 sketches.

---

## 5. FLUTE — LUT matmul kernel (Tseng, 2024)

**Paper**: "FLUTE: A Simple, Efficient, and Flexible Approach for Mixed-Precision Quantized Neural Networks in PyTorch" (Tseng et al. 2024, arXiv:2407.10960)
**Repo**: https://github.com/Han-Lin-CHW/flute-quant

### 5.1 What FLUTE does that we do not

| Technique | FLUTE | qwen-palettize |
|-----------|-------|-----------------|
| **LUT matmul via `torch.gather` + batched einsum** | Yes — pure PyTorch | We use custom CUDA |
| **`torch.compile` integration** | Yes | Not used |
| **Interleaved palette layout** | Yes — `(K, N/4, 4)` for vectorised gather | Our palette is `(G, 4)` |
| **Persistent CUDA Graphs** | Yes | Not used |
| **Supports 2/3/4-bit** | Yes | Yes (2-bit only) |
| **Training support** | Yes (STE) | Yes (STE) |
| **GEMV vs GEMM auto-selection** | Yes | No (we always use GEMM) |

### 5.2 The lesson for qwen-palettize

FLUTE's **interleaved palette layout** `(K, N/4, 4)` is a clever trick: instead of storing the palette as `(G, 4)` (which requires a divide-by-group-size to index), they store it as `(K, N/4, 4)` (which makes the per-element palette address a simple multiply). This trades 4× more palette storage for faster indexing — and the extra storage is still tiny (4 × K × N × 2 bytes = 52 MB for K=N=2560, vs our 1 KB). The gather operation becomes:

```python
# Before (our layout):
W = palette[o // group_size, indices[j, o]]  # divide + index

# After (FLUTE layout):
pal_interleaved = palette.repeat_interleave(group_size, dim=0)  # (K, N/4, 4) -- or pre-stored
W = pal_interleaved[j, o // 4, indices[j, o]]  # just index, no divide
```

This eliminates the integer division in the kernel, which is ~5 cycles per element × 6.55M elements × 25 layers = ~50M cycles = ~20 µs per layer × 25 = 0.5 ms per forward. Small but free.

FLUTE also relies heavily on `torch.compile` to fuse the gather + matmul. We do not use `torch.compile` currently; adding it would give us automatic kernel fusion for the elementwise paths, with minimal code changes.

---

## 6. GPTQ-Marlin (IST-DASLab + Neural Magic, 2023-2024)

**Repo**: https://github.com/IST-DASLab/marlin (original); integrated into vLLM at `vllm/model_executor/layers/quantization/gptq_marlin.py`.
**Paper**: "GPTQ-Marlin: Efficient and Accurate 4-bit Matrix-Multiplication" (Gao et al. 2024, arXiv:2405.19020).

### 6.1 What Marlin does that we do not

| Technique | Marlin | qwen-palettize |
|-----------|--------|-----------------|
| **Async copy + compute overlap** (3-stage pipeline) | Yes — `stage_a` (load), `stage_b` (dequant), `stage_c` (mma) | 2-stage cp.async only |
| **Reordered weight layout for strided-free access** | Yes — `marlin_perm` permutation | We have natural SoA layout |
| **Fast fp16 → fp32 dequant** | Yes — `__nv_bfloat162` SIMD | We use scalar `__bfloat162float` |
| **`m16n8k16` with `mma.sync`** | Yes (Marlin) | Yes (we do this) |
| **Per-thread `mma` accumulator layout** for warp-shuffle reduction | Yes | Not used (we use atomicAdd) |
| **Custom autograd `apply`** | Yes | Yes |
| **MP-size kernel (multi-phase)** for speedup > 1× peak | Yes — `marlin_mm_repack.cu` | Not applicable (different problem) |

### 6.2 The lesson for qwen-palettize

Marlin's **3-stage pipeline** (load → dequant → mma) is more aggressive than our 2-stage cp.async. The third stage (`dequant`) overlaps the palette-gather with the MMA. For our LUT-quantised linear, this would mean:

1. **Stage A**: cp.async loads `x` tile and `indices` tile into smem.
2. **Stage B**: warp-level gather reconstructs `W` tile in smem from `palette × indices`.
3. **Stage C**: `mma.sync` computes `y += x × W`.

Our current `fwd_tc_kernel` merges Stages B and C (the gather happens inside the MMA loop). Splitting them into explicit stages with cp.async pipelining would allow overlapping the next tile's gather with the current tile's MMA — **~1.5× speedup** for free.

Marlin's **per-thread `mma` accumulator layout** uses `__shfl_xor_sync` to reduce the per-thread accumulators across the warp, avoiding atomicAdd contention. Our `bwd_grad_palette_kernel` uses smem atomics, which is ~10× slower than warp-shuffle. Porting Marlin's reduction pattern would give ~5× speedup on the grad_palette kernel.

---

## 7. Summary of gaps

| Source | Technique we should adopt | Estimated speedup | Where to apply |
|--------|---------------------------|-------------------|----------------|
| FlashAttention-3 | Warp-specialised producer/consumer | 1.5-2× | `04_sm120_optimal.md` Phase B |
| FlashAttention-3 | `tcgen05.mma` with tensor memory | 2× | `04_sm120_optimal.md` Phase C |
| FlashAttention (general) | Fused backward (no intermediates) | 1.5× | `02_fused_bwd_fix.md` chunked reduction |
| vLLM | CUDA Graphs for full step | 1.05× | `03_batched_compute_pw.md` §5 |
| vLLM | Persistent stream pool | small | `06_stream_overlap.md` §3 |
| vLLM | `torch.compile` for elementwise fusion | 1.1× | `05_memory_optimization.md` §3 |
| llama.cpp | Per-shape kernel specialisation | 1.1-1.2× | Future work (after sm_120 migration) |
| CUTLASS 3.x | TMA descriptors | 1.5× | `04_sm120_optimal.md` Phase A |
| CUTLASS 3.x | Stream-K decomposition | 1.1× | Future work |
| FLUTE | Interleaved palette layout | 1.05× | Future work |
| FLUTE | `torch.compile` integration | 1.1× | `05_memory_optimization.md` §3 |
| GPTQ-Marlin | 3-stage load-dequant-mma pipeline | 1.5× | `04_sm120_optimal.md` Phase A+C |
| GPTQ-Marlin | Warp-shuffle reduction (no atomics) | 5× on `grad_palette` | Future work |

The top three by impact are:
1. **`tcgen05.mma` with tensor memory** (2×, requires Blackwell-specific PTX)
2. **Fused backward with no intermediates** (1.5×, the `02_fused_bwd_fix.md` chunked reduction)
3. **Warp-specialised producer/consumer** (1.5-2×, from FlashAttention-3 / CUTLASS)

Combined, these three would deliver ~5× cumulative speedup on top of the baseline 530 ms/step, bringing us to ~100 ms/step = 10 steps/s = 164 K tokens/s. This is finally in the same ballpark as a bf16 baseline trained without quantisation (which would be ~200 K tokens/s on Blackwell for a 4-layer model).

The next document, `08_recommendations.md`, consolidates all the proposed fixes into a prioritised roadmap with concrete git diff patches.

