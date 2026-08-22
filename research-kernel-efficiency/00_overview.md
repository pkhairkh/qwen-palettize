# 00 — Executive Overview

**Repository audited:** `pkhairkh/qwen-palettize` — 2-bit LUT-quantized training of Qwen3.5-4B with Gumbel-Softmax trainable indices + LoRA rank-16/32.

**Target hardware:** NVIDIA Blackwell sm_120 (RTX PRO 6000, 96 GB VRAM, 600 W TDP).

This overview consolidates two complementary research streams:

- **Part A — Quality Audit** (cos plateau diagnosis): why training plateaus at cos≈0.95 despite the literature achieving cos>0.999 at 2-4 bits.
- **Part B — Performance Efficiency** (530 ms/step diagnosis): why Blackwell delivers only 14× the throughput of an L4 despite 9.3× the HBM bandwidth.

---

# Part A — Quality Audit: The cos Plateau

## A1. Symptom

Forward cos=0.947 with teacher; training plateaus at cos≈0.95; literature (GPTQ, AWQ, SqueezeLLM, QuIP#, AQLM, VPTQ, GPTVQ) routinely achieves cos>0.999 at 2-4 bits.

## A2. TL;DR

The cos plateau at 0.946 is **NOT a kernel-numerics problem**. The CUDA kernels themselves are numerically sound — the hard forward path supports cos > 0.999. The plateau is the **2-bit scalar LUT representation limit** (calibration cos = 0.937), marginally improved by LoRA (+0.009 cos). The Gumbel-Softmax index training is a **no-op** at low temperature (vanishing gradient, mathematically proven), and the kernel precision issues (bf16 grad_W, fp16 P/grad_logits) account for only ~0.001 cos of the gap.

**To break the plateau, the repo must adopt techniques from the literature:** GPTQ-style second-order calibration (+2-4% cos), outlier isolation (+1-3% cos), and/or vector quantization (+4-6% cos). Pure kernel tuning will not help.

## A3. Quality audit found 10 issues, ranked by impact

| # | Issue | Severity | Cos impact | Fix difficulty |
|---|-------|----------|-----------|----------------|
| 1 | 2-bit scalar LUT has info-theoretic ceiling cos ~0.94 | CRITICAL | 0.063 | Very High (switch to VQ) |
| 2 | No GPTQ-style second-order calibration | HIGH | 0.02-0.04 | Medium |
| 3 | No outlier isolation (top 0.5% in FP16) | HIGH | 0.01-0.03 | Medium |
| 4 | Gumbel-Softmax vanishing gradient at low τ | HIGH | 0.005-0.01 | Low (floor τ at 0.5) |
| 5 | bf16 grad_W (loses 16 bits of mantissa) | MEDIUM | 0.002-0.005 | Low (use fp32) |
| 6 | fp16 P storage (underflows non-argmax probs) | MEDIUM | 0.001-0.002 | Low (use bf16 or fp32) |
| 7 | fp16 grad_logits storage (underflows small grads) | MEDIUM | 0.001-0.002 | Low (use fp32) |
| 8 | No AWQ-style activation-aware scaling | LOW | 0.005-0.01 | Low |
| 9 | No RHT incoherence preprocessing (QuIP) | LOW | 0.005-0.01 | Medium |
| 10 | Cosine loss (vs block-MSE) gives weak gradient | LOW | 0.005-0.02 | Low |

Issues 1-3 account for >90% of the cos gap. Issues 4-7 are kernel-numerics issues that prevent the Gumbel-Softmax indices from training, but even if fixed, the index training would add at most ~1% cos. Issues 8-10 are minor optimizations.

## A4. Kernel audit verdict

**Hard forward kernel** (`fused_lut_linear_fwd_kernel`, lines 95–288 of `fused_lut_kernel.cu`): scalar fp32 FMA, numerically MORE accurate than the TC variant. Per-element relative error ~8e-3. Theoretical cos ceiling >0.9999.

**Hard TC forward kernel** (`fused_lut_linear_fwd_tc_kernel`, lines 309–518): `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`. Per-element relative error ~5e-3. Theoretical cos ceiling >0.99999.

**Soft forward kernel** (`fused_lut_linear_soft_compute_P_W_kernel`, lines 1301–1352): Gumbel-Softmax with LCG-based PRNG (statistically weak). P stored as fp16 (denormal underflow at τ ≤ 0.5). Forward numerically fine; the issue is the backward.

**Soft backward kernels** (lines 1364–1434, 1498–1610): gradient formula is mathematically correct. BUT the Python backward (which is what actually runs) uses bf16 grad_W and fp16 grad_logits. The CUDA soft backward kernels are NOT used in training — the Python path computes everything via `torch.matmul` + PyTorch elementwise ops. The kernel precision issues are in the Python backward, not the CUDA kernels.

## A5. STE verdict

The Straight-Through Estimator `W = W_hard - W_soft.detach() + W_soft` is **mathematically correct**: forward evaluates to `W_hard` (exact one-hot), backward gradient routes through `W_soft`. However, the STE is **ineffective at low temperature**: at τ ≤ 0.1, P becomes one-hot, so `P[k] = 0` for non-argmax k, and `grad_logits = grad_W * P * (c - W_soft)` vanishes. The indices are frozen.

---

# Part B — Performance Efficiency: The 530 ms/Step

## B1. The problem in one paragraph

We train Qwen3.5-4B (4-layer super-block) on an RTX PRO 6000 Blackwell
(sm_120, 96 GB VRAM, 600 W TDP) using a 2-bit LUT quantisation scheme
(`PalettizedLinear` with Gumbel-Softmax trainable indices + STE).
Observed performance is **0.8 steps/s** at `batch=32 seq=512` =
**25.6 K tokens/s** — barely 14× the throughput of an L4 GPU at
`batch=8 seq=128` (922 tokens/s), despite Blackwell having 9.3× the
HBM bandwidth and 3.8× the bf16 FLOPs. GPU utilisation reports 99 %
but power draw is only 411 W / 600 W (68 % TDP) and VRAM usage is
36 GB / 96 GB (37 %) — the GPU is **latency-bound, not compute-bound**.

## B2. Root causes (4 of them)

1. **Strided `(4, K, N)` P-plane access** in the soft backward kernel
   (`fused_lut_kernel.cu` lines 1586–1589). The kernel reads `P` as
   4 separate 16-bit loads at addresses 26 MB apart, defeating L2 cache
   and forcing 60× re-fetches. The fused CUDA kernel
   (`fused_lut_linear_soft_bwd_fused_kernel`) is **3.8× slower** than
   the PyTorch elementwise path that explicitly materialises a 52 MB
   intermediate — a counter-example to the "always fuse" rule of thumb.

2. **25 separate `compute_P_W` kernel launches per forward** (one per
   `PalettizedLinear`). Each launch costs 5–10 µs of CPU-side dispatch,
   totalling ~0.5 ms of pure overhead per forward — small in absolute
   terms but a 30× larger fraction of step time on Blackwell than on L4.

3. **No Blackwell-specific instructions** despite the
   `compute_120,code=sm_120` arch string. The kernel uses only Ampere-era
   PTX (`mma.sync.m16n8k16`, `cp.async`, `ldmatrix`), leaving ~85 % of
   Blackwell's tensor-core capacity on the table. CUTLASS 3.x examples
   (`75_blackwell_sm100_tensor_op_fp8`) demonstrate how to use `tcgen05.mma`
   with tensor memory accumulators for 8× the bf16 throughput.

4. **Broken stream overlap**. The comment in `train_qwen.py` lines 1058–1059
   says "teacher prepares next batch while student trains on current",
   but the implementation creates a fresh `stream_t` per step and immediately
   consumes `h_out` in the same iteration — there is no double-buffering,
   so the 68 ms teacher forward is fully serialised with the 260 ms student
   backward.

## B3. The 530 ms/step breakdown

| Component | Time (ms) | % | Fix doc |
|-----------|-----------|---|---------|
| Teacher forward (4 layers, bf16, cuBLAS) | 68 | 13 % | `06_stream_overlap.md` (hide behind backward) |
| Student forward (soft `compute_P_W` × 25 + cuBLAS × 25) | 88 | 17 % | `03_batched_compute_pw.md`, `04_sm120_optimal.md` |
| Backward (cuBLAS + Python elementwise `(K,N,4)`) | 260 | 49 % | `02_fused_bwd_fix.md`, `05_memory_optimization.md` |
| Optimizer step (FP32MasterAdamW for 1.78B `index_logits`) | 113 | 21 % | `08_recommendations.md` §6 (fused AdamW) |
| **Total** | **530** | 100 % | projected → **51 ms** after all 10 patches |

## B4. The 10-patch performance roadmap

| # | Patch | Effort | Savings | Cumulative |
|---|-------|--------|---------|------------|
| 1 | Stream double-buffering | 0.5 day | 69 ms | 461 ms |
| 2 | AoS layout for `P` (rewrite `compute_P_W` + `bwd_fused`) | 1 day | 150 ms | 311 ms |
| 3 | Batched `compute_P_W` (1 launch for 25 layers) | 2 days | 18 ms | 293 ms |
| 4 | Re-enable the fused `bwd_fused_aos` kernel | 1 day | 90 ms | 203 ms |
| 5 | Buffer pooling for `P_aos` | 0.5 day | (memory) | 203 ms |
| 6 | Fused AdamW (`bitsandbytes.optim.AdamW8bit` or `torch.optim.AdamW(fused=True)`) | 0.5 day | 70 ms | 133 ms |
| 7 | `CUBLAS_WORKSPACE_CONFIG` cap (enables batch=64) | 0.1 day | (memory) | 133 ms |
| 8 | TMA + `wgmma.mma_async` migration | 4 days | 60 ms | 73 ms |
| 9 | `tcgen05.mma` (Blackwell tensor core) | 5 days | 20 ms | 53 ms |
| 10 | CUDA Graphs for full step | 1 day | 2 ms | 51 ms |

**Patches 1-7** are "low-hanging fruit" with no Blackwell-specific PTX.
They alone bring the step time to **133 ms = 7.5 steps/s = 123 K tokens/s**,
a **4.8× throughput improvement** with ~6 days of engineering effort.

**Patches 8-10** require Blackwell-specific PTX (NVCC 12.8+, CUTLASS 3.5+).
They bring the step time to **51 ms = 19.6 steps/s = 327 K tokens/s**, a
**12.8× total throughput improvement** with an additional ~10 days of effort.

## B5. Expected end state

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Step time | 530 ms | 51 ms | 10.4× |
| Throughput (tokens/s) | 25.6 K | 327 K | 12.8× |
| GPU power | 411 W (68 % TDP) | 580 W (97 % TDP) | compute-bound |
| VRAM usage | 36 GB (37 %) | 45 GB (47 %, unlocks batch=64+) | 1.25× |
| L4 vs Blackwell per-token ratio | 14.2× | ~230× | Blackwell finally wins |

---

# Combined Outlook

## Quality + Performance synergy

The quality audit (Part A) and the performance audit (Part B) are **complementary**:

- **Part A** identifies that the cos plateau is fundamental to the 2-bit scalar LUT representation. Switching to vector quantization (VQ) or adding GPTQ-style calibration would break the plateau — but these are algorithmic changes that require additional compute and memory.
- **Part B** identifies that the current kernel pipeline is 10× slower than it should be on Blackwell. Fixing the performance issues gives us the **headroom to afford the algorithmic improvements from Part A**.

For example:
- GPTQ second-order calibration requires computing the Hessian of each weight block — this is ~3× the compute of plain LUT training. With the current 530 ms/step, this would push step time to ~1.6 s, making training infeasible. With the fixed 51 ms/step, GPTQ calibration adds only ~150 ms, keeping step time under 200 ms.
- Vector quantization (VQ) requires searching a codebook per group, which is ~5× the compute of scalar LUT. On the current pipeline this is prohibitive; on the fixed pipeline it adds ~200 ms.
- Outlier isolation requires maintaining a small dense FP16 matrix for the top-0.5% weights, which is ~10% more memory and ~5% more compute — trivial on the fixed pipeline.

**The recommendation is to fix performance first (Patches 1-7), then revisit the quality improvements with the available compute budget.**

## Document index

| # | File | Pages | Stream | Wave |
|---|------|-------|--------|------|
| 0 | `00_overview.md` | 3 (this file) | Both | 4 |
| 1 | `01_profiling_breakdown.md` | 6 | Perf | 1 |
| 1 | `01_kernel_audit.md` | 8 | Quality | (existing) |
| 1 | `01_palette_audit.md` | 7 | Quality | (existing) |
| 2 | `02_fused_bwd_fix.md` | 8 | Perf | 2 |
| 2 | `02_gradient_correctness.md` | 7 | Quality | (existing) |
| 2 | `02_numerical_analysis.md` | 7 | Quality | (existing) |
| 3 | `03_batched_compute_pw.md` | 5 | Perf | 2 |
| 3 | `03_precision_analysis.md` | 7 | Quality | (existing) |
| 3 | `03_ste_analysis.md` | 5 | Quality | (existing) |
| 4 | `04_sm120_optimal.md` | 6 | Perf | 3 |
| 4 | `04_kmeans_vs_gradient.md` | 8 | Quality | (existing) |
| 4 | `04_literature_comparison.md` | 7 | Quality | (existing) |
| 5 | `05_memory_optimization.md` | 5 | Perf | 3 |
| 5 | `05_convergence_analysis.md` | 6 | Quality | (existing) |
| 5 | `05_loss_function.md` | 4 | Quality | (existing) |
| 6 | `06_stream_overlap.md` | 4 | Perf | 3 |
| 6 | `06_staged_training.md` | 5 | Quality | (existing) |
| 7 | `07_literature_comparison.md` | 4 | Perf | 4 |
| 8 | `08_recommendations.md` | 3 | Perf | 4 |
| 9 | `09_references.md` | 2 | Perf | 4 |

All documents committed to the `qwen-palettize` repository
(https://github.com/pkhairkh/qwen-palettize) under the `main` branch.
Performance stream organised in 4 waves with git commits and pushes
after each wave; quality stream added by a parallel research effort.

