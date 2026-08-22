# 00 — Executive Overview

**Repository audited:** `pkhairkh/qwen-palettize` — 2-bit LUT-quantized training of Qwen3.5-4B with Gumbel-Softmax trainable indices + LoRA rank-16/32.

**Target hardware:** NVIDIA Blackwell sm_120 (RTX PRO 6000, 96 GB VRAM).

**Symptom:** Forward cos=0.947 with teacher; training plateaus at cos≈0.95; literature (GPTQ, AWQ, SqueezeLLM, QuIP#, AQLM, VPTQ, GPTVQ) routinely achieves cos>0.999 at 2-4 bits.

---

## 1. TL;DR

The cos plateau at 0.946 is **NOT a kernel-numerics problem**. The CUDA kernels themselves are numerically sound — the hard forward path supports cos > 0.999. The plateau is the **2-bit scalar LUT representation limit** (calibration cos = 0.937), marginally improved by LoRA (+0.009 cos). The Gumbel-Softmax index training is a **no-op** at low temperature (vanishing gradient, mathematically proven), and the kernel precision issues (bf16 grad_W, fp16 P/grad_logits) account for only ~0.001 cos of the gap.

**To break the plateau, the repo must adopt techniques from the literature:** GPTQ-style second-order calibration (+2-4% cos), outlier isolation (+1-3% cos), and/or vector quantization (+4-6% cos). Pure kernel tuning will not help.

---

## 2. The audit found 10 issues, ranked by impact

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

---

## 3. The kernel audit verdict

**Hard forward kernel** (`fused_lut_linear_fwd_kernel`, lines 95–288 of `fused_lut_kernel.cu`):
- Uses scalar fp32 FMA (dequantize bf16→fp32, multiply in fp32, accumulate in fp32).
- Numerically MORE accurate than the TC variant.
- Per-element relative error: ~8e-3 (bf16 output quantization dominates).
- **Theoretical cos ceiling: >0.9999.**

**Hard TC forward kernel** (`fused_lut_linear_fwd_tc_kernel`, lines 309–518):
- Uses `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`.
- bf16×bf16 multiply with fp32 accumulation.
- Per-element relative error: ~5e-3 (mma multiply rounding).
- **Theoretical cos ceiling: >0.99999.**

**Soft forward kernel** (`fused_lut_linear_soft_compute_P_W_kernel`, lines 1301–1352):
- Gumbel-Softmax with LCG-based PRNG (statistically weak).
- P stored as fp16 (denormal underflow at τ ≤ 0.5).
- W computed in fp32, stored as bf16.
- **Forward numerically fine; the issue is the backward (see below).**

**Soft backward kernels** (lines 1364–1434, 1498–1610):
- Gradient formula is mathematically correct (proven in 03_ste_analysis.md).
- BUT: the Python backward (which is what actually runs) uses bf16 grad_W and fp16 grad_logits.
- The CUDA soft backward kernels are NOT used in training — the Python path computes everything via `torch.matmul` + PyTorch elementwise ops.
- **The kernel precision issues are in the Python backward, not the CUDA kernels.**

---

## 4. The STE verdict

The Straight-Through Estimator `W = W_hard - W_soft.detach() + W_soft` is **mathematically correct**:
- Forward: evaluates to `W_hard` (exact one-hot).
- Backward: gradient routes through `W_soft` (the only differentiable term).
- Gradient formula: `grad_logits = grad_W * P * (c - W_soft)` is the correct Jacobian.

**However, the STE is ineffective at low temperature** (mathematically proven):
- At τ ≤ 0.1, P becomes one-hot, so `P[k] = 0` for non-argmax k.
- For the argmax k, `(c[k] - W_soft) = (c[k] - c[k]) = 0`.
- Therefore `grad_logits = 0` for all k.
- The developers' own comment confirms: *"all 25 index_logits grads are 0.0."*

**The fix is to floor τ at 0.5** (not 0.1), maintaining gradient flow throughout training. Combined with fp32 storage for P and grad_logits, this would enable actual index training — but the upside is bounded at ~1% cos by the 2-bit scalar LUT ceiling.

---

## 5. The literature verdict

The qwen-palettize repo's approach (k-means + Gumbel-Softmax + LoRA) is **missing all six techniques** that the literature uses to achieve cos > 0.99 at 2 bits:

1. **Second-order compensation** (GPTQ, SpQR, OWQ, GPTVQ, VPTQ) — adds +2-4% cos.
2. **Outlier isolation** (LLM.int8, SpQR, OWQ, SqueezeLLM) — adds +1-3% cos.
3. **Vector quantization** (QuIP#, GPTVQ, AQLM, VPTQ) — adds +4-6% cos, fundamental change.
4. **Incoherence preprocessing** (QuIP, QuIP#) — adds +0.5-1% cos.
5. **Activation-aware scaling** (AWQ) — adds +0.5-1% cos.
6. **Properly-implemented trained indices** (AQLM) — adds +1-2% cos.

The closest analog is **AQLM** (ICML 2024), which also trains codebook indices via gradient descent. AQLM achieves cos > 0.99 at 2 bits/weight by using:
- Additive VQ (2 codebooks × 256 entries = 65536 distinct values per block, vs 4 for qwen-palettize).
- Direct STE without softmax relaxation (no vanishing gradient).
- Block-MSE loss (strong gradient signal).
- GPTQ-style initialization before gradient descent.
- 100K-500K training steps.

The qwen-palettize repo's approach (scalar 2-bit + Gumbel-Softmax + cosine loss + 8K steps) is missing all five ingredients.

---

## 6. Recommended action plan

### 6.1 Short-term (1-2 weeks, +0.01-0.03 cos)

These are low-effort changes that can be applied to the existing codebase without major refactoring:

1. **Floor τ at 0.5** in the Gumbel-Softmax training schedule (was 2.0 → 0.1; change to 1.0 → 0.5).
2. **Switch to fp32 for P and grad_logits** in the soft kernels (currently fp16).
3. **Switch to fp32 for grad_W** in the Python backward (currently bf16).
4. **Add AWQ-style activation-aware scaling** at calibration time.
5. **Switch loss from pure cosine to 50/50 cosine + block-MSE**.
6. **Train for 50K steps** (currently 8K) with cosine LR decay.

Expected cos: 0.946 → ~0.96.

### 6.2 Medium-term (1-2 months, +0.02-0.04 cos)

These require more substantial changes but stay within the scalar LUT framework:

7. **Add GPTQ-style second-order calibration** as a post-k-means refinement step.
8. **Add outlier isolation** — extract top 0.5% of weights into a sparse FP16 matrix.
9. **Add RHT incoherence preprocessing** (QuIP-style) before palettization.

Expected cos: 0.96 → ~0.98.

### 6.3 Long-term (3-6 months, +0.02-0.06 cos)

These require fundamental algorithmic changes, switching from scalar 2-bit LUT to VQ:

10. **Switch from scalar 2-bit LUT to 8-dim VQ codebook** (QuIP#-style E8 lattice, or AQLM-style additive VQ with K=2 codebooks × 256 entries).
11. **Replace Gumbel-Softmax STE with AQLM-style direct STE** (no softmax, no vanishing gradient).
12. **Rewrite the CUDA kernels** to support VQ codebook lookups (currently they do scalar `spalette[group][idx_val]` lookups).
13. **Add per-block end-to-end training** (joint codebook + indices + LoRA).

Expected cos: 0.98 → >0.99.

---

## 7. What NOT to do

Based on the audit, the following changes would be **wasted effort**:

1. **Tuning the hard kernel TC variant** — the TC variant is already numerically fine (cos > 0.99999). Optimizing it further (e.g., WGMMA on Blackwell) would not change the plateau.
2. **Increasing LoRA rank beyond 32** — diminishing returns; rank-256 LoRA still leaves cos < 0.97 due to the high-rank nature of the quantization error.
3. **Tuning the Gumbel PRNG** — replacing the LCG with a better PRNG (e.g., Philox) would not change the vanishing-gradient behavior.
4. **Adding more calibration sequences** — the calibration is already near-optimal for k-means; more sequences would not push past the scalar LUT ceiling.
5. **Switching to bf16 P instead of fp16 P** — this would help slightly (no denormal underflow), but the dominant issue is the scalar LUT ceiling, not P precision.

---

## 8. Document map

This overview is the entry point. The detailed analyses are in the companion documents:

| Document | Content | Pages |
|----------|---------|-------|
| `01_kernel_audit.md` | Line-by-line audit of `fused_lut_kernel.cu` | 9.7 |
| `02_numerical_analysis.md` | fp16/bf16/fp32 precision analysis | 7.0 |
| `03_ste_analysis.md` | STE correctness proofs + alternatives | 5.2 |
| `04_literature_comparison.md` | GPTQ/AWQ/SqueezeLLM/QuIP#/AQLM/VPTQ/GPTVQ comparison | 7.1 |
| `05_convergence_analysis.md` | Root-cause analysis of the 0.946 plateau | 5.9 |
| `06_recommendations.md` | Concrete code patches (Python + CUDA) | 5.0 |
| `07_references.md` | arxiv papers, GitHub repos, blog posts | 2.5 |
| **Total** | | **~42 pages** |

Each document is self-contained and can be read independently. The recommended reading order is: this overview → 01 (kernel audit) → 05 (convergence) → 04 (literature) → 03 (STE) → 02 (numerical) → 06 (recommendations) → 07 (references).

---

## 9. Bottom line

The qwen-palettize repo's cos plateau at 0.946 is a **representation problem, not a numerical problem**. The 2-bit scalar LUT scheme has a fundamental ceiling at cos ~0.94, and the LoRA + palette training can only recover ~1% above that ceiling. The Gumbel-Softmax index training is a no-op at low temperature, but even if it worked perfectly, it would add only ~1% cos.

**To break the plateau, the repo must adopt VQ codebooks (QuIP#-style E8 lattice or AQLM-style additive VQ) and GPTQ-style second-order calibration.** Pure kernel tuning (fp32 vs bf16, TC vs scalar, etc.) cannot break the ceiling. The path forward is well-documented in the literature — the repo just needs to adopt it.
