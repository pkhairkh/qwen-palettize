# 00 — Executive Summary: Gap Analysis

**Question.** Our 2-bit per-group (GS=256) palettization with Gumbel-Softmax trainable indices + LoRA reaches cos=0.95 on Qwen3.5-4B (super-block 0). The published literature reaches cos>0.999 at similar or lower bitwidths. **What are we missing?**

**Answer (one sentence).** The gap is structural, not parametric: our 4-entry single-weight codebook at GS=256 lacks the resolution to fit LLM weight distributions, and no amount of Gumbel-Softmax training or LoRA compensation can break the resulting accuracy ceiling — we must adopt at least one of: vector quantization (`g=2`, GPTVQ), additive multi-codebook (`M=2`, AQLM), Hadamard pre-rotation (QuIP#), or dense/sparse outlier removal (SqueezeLLM).

---

## 1. The Five-Number Summary

| Metric | Our approach | Published SOTA at 2-bit | Gap |
|---|---|---|---|
| Bitwidth | 2.0 bit (K=4) | 2.0 bit (AQLM M=2, QuIP#, GPTVQ g=2) | Same |
| Group size | 256 | 64–128 (GPTVQ, AQLM use 64 or per-channel) | 2–4× larger |
| Final cos (super-block output) | 0.95 | >0.999 (AQLM), >0.99 (QuIP#, GPTVQ) | **0.04–0.05** |
| Training cost | 8000 steps × 1s = ~2.2h | PTQ-only: ~3–4h (QuIP#, GPTVQ); PTQ+QAT: ~12h (AQLM) | Comparable |
| Production-readiness | Tier 3 (research code, 1 GPU) | Tier 1–2 (llama.cpp, AutoGPTQ, HF integration) | Large |

The 0.04–0.05 cos gap is the central finding. It corresponds to ~3–5× higher output reconstruction error than SOTA, which translates to measurable perplexity degradation and noticeable generation quality loss.

---

## 2. The Six Structural Gaps

Detailed analysis is in `07_gap_analysis.md`; the executive summary is below.

### Gap 1: No pre-quantization transformation (the AWQ/SmoothQuant gap)

**What we do:** k-means operates on raw weights `W_orig`. The palette adapts to the weight-space distribution but ignores that channels matter unequally for output reconstruction.

**What SOTA does:** SmoothQuant, AWQ, OmniQuant, AffineQuant all apply an invertible per-channel transformation `T(·)` before quantization, chosen so that `T(W)` fits the grid better. AWQ's per-channel scale `s[j] = √(max|x[j]| / max|W[j,:]|)` is a 10-line change.

**Expected gain:** +0.01–0.03 cos.

### Gap 2: No Hessian-inverse error propagation (the GPTQ gap)

**What we do:** k-means uses only the Hessian *diagonal* `H_ww = 2 Σ_i x²[i,w]` as per-weight importance. The full Hessian `H = Xᵀ X` is computed (line 84 of `palettize_core.py`) and immediately discarded (line 85 takes only the diagonal).

**What SOTA does:** GPTQ, GPTVQ, QuIP, QuIP# all use the *full inverse Hessian* `H⁻¹` to propagate the residual error of each quantized weight onto the remaining weights. The update `W[remaining] -= e · H⁻¹[remaining, current] / H⁻¹[current, current]` is a 20-line addition to our calibration loop.

**Expected gain:** +0.01–0.02 cos (calibration).

### Gap 3: Single-weight codebook (the GPTVQ/AQLM gap) — **THE bottleneck**

**What we do:** Each weight is independently assigned to one of 4 codebook entries (K=4, g=1).

**What SOTA does:**
- **GPTVQ** jointly quantizes blocks of `g=2–8` weights to a shared codebook of `K^g` entries. At `g=2`, effective codebook is 16 (vs. our 4) at the same 2-bit storage.
- **AQLM** composes `M=2` codebooks of K=256 each, giving an effective codebook of `K^M = 65536` entries. At 2-bit storage per weight, this is the published SOTA (cos>0.999).
- **QuIP#** applies a Hadamard pre-rotation that drives weight coherence to random-matrix levels, making the 4-entry grid fit much better.

**Expected gain:** +0.03–0.06 cos (potentially closes the gap entirely).

**This is the single most important gap.** All three techniques attack the same fundamental limit: 4 entries per group of 256 weights is too coarse for LLM weight distributions, *regardless* of how those 4 entries are chosen. Our Gumbel-Softmax + LoRA can refine the 4 entries but cannot make 4 behave like 16 or 65K.

### Gap 4: No outlier removal (the SqueezeLLM gap)

**What we do:** All weights go into the dense 2-bit codebook, including the ~0.5% of high-sensitivity outlier weights that LLMs are known to have.

**What SOTA does:** SqueezeLLM peels off the top 0.45% of weights by sensitivity `s = (w - q)² · H_ww` and stores them in a sparse FP16 residual. The dense majority then quantizes much better (the codebook no longer wastes a level on outliers).

**Expected gain:** +0.02–0.04 cos.

### Gap 5: Group size too large (the AutoGPTQ gap)

**What we do:** GS=256.

**What SOTA does:** GS=128 (AutoGPTQ default, GPTQ standard), GS=64 (SqueezeLLM, AffineQuant aggressive). Some methods (AQLM, QuIP#) use per-channel.

**Expected gain:** +0.005–0.01 cos (halving GS from 256 → 128). Trivial to change.

### Gap 6: Gumbel-Softmax has known gradient damping (the LLT/LUT-Q gap)

**What we do:** Gumbel-Softmax relaxation with linear τ anneal (2.0 → 0.1 over 4000 steps). At low τ, the soft assignment `p = softmax((logits + gumbel)/τ)` becomes numerically one-hot, and the gradient `∂p/∂logits` vanishes. This is documented in `research-indices-training/03_gradient_flow_analysis.md`.

**What SOTA does:**
- **LUT-Q** does not differentiate indices at all — it maintains an FP shadow `W` and recomputes indices via k-means every step. No gradient damping because there's no gradient on indices.
- **LLT** uses deterministic softmax (no Gumbel noise) plus `1/√(N_k)` gradient rescaling. The rescaling prevents codebook collapse, the missing piece in plain softmax.
- **AQLM** uses beam search (discrete, no relaxation) + STE.

**Expected gain:** +0.01–0.02 cos (if we switch to LUT-Q-style FP shadow + k-means reassignment) or +0.005–0.015 (if we keep Gumbel but add `1/√(N_k)` rescaling and drop the noise).

---

## 3. The "Minimum Viable SOTA" Path

If we could adopt only **three** techniques, in priority order:

1. **GPTVQ vector quantization `g=2`** (Gap 3). +0.03–0.05 cos. Medium implementation cost (2D k-means + kernel variant).
2. **AWQ-style per-channel scale** (Gap 1). +0.01–0.03 cos. Small implementation cost (10 lines in `palettize_core.py`).
3. **SqueezeLLM dense/sparse split** (Gap 4). +0.02–0.04 cos. Medium implementation cost (sparse matmul + CSR).

**Combined expected cos:** 0.95 + 0.07 ± 0.04 = **~0.99–0.999**. This matches the published SOTA at 2-bit.

If we can adopt a fourth: **QuIP# Hadamard pre-rotation** (also attacks Gap 3 from a different angle). Small implementation cost (fast Hadamard transform). Expected additional gain: +0.02–0.04 cos.

---

## 4. What We Should NOT Do

Several tempting changes are *not* recommended, based on the literature review:

1. **Do not increase LoRA rank.** Our LoRA rank-16/32 is already in the QLoRA-standard range. Going to rank-64+ would help marginally (~0.005 cos) but does not address the structural codebook-resolution limit. Diminishing returns.

2. **Do not extend Gumbel-Softmax training.** Our 8000 steps already exhaust what Gumbel-Softmax can achieve at K=4 (the gradient damping at low τ is fundamental, not a hyperparameter issue). Going to 50K steps would not break the cos=0.95 ceiling.

3. **Do not switch to from-scratch training** (BitNet-style) without first trying the structural fixes. From-scratch is expensive (~weeks of GPU time) and may not even help at our scale (Qwen3.5-4B is in the "small model" regime where BitNet's results are weaker). Try structural fixes first; if they don't reach cos>0.99, then consider from-scratch.

4. **Do not invest in productionization** (framework integration, multi-platform kernels) until the accuracy gap is closed. Productionizing a cos=0.95 model is premature; the deployment value of cos=0.95 2-bit is limited.

5. **Do not add more loss terms** (KL divergence, intermediate-layer distillation, attention-map matching). Our `1-cos + norm_mse` is already a reasonable distillation loss; the gap is not in the loss function but in the codebook structure.

---

## 5. Implementation Priority and Effort

| Priority | Technique | Gap | Effort (eng-days) | Expected cos gain |
|---|---|---|---|---|
| **P0** | Halve GS: 256 → 128 | Gap 5 | 0.5 | +0.005–0.01 |
| **P0** | GPTVQ `g=2` vector quantization | Gap 3 | 5–7 | +0.03–0.05 |
| **P0** | AWQ per-channel scale | Gap 1 | 1–2 | +0.01–0.03 |
| **P0** | SqueezeLLM dense/sparse split | Gap 4 | 3–5 | +0.02–0.04 |
| **P1** | LLT `1/√(N_k)` gradient rescale | Gap 6 | 1 | +0.005–0.015 |
| **P1** | Drop Gumbel noise (deterministic-ST) | Gap 6 | 0.5 | +0.005–0.01 |
| **P1** | GPTQ Hessian-inverse error propagation | Gap 2 | 2–3 | +0.01–0.02 |
| **P1** | QuIP# Hadamard pre-rotation | Gap 3 (alt) | 2 | +0.02–0.04 |
| **P2** | Tighten logit clamp `±20` → `±5τ` | Gap 6 | 0.1 | +0.005–0.015 |
| **P2** | Lloyd-Max Gaussian codebook init | — | 0.1 | +0.005–0.01 |
| **P2** | Learnable clipping threshold (OmniQuant) | — | 1 | +0.005–0.015 |
| **P3** | LUT-Q FP shadow + k-means reassignment | Gap 6 (replaces Gumbel) | 5–7 | +0.01–0.02 |
| **P3** | ExLlamaV2 per-layer mixed bitwidth | — | 5 | +0.01–0.03 (at same avg bw) |

**Total P0+P1 effort:** ~15–22 eng-days. Expected total cos gain: +0.07 to +0.13. This should take us from cos=0.95 to cos>0.99, with a realistic path to cos>0.999 if all gains compound.

---

## 6. Reading Guide

The remaining files in this literature review provide the detailed support for the executive summary above:

- **`01_gptq_family.md`** — GPTQ, AutoGPTQ, GPTVQ, PyGPT. The closed-form second-order family. Motivates Gap 2 (Hessian-inverse) and Gap 3 (vector quantization).
- **`02_awq_smoothquant.md`** — AWQ, SmoothQuant, OmniQuant, AffineQuant. The pre-quantization transformation family. Motivates Gap 1 (per-channel scale).
- **`03_codebook_methods.md`** — LUT-Q, LLT, SqueezeLLM, QuIP#, AQLM, GPTVQ. The codebook family. Motivates Gap 3 (additive/vector), Gap 4 (dense/sparse), Gap 6 (training recipe).
- **`04_1bit_methods.md`** — BitNet, BitNet b1.58, BNN. The 1-bit from-scratch family. Motivates the from-scratch question (not recommended yet).
- **`05_production_frameworks.md`** — QLoRA, llama.cpp, ExLlamaV2. Production deployment. Motivates the productionization question (deferred).
- **`06_comparison_table.md`** — All 20 methods × 10 dimensions in one table. The reference.
- **`07_gap_analysis.md`** — Detailed gap-by-gap analysis with code-level specifics.
- **`08_recommendations.md`** — Prioritized implementation plan with concrete code patches.
- **`09_formal_review.md`** — Academic-style literature review (abstract, intro, background, methods, discussion, conclusion).
- **`10_references.md`** — Complete bibliography with arxiv URLs and years.

---

## 7. Bottom Line

**The cos=0.95 plateau is real and structural.** Our Gumbel-Softmax + LoRA + cos-loss approach is a reasonable "kitchen sink" design that combines ideas from SqueezeLLM (k-means), LLT (soft assignment), QLoRA (LoRA), and BitNet (FP shadow + STE). But it doesn't fully commit to any single approach, and it misses the structural codebook improvements (vector quantization, additive composition, Hadamard pre-rotation) that the SOTA at 2-bit uses.

**The path forward is clear:** adopt GPTVQ `g=2`, AWQ per-channel scaling, and SqueezeLLM dense/sparse split. Together, these should close ~80% of the gap (to cos>0.99) in ~2 weeks of engineering. Adding QuIP# Hadamard pre-rotation should close the remaining ~20% (to cos>0.999).

**The path NOT recommended:** continuing to tune Gumbel-Softmax hyperparameters (τ schedule, LR, gradient clipping). These have been explored extensively in the existing `research-indices-training/` reports and have hit their ceiling. Further tuning will not break cos=0.95.
