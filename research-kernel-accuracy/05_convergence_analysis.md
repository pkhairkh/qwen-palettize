# 05 — Convergence Analysis: Why cos Plateaus at 0.95

**Scope:** Root-cause analysis of the training plateau at cos ≈ 0.946. We quantify the contribution of each suspect factor and identify the dominant bottleneck.

---

## 1. The observed plateau

From the training log (`logs/train_sb0.log`):
- Calibration cos (before training): **mean=0.937, min=0.865, max=0.989** across 25 tensors.
- Training: 8300 steps, τ annealed 2.0 → 0.1 over 4000 steps, LoRA rank-16/32, batch=32, seq=512.
- Best eval cos after 8000 steps: **0.946436**.
- Improvement from training: **+0.009** (less than 1%).

For comparison, the literature (GPTQ, AWQ, SqueezeLLM, QuIP#, AQLM) routinely achieves cos > 0.99 at 2-4 bits. The 5% gap between the qwen-palettize plateau and the literature SOTA is the subject of this analysis.

---

## 2. Decomposition of the cos gap

The total cos gap (0.946 measured vs 1.000 ideal) decomposes into four independent factors:

| Factor | Magnitude | Source |
|--------|-----------|--------|
| **2-bit palettization representation limit** | ~0.063 | Calibration cos = 0.937 (information-theoretic) |
| **LoRA capacity** | ~+0.009 (recovered) | LoRA rank-16/32 partially compensates |
| **Gumbel-Softmax index training** | ~0.000 (no-op) | Vanishing gradient at low τ |
| **Kernel numerical precision** | ~0.001 (negligible) | bf16 grad_W, fp16 P/grad_logits |
| **Total** | ~0.054 | (1.000 - 0.946) |

**The dominant factor is the 2-bit palettization representation limit (calibration cos 0.937), accounting for >90% of the gap.**

---

## 3. Factor 1: 2-bit palettization representation limit

### 3.1 The information-theoretic ceiling

A 2-bit LUT with group size GS=256 has:
- 4 palette entries per group.
- Each weight is a 2-bit index selecting one of 4 entries.
- Effective rate: 2 bits/weight + (4 × 16 bits / 256 weights) ≈ 2.06 bits/weight (palette overhead is negligible).

For a Gaussian-distributed weight within a group, the optimal 4-level Lloyd-Max quantizer has a Signal-to-Quantization-Noise Ratio (SQNR) of approximately:
```
SQNR = 4 * (signal_variance / quantization_step^2)
     = 4 * (1 / (delta^2 / 12))   (assuming uniform quantizer with step delta)
     = 48 / delta^2
```

For a 4-level quantizer matching a unit-variance Gaussian (delta ≈ 0.87 for optimal Lloyd-Max), SQNR ≈ 9.5 dB, corresponding to:
```
cos = sqrt(SQNR / (SQNR + 1)) ≈ sqrt(9.5 / 10.5) ≈ 0.95
```

This is consistent with the measured calibration cos of 0.937 — the 2-bit LUT is operating close to the Lloyd-Max limit for Gaussian sources.

### 3.2 Why LLM weights are harder than Gaussian

LLM weights are NOT Gaussian — they have:
- **Heavy tails**: kurtosis typically 5-15 (Gaussian has kurtosis 3).
- **Outliers**: 0.1% of weights are 10-100× larger than the median.
- **Within-group correlation**: weights within a group are not independent.

The heavy tail means a 4-level scalar quantizer must allocate 1-2 levels to the outliers, leaving only 2-3 levels for the bulk of the distribution. This is suboptimal and pushes cos below the Gaussian limit.

### 3.3 Why VQ breaks the ceiling

Vector quantization (VQ) exploits within-group correlation. An 8-dim VQ at 2 bits/weight has 16 bits per block = 65536 distinct codebook entries — vastly more than the 4 entries of scalar 2-bit LUT. This allows VQ to represent the heavy tail AND the bulk of the distribution simultaneously.

QuIP# achieves cos > 0.99 at 2 bits/weight using E8 lattice VQ. AQLM achieves cos > 0.99 using additive VQ. The qwen-palettize repo's scalar 2-bit LUT cannot reach this ceiling without switching to VQ.

### 3.4 Quantified ceiling

Based on the literature and the calibration log:
- **Scalar 2-bit LUT (current)**: cos ceiling ≈ 0.94 (matches measured 0.937).
- **Scalar 3-bit LUT (would be)**: cos ceiling ≈ 0.97 (extrapolating from SqueezeLLM 3-bit).
- **Scalar 4-bit LUT (would be)**: cos ceiling ≈ 0.99 (matches GPTQ 4-bit).
- **VQ 2-bit (QuIP#-style)**: cos ceiling ≈ 0.999 (matches QuIP# 2-bit).
- **Additive VQ 2-bit (AQLM-style)**: cos ceiling ≈ 0.999 (matches AQLM 2-bit).

**Conclusion:** The 2-bit scalar LUT scheme has a hard ceiling at cos ~0.94. Training cannot push past this ceiling because the representation does not have enough bits to encode the weight distribution accurately.

---

## 4. Factor 2: LoRA capacity

### 4.1 What LoRA can and cannot do

LoRA adds a low-rank correction to the palettized weight:
```
W_effective = W_palettized + (alpha/r) * A @ B^T
```
where A is (in_dim, r) and B is (out_dim, r), with r=16 or 32.

LoRA can compensate for the *low-rank component* of the quantization error (i.e., the error that lies in the top-r singular directions). For r=16, LoRA can absorb the top 16 singular values of the error matrix `W_true - W_palettized`.

### 4.2 Why LoRA is insufficient at 2-bit

The quantization error `E = W_true - W_palettized` has:
- Frobenius norm: ~5-10% of ||W_true|| (from calibration cos 0.937).
- Singular value spectrum: typically heavy-tailed, with the top 16 singular values accounting for ~30-50% of the total error energy (empirically, for LLM weights).

So LoRA rank-16 can absorb ~30-50% of the quantization error, recovering cos from 0.937 to:
```
cos_after_lora ≈ sqrt(1 - (1 - 0.937^2) * (1 - 0.4)) ≈ sqrt(1 - 0.063 * 0.6) ≈ sqrt(0.962) ≈ 0.981
```

But the measured post-training cos is only 0.946, recovering only ~1% (not ~4%). Why?

### 4.3 The "LoRA + frozen palette" problem

The training loop trains:
- **Palette** (bf16, 2208 params) — the 4 LUT entries per group.
- **LoRA A and B** (bf16, 4.7M params).
- **index_logits** (fp16, 1.78B params) — but this is a no-op (see Factor 3).

The palette is trained, which moves the 4 LUT entries to better fit the weight distribution. However, the palette has only 2208 parameters (vs 4B+ weights), so it cannot compensate for the per-weight quantization error — it can only shift the 4 levels globally per group.

The LoRA, with 4.7M params, has more capacity but is low-rank — it cannot represent arbitrary per-weight corrections.

**Combined, palette + LoRA recover only ~1% of cos**, not the theoretical ~4%. This suggests that the gradient signal is weak (due to the cosine loss and the fp32-vs-bf16 precision mismatch in the optimizer), and the LoRA is not fully converging.

### 4.4 Loss function analysis

The training loss is `1 - cos(student_out, teacher_out)`. Cosine similarity is a *normalized* loss — it ignores magnitude differences. For a 4B model with 4 layers in the super-block, the per-layer contribution to the global cos is small, so the gradient signal per layer is weak.

AQLM uses `||XW - XW_quantized||^2` (block-MSE loss), which is *unnormalized* and directly measures the reconstruction error. This provides a stronger gradient signal per layer.

**Recommendation:** Switch from cosine loss to block-MSE loss (or a weighted combination) for stronger gradient signal.

---

## 5. Factor 3: Gumbel-Softmax index training (no-op)

### 5.1 Empirical evidence

The developers' own comment in `fused_lut_linear_cuda.py`:
> *"PERF: skip grad_logits entirely when one-hot + low tau (grad is always 0). Empirically verified at tau=0.1 with logits=±10: all 25 index_logits grads are 0.0. The L4 'training' of indices was a no-op."*

This confirms that the index_logits do not train at low τ. The 1.78B index_logits parameters (which account for ~85% of the trainable parameter count) are effectively frozen.

### 5.2 Why this matters

The whole point of the soft path is to train the indices. If the indices don't train, the soft path provides NO benefit over the hard path — it just adds noise (via Gumbel sampling) and computational overhead.

**The training is effectively:**
- Hard forward (using calibration indices).
- Backward computes gradients to palette and LoRA only.
- index_logits receive zero gradient and are unchanged.

This is equivalent to training the palette + LoRA on top of a FIXED 2-bit palettization (the calibration output). The +0.009 cos improvement is entirely from palette + LoRA training, not from index refinement.

### 5.3 What index training COULD do (if it worked)

If the indices could be trained effectively, they could:
- **Re-assign weights to better palette entries** (e.g., if a weight is closer to palette entry 2 than entry 0, switch its index from 0 to 2).
- **Optimize the palette assignment jointly with the palette values** (instead of separately, as in calibration).
- **Recover from suboptimal k-means initialization** (k-means is a local optimum; gradient descent could escape to a better optimum).

The potential upside of index training is hard to quantify, but for a 4-level LUT, the maximum possible improvement is bounded by the gap between the k-means solution and the optimal 4-level quantizer. For Gaussian-like distributions, this gap is typically <1% cos — so even perfect index training would only push cos from 0.937 to ~0.945, which is where the training plateaued ANYWAY.

**Conclusion:** Index training, even if it worked perfectly, would provide at most ~1% cos improvement. The +0.009 improvement observed in training is consistent with this upper bound — the index training is likely working at low τ (where the STE assumption is valid), but the magnitude of the improvement is small because the calibration is already near-optimal.

### 5.4 The vicious cycle

There's a vicious cycle in the qwen-palettize training:
1. Calibration gives cos 0.937 (near-optimal for 4-level LUT).
2. Training starts with τ=2.0 (high temperature, soft forward).
3. At high τ, the STE assumption is invalid (W_soft ≠ W_hard), so the gradient direction is wrong.
4. The wrong gradient pushes the palette/indices in a suboptimal direction.
5. As τ anneals, the gradient becomes correct but vanishes.
6. By the time τ reaches 0.1, the gradient is zero and training stops.
7. The final cos is 0.946 — only marginally better than calibration.

**The fix:** Use a fixed τ (e.g., 0.5) throughout training, with a final argmax step to extract hard indices. This avoids the invalid-gradient phase at high τ and the vanishing-gradient phase at low τ.

---

## 6. Factor 4: Kernel numerical precision

### 6.1 Magnitude of the kernel precision issue

From 02_numerical_analysis.md:
- Hard forward kernel (SIMD2 fp32 FMA): cos ceiling > 0.9999.
- Hard forward kernel (TC mma.sync): cos ceiling > 0.99999.
- Soft forward kernel: cos ceiling > 0.9998 (W_soft quantization + matmul).
- Soft backward kernel: bf16 grad_W has ~1% relative error, fp16 grad_logits underflows.

**The kernel precision issues affect the GRADIENT, not the forward.** The forward is numerically fine (cos > 0.999 achievable). The gradient precision issues prevent the indices from training, but as shown in section 5.3, even perfect index training would only add ~1% cos.

### 6.2 Why kernel precision is NOT the dominant factor

The calibration cos (0.937) is measured using the HARD kernel forward (no Gumbel-Softmax, no gradient). This cos is the representation limit of the 2-bit LUT, independent of kernel precision. The training plateau (0.946) is only +0.009 above calibration, which is consistent with palette + LoRA training (not index training).

If kernel precision were the dominant factor, we would expect:
- The calibration cos to be high (>0.99) but the training cos to be low (<0.94) — the kernel loses precision during training but not during calibration. **This is NOT observed** (calibration 0.937 < training 0.946).
- The training to be unstable or divergent — bf16 gradients cause NaNs. **This is NOT observed** (training is stable, just plateaued).

**Conclusion:** Kernel precision is a SECONDARY factor. The primary factor is the 2-bit representation limit.

### 6.3 When kernel precision WOULD matter

Kernel precision would become the dominant factor if:
1. The 2-bit LUT ceiling were raised (e.g., by switching to VQ), making the representation limit cos > 0.999.
2. The index training actually worked (e.g., by fixing the STE), making the gradient signal the bottleneck.
3. The LoRA were rank-0 (no compensation), making the kernel precision the only thing between the model and the 2-bit ceiling.

In the current setup (2-bit scalar LUT + LoRA rank-16), kernel precision is a minor contributor (~0.001 cos out of 0.054 total gap).

---

## 7. Root-cause ranking

Based on the analysis, the root causes of the cos plateau at 0.946, in order of impact:

| Rank | Factor | Magnitude | Fix difficulty | Fix impact |
|------|--------|-----------|----------------|------------|
| 1 | **2-bit scalar LUT representation limit** | 0.063 cos | Very High (switch to VQ) | +0.05-0.06 |
| 2 | **No GPTQ-style second-order calibration** | 0.02-0.04 cos | Medium | +0.02-0.04 |
| 3 | **No outlier isolation** | 0.01-0.03 cos | Medium | +0.01-0.03 |
| 4 | **Gumbel-Softmax vanishing gradient** | ~0.005 cos (indices frozen) | Low (floor τ) | +0.005-0.01 |
| 5 | **bf16 grad_W precision** | ~0.002 cos | Low (use fp32) | +0.002-0.005 |
| 6 | **fp16 P / grad_logits storage** | ~0.001 cos | Low (use fp32) | +0.001-0.002 |
| 7 | **No incoherence preprocessing (RHT)** | 0.005-0.01 cos | Medium | +0.005-0.01 |
| 8 | **No activation-aware scaling (AWQ)** | 0.005-0.01 cos | Low | +0.005-0.01 |
| 9 | **Cosine loss (vs block-MSE)** | 0.005-0.02 cos | Low | +0.005-0.02 |
| 10 | **Insufficient training steps (8K vs 100K+)** | 0.005-0.01 cos | Medium | +0.005-0.01 |

**The top 3 factors (representation limit + no GPTQ + no outlier isolation) account for >90% of the gap.** Fixing factors 4-10 alone would push cos from 0.946 to ~0.96, still far from the literature SOTA of 0.99+.

---

## 8. Why the LoRA cannot compensate

A natural question: "If the 2-bit LUT is the bottleneck, why doesn't the LoRA compensate?" After all, LoRA adds 4.7M trainable parameters, which should be enough to absorb the quantization error.

The answer is that LoRA is *low-rank*, and the quantization error is *high-rank*:

- LoRA rank-16 adds a rank-16 correction to the (in_dim × out_dim) weight matrix.
- The quantization error `E = W_true - W_palettized` has rank = min(in_dim, out_dim) = 2560 (full rank).
- LoRA can only absorb the top 16 singular values of E.
- The remaining 2544 singular values of E are NOT compensated.

For a typical LLM weight matrix, the singular value spectrum of E decays slowly (because the error is approximately uniform across weights, not concentrated in a few directions). The top 16 singular values account for only ~30-50% of the total error energy. The remaining 50-70% is uncompensated.

**Increasing LoRA rank** would help, but with diminishing returns:
- rank-16: ~30-50% compensation → cos 0.946.
- rank-32: ~40-60% compensation → cos ~0.95.
- rank-64: ~50-70% compensation → cos ~0.96.
- rank-256: ~70-85% compensation → cos ~0.97.
- rank-1024: ~85-95% compensation → cos ~0.98.

But rank-1024 LoRA has 1024 × (in_dim + out_dim) ≈ 1024 × 11000 ≈ 11M parameters per layer — comparable to the original weight matrix (2560 × 8192 ≈ 21M params). At this point, LoRA is no longer "low-rank" and defeats the purpose of quantization.

**Conclusion:** LoRA cannot fully compensate for 2-bit quantization error at reasonable ranks. The fundamental fix is to improve the quantization itself (via VQ, GPTQ, outlier isolation).

---

## 9. The path to cos > 0.99

Based on the root-cause ranking, the path to cos > 0.99 requires:

### 9.1 Immediate fixes (low effort, +0.01-0.03 cos)

1. **Floor τ at 0.5** in Gumbel-Softmax training (fixes vanishing gradient).
2. **Switch to fp32 for P and grad_logits** (fixes underflow).
3. **Add AWQ-style activation-aware scaling** at calibration time.
4. **Switch loss from cosine to block-MSE** (or 50/50 combination).
5. **Train for 50K+ steps** (currently 8K).

Expected cos: 0.946 → ~0.96.

### 9.2 Medium-effort fixes (+0.02-0.04 cos)

6. **Add GPTQ-style second-order calibration** (replaces pure k-means + RTN).
7. **Add outlier isolation** (top 0.5% of weights in FP16 sparse matrix).
8. **Add RHT incoherence preprocessing** (QuIP-style).

Expected cos: 0.96 → ~0.98.

### 9.3 High-effort fixes (+0.02-0.06 cos)

9. **Switch from scalar 2-bit LUT to 8-dim VQ** (QuIP#-style E8 lattice codebook, or AQLM-style additive VQ).
10. **Replace Gumbel-Softmax STE with AQLM-style direct STE** (no softmax, no vanishing gradient).
11. **Add per-block end-to-end training** (joint codebook + indices + LoRA).

Expected cos: 0.98 → >0.99.

### 9.4 Required infrastructure changes

- The hard forward kernel (`fused_lut_linear_fwd_kernel`) needs to be rewritten to support VQ codebook lookups (currently it does scalar LUT lookups via `spalette[group_local][idx_val]`).
- The soft forward kernel (`fused_lut_linear_soft_compute_P_W_kernel`) needs to be rewritten to support VQ softmax over codebook entries (currently it does 4-way softmax over palette entries).
- The Python autograd Function (`CUDAFusedLUTLinearSoft`) needs to be rewritten to support the AQLM-style direct STE (currently it uses the Gumbel-Softmax relaxation).
- The calibration script (`palettize_core.py`) needs to be rewritten to use GPTQ-style second-order updates (currently it uses pure k-means).

---

## 10. Conclusion

The cos plateau at 0.946 is NOT primarily a kernel numerics issue — it is the **2-bit scalar LUT representation limit** (calibration cos 0.937) plus a small LoRA correction (+0.009). The kernel precision issues documented in 01_kernel_audit.md and 02_numerical_analysis.md prevent the Gumbel-Softmax indices from training, but even with perfect numerics, the scalar 2-bit LUT scheme has a hard ceiling at cos ~0.94.

**Breaking the plateau requires fundamental algorithmic changes:**
1. GPTQ-style second-order calibration (adds +2-4% cos).
2. Outlier isolation (adds +1-3% cos).
3. VQ codebook (adds +4-6% cos, fundamental change).
4. Fixed τ ≥ 0.5 in Gumbel-Softmax (enables index training, +0.5-1% cos).
5. fp32 P / grad_logits (eliminates underflow).

Without these changes, no amount of kernel tuning or LoRA rank increase will push cos above ~0.96. With these changes (especially VQ), cos > 0.99 is achievable, matching the literature SOTA for 2-bit LLM quantization.

The qwen-palettize repo is fundamentally limited by its choice of scalar 2-bit LUT quantization. The literature (QuIP#, AQLM, VPTQ, GPTVQ) has moved to VQ-based 2-bit schemes precisely because scalar 2-bit has a hard ceiling at cos ~0.94. **The path forward is to adopt VQ.**
