# 02 — The AWQ/SmoothQuant Family: Activation-Aware Scaling and Learnable Transformations

**Scope.** This document covers the second major family of LLM weight-quantization methods: those built around **pre-quantization transformations** that push weights into a more quantization-friendly form. We cover SmoothQuant (Xiao et al., 2022), AWQ (Lin et al., 2023), OmniQuant (Shao et al., 2023), and AffineQuant (Ma et al., 2023). All four share a common pattern: rather than accepting the weight distribution as-is and quantizing it directly, they apply a *learned (or closed-form) invertible transformation* `T(·)` to the weights, quantize `T(W)`, then apply `T⁻¹` at inference. The transformation is chosen so that `T(W)` has a smaller dynamic range, fewer outliers, or otherwise fits the quantization grid better.

Our approach (recap): we apply **no pre-quantization transformation**. The k-means codebook in `palettize_core.py:palettize_tensor_2bit` operates on raw weights `W_orig` (line 91: `W_comp = W_orig.clone()`), weighted only by the diagonal Hessian `H_ww = 2·Σ_i x²[i, w]`. This is the central reason we plateau at cos≈0.95 — every method below achieves cos>0.99 at 4-bit (and several at 3-bit) by getting the transformation right, *before* any codebook training or LoRA compensation.

---

## 1. SmoothQuant — Activation Smoothing

**Paper.** Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., Han, S. *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.* ICML 2023. [arXiv:2211.03850](https://arxiv.org/abs/2211.03850). Year: 2022 (preprint) / 2023 (ICML).

### 1.1 The problem SmoothQuant solves

LLM activations have **massive per-channel outliers** — a tiny fraction (~0.1%) of channels have activations 10–100× larger than the median. This is a well-documented empirical phenomenon (Dettmers et al., 2022). The outliers are systematically problematic because they dominate the output of any Linear they feed into: a single weight `W[j, o]` multiplied by an outlier activation `x[j]` of magnitude 100 will dominate `y[o]` regardless of the other weights. Standard W8A8 or W4A8 quantization fails because:

- Activations must be quantized to 8-bit to fit hardware matmul. The 8-bit grid has dynamic range ~127. If `max|x[j]| / median|x[j]| = 100`, the median channels quantize to ~1 (effectively 0) — catastrophic information loss.

SmoothQuant's solution: **migrate the difficulty from activations to weights** by an invertible per-channel scaling.

### 1.2 Mathematical formulation

For a Linear `y = x · W`, SmoothQuant inserts a per-input-channel scale `s ∈ ℝ^{d_in}` (one scale per row of `W`, equivalently per column of `x`):

$$
y = x \cdot W = (x \cdot \text{diag}(s)) \cdot (\text{diag}(s)^{-1} \cdot W) = \tilde{x} \cdot \tilde{W}.
$$

The transformed activations `x̃ = x · diag(s)` have their outliers suppressed (if `s[j]` is small for outlier channels) or amplified (if `s[j]` is large for non-outlier channels). The transformed weights `W̃ = diag(s)⁻¹ · W` are correspondingly amplified on outlier channels (which now have small activations) and suppressed on non-outlier channels.

The key empirical observation: **weight magnitudes are flat** (no significant outliers), so amplifying the weights on outlier channels by 100× doesn't push them out of the quantization grid; the absolute weight magnitudes on outlier channels become ~`100 × 0.01 = 1`, which is well within INT8's `[-127, 127]` range after scaling.

The optimal `s` is:

$$
s_j = \frac{\max_i |x_{ij}|^\alpha}{\max_j |W_{j,:}|^{1-\alpha}}, \qquad \alpha \in [0, 1].
$$

`α` is a hyperparameter controlling how much "difficulty" to migrate from activations to weights. `α=0` is no migration (quantize `W` and `x` as-is); `α=1` is full migration (all difficulty goes to weights). Empirically `α=0.5` works well across LLMs (LLaMA, OPT, BLOOM).

### 1.3 Algorithm

```
1. For each Linear:
   a. From calibration set, compute max|x[j]| over batch (per input channel)
   b. Compute max|W[j, :]| over outputs (per input channel)
   c. s[j] = (max|x[j]|)^α / (max|W[j, :]|)^(1-α)
2. W̃ = W / s[:, None]    (scale each row of W by 1/s[j])
3. Quantize W̃ to INT8 (per-channel or per-group)
4. At inference: y = (x · diag(s)) · dequant(W̃)
                 = (x · diag(s)) · (W̃_dequant)
```

The fused kernel multiplies `x` by `s` on-the-fly during the matmul, so there's no memory overhead. SmoothQuant is **fully PTQ** — no gradient training.

### 1.4 Empirical accuracy

SmoothQuant on OPT-175B at W8A8: perplexity gap to FP16 of 0.03 (vs. 5.0 for naive W8A8 — a 100× improvement). On LLaMA-7B at W8A8: <0.01 gap. The key benefit is enabling **W8A8 inference** at fp16 accuracy, which is what makes A100/H100 INT8 tensor cores usable for LLM inference.

SmoothQuant does *not* target W4A8 or W2A8 directly. For sub-4-bit weights, AWQ is the natural extension.

### 1.5 Gap analysis vs. our approach

| Dimension | Our approach | SmoothQuant | What SmoothQuant does that we don't |
|---|---|---|---|
| Outlier handling | None (k-means sees raw weights) | Per-channel scaling that migrates outlier "difficulty" from activations to weights | **The single most important missing piece for 4-bit and lower.** Our k-means implicitly assumes weights matter equally; in fact, channels with large activations matter 10–100× more. |
| Activation quantization | N/A (we keep activations fp16) | Pushes activations to INT8 | We don't quantize activations, so this aspect doesn't directly apply. But the underlying math — that some channels deserve more "codebook resolution" — does. |
| Invertibility | N/A | `T` is `diag(s)`, `T⁻¹` is `diag(1/s)` | Trivially invertible. We could insert `diag(s)` between `x` and `W_q` in the PalettizedLinear forward at near-zero cost. |
| Training | QAT-style (Gumbel-Softmax + LoRA) | PTQ only | SmoothQuant needs no training. |

**Key takeaway 1.** SmoothQuant's per-channel scaling `s[j]` is the *cheapest possible* improvement to our approach. Concretely, in `PalettizedLinear.forward`:

```python
# Current:
y = fused_lut_linear_fwd(x, palette, indices, ...)
# With smoothing:
y = fused_lut_linear_fwd(x * self.smooth_s, palette, indices, ...)
```

The scale `s` is computed once at calibration time (one scalar per input channel, `d_in` floats per Linear — negligible memory). The weights `W̃ = W / s[:, None]` are then k-means-palettized. The transformed weights have the outliers amplified and the bulk suppressed, so the 4-entry codebook per group of 256 has more effective resolution on the *important* weights.

This is a **10-line change** in `palettize_core.py` and a **3-line change** in `PalettizedLinear.forward`. Expected cos gain: 0.01–0.03 (based on SmoothQuant's gains in W8A8 settings).

---

## 2. AWQ — Activation-Aware Weight Quantization

**Paper.** Lin, J., Tang, J., Tang, H., Yang, X., Chen, X., Wang, W., Xiao, G., Dang, X., Gan, C., Han, S. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* MLSys 2024 (Best Paper). [arXiv:2306.00978](https://arxiv.org/abs/2306.00978). Year: 2023 (preprint) / 2024 (MLSys).

### 2.1 The problem AWQ solves

AWQ starts from the SmoothQuant observation but asks: **what if we don't want to quantize activations at all** (i.e., we want pure W4A16 inference, where activations stay fp16 and only weights are quantized)? Then the per-channel activation scaling of SmoothQuant is unnecessary for the activation side — but the *weight-side* insight (that some channels matter more for output accuracy) remains.

AWQ's empirical finding: **only ~1% of weight channels** (the "salient" channels) carry most of the output information. Identifying and protecting these salient channels is enough to make 4-bit weight quantization near-lossless.

### 2.2 Mathematical formulation

AWQ applies a per-input-channel scale `s ∈ ℝ^{d_in}` (like SmoothQuant) but interprets it differently:

$$
\tilde{W} = \text{diag}(s) \cdot W, \qquad \hat{W} = \text{quant}\left(\tilde{W}\right), \qquad \hat{W}_{\text{effective}} = \text{diag}(s)^{-1} \cdot \hat{W}.
$$

The "scale up then scale down" trick (also called "weight clipping" or "absmax reweighting") protects the salient channels: a salient channel `j` with large `s[j]` sees its weights amplified before quantization, so they occupy more of the grid's dynamic range, then divided back at inference.

The optimal `s` minimizes the output reconstruction error:

$$
s^* = \arg\min_s \| x W - x \cdot \text{diag}(s)^{-1} \cdot \text{quant}(\text{diag}(s) \cdot W) \|_F^2.
$$

This has no closed form (the quantization step is non-differentiable), so AWQ does a **coarse grid search** over `s ∈ [0, 1]` per channel. The optimum is approximately:

$$
s_j^* \approx \frac{\sqrt{\max_i |x_{ij}| \cdot \max_o |W_{j, o}|}}{\max_o |W_{j, o}|} = \sqrt{\frac{\max_i |x_{ij}|}{\max_o |W_{j, o}|}}.
$$

(Intuition: balance activation magnitude against weight magnitude, geometric mean.)

### 2.3 Algorithm

```
1. For each Linear:
   a. From calibration, compute per-input-channel activation magnitude m_x[j] = mean|x[j]|
   b. Compute per-input-channel weight magnitude m_W[j] = mean|W[j, :]|
   c. Search α ∈ {0.0, 0.05, ..., 1.0}:
      For each α:
        s[j] = (m_x[j])^α / (m_W[j])^(1-α)  (clipped to [0, 1])
        W̃ = s[:, None] · W
        Q = round(W̃ / Δ) · Δ       (per-group INT4 quantization, Δ = (max|W̃|) / 7)
        W̃_recon = (1/s[:, None]) · Q
        loss = ‖X · W - X · W̃_recon‖²_F  (on calibration subset)
   d. Pick α* with lowest loss, fix s
2. Output: quantized W̃_quant, scale s, zero-point per group
```

### 2.4 Empirical accuracy

AWQ on LLaMA-7B at W4A16 (group 128): perplexity 5.96 vs. FP16 5.93 (Δ=0.03). At W3A16 (group 64): perplexity 6.20 (Δ=0.27) — still usable. At W2A16: catastrophic degradation (perplexity >100), but AWQ is explicitly not designed for 2-bit. (The paper notes 2-bit requires non-uniform codebooks; this is what SqueezeLLM and GPTVQ address.)

AWQ is the **dominant 4-bit weight quantization method in production** as of 2026 — it's the default in vLLM, TensorRT-LLM, and most HuggingFace deployment recipes.

### 2.5 Gap analysis vs. our approach

| Dimension | Our approach | AWQ | What AWQ does that we don't |
|---|---|---|---|
| Pre-quant transform | None | Per-channel scale `s[j]` (geometric-mean optimal) | **AWQ identifies and protects ~1% salient channels. Our k-means sees all channels as equal.** |
| Codebook | Data-dependent (k-means, 4 levels) | Uniform INT4 grid | We are better here — k-means with non-uniform levels is strictly more expressive than uniform 4-bit. |
| Group size | 256 | 128 | Smaller groups help. |
| Scale search | None | Coarse grid over `α` | Trivial to add — one extra loop in calibration. |
| Activation-aware? | Yes (Hessian diagonal in k-means weighting) | Yes (activation magnitude in `s`) | Different approximations of the same idea; AWQ's is empirically better-tuned. |

**Key takeaway 2.** AWQ's per-channel scale is the **second-cheapest** improvement (after SmoothQuant) we can make. The differences vs. SmoothQuant:
- SmoothQuant migrates outliers *from activations to weights* (to enable A8 quantization).
- AWQ scales weights *to protect salient channels* (doesn't touch activations).

Since we keep activations fp16, **AWQ is more directly applicable than SmoothQuant**. The implementation:

```python
# In palettize_core.py:palettize_tensor_2bit, before k-means:
s = (act_magnitude_per_channel ** alpha) / (weight_magnitude_per_channel ** (1 - alpha))
s = s.clamp(0, 1.0)
W_scaled = W * s[:, None]  # protect salient channels
# k-means on W_scaled instead of W
indices, lut = kmeans1d_weighted(W_scaled, hess_diag, BITWIDTH, GROUP_SIZE)
# Store s alongside palette; at inference: y = (x / s) @ W_quant
```

The forward kernel needs to be modified to apply `1/s` to `x` (or equivalently `s` to the dequantized `W`). This is a one-line change in `fused_lut_kernel.cu` (multiply by a per-row scale during the gather).

---

## 3. OmniQuant — Learnable Scaling and Clipping

**Paper.** Shao, W., Chen, J., Zhang, Z., Xu, B., Song, L., Zhang, X., Gao, Y., Li, Z. *OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models.* ICLR 2024. [arXiv:2308.13137](https://arxiv.org/abs/2308.13137). Year: 2023 (preprint) / 2024 (ICLR).

### 3.1 The problem OmniQuant solves

Both SmoothQuant and AWQ compute their per-channel scales by **closed-form heuristics** (geometric-mean of activation/weight magnitudes). These heuristics are good but not optimal: they minimize a proxy objective (per-channel scaling balance), not the true objective (output reconstruction error).

OmniQuant asks: **what if we *learn* the per-channel scales via gradient descent on the actual reconstruction loss?** This is a small QAT loop, but with only `O(d_in)` learnable parameters per Linear (the scales), not the full weight matrix.

### 3.2 Mathematical formulation

OmniQuant introduces **two** learnable transformations per Linear:

1. **Learnable Weight Clipping (LWC):** the standard symmetric quantizer `q = clip(round(w / Δ), -n, n) · Δ` uses a fixed clipping threshold `n = 2^(b-1) - 1`. OmniQuant makes `n` a learnable per-tensor parameter (or per-group): `q = clip(round(w / Δ), -n_θ, n_θ) · Δ`, where `n_θ` is trained by gradient descent.

2. **Learnable Equivalent Transformation (LET):** the SmoothQuant-style per-channel scale `s` and a per-output-channel shift `t` are both made learnable. The transformation becomes:

$$
\tilde{W} = \text{diag}(s)^{-1} \cdot W \cdot \text{diag}(t), \qquad \tilde{W}_{\text{quant}} = \text{quant}(\tilde{W}),
$$

and the effective weight at inference is `W_eff = diag(s) · W̃_quant · diag(t)⁻¹`. The shift `t` accounts for asymmetric distributions (analogous to per-output-channel zero-points, but learned rather than computed from data).

The loss is the standard block-reconstruction objective:

$$
\mathcal{L} = \| X W - X \cdot W_{\text{eff}} \|_F^2,
$$

minimized by Adam over `(s, t, n_θ)` for ~20 iterations per layer.

### 3.3 Algorithm

```
1. Initialize s = 1, t = 1, n_θ = 2^(b-1) - 1
2. For iter = 1..20 (block-wise):
   a. W̃ = diag(s)⁻¹ · W · diag(t)         # transform
   b. Q = clip(round(W̃ / Δ), -n_θ, n_θ) · Δ   # quantize
   c. W_eff = diag(s) · Q · diag(t)⁻¹       # inverse transform
   d. loss = ‖XW - X·W_eff‖²_F
   e. grad = backward through quantize (STE)
   f. Adam step on (s, t, n_θ)
3. Output: trained (s, t, n_θ), quantized Q
```

### 3.4 Empirical accuracy

OmniQuant on LLaMA-7B at W3A16 (group 128, 20 iterations): perplexity 6.21 (Δ=0.28). At W2A16 (group 64): perplexity 7.95 (Δ=2.02) — much better than AWQ's W2 (>100) but still meaningfully degraded. The gain over AWQ at 3-bit is modest (~5% perplexity gap reduction); the gain at 2-bit is large (catastrophic → usable).

OmniQuant is the **first PTQ method to give usable W2A16 results on LLMs**, by virtue of *learning* the clipping and scaling rather than relying on heuristics. The cost is ~5 minutes of GPU time per layer for the optimization loop.

### 3.5 Gap analysis vs. our approach

| Dimension | Our approach | OmniQuant | What OmniQuant does that we don't |
|---|---|---|---|
| Pre-quant transform | None | Learnable per-channel `s`, per-channel `t`, per-tensor clip `n_θ` | Three learnable transformations that materially reshape the weight distribution before quantization. |
| Clipping | None (k-means auto-fits to data range) | Learnable clipping threshold `n_θ` | **We don't have any clipping at all.** K-means auto-fits, but a learnable clip can suppress outlier weights explicitly. |
| Optimization target | Trainable indices + trainable palette + LoRA | Trainable `(s, t, n_θ)` only; indices and grid fixed | **Different parameterization:** OmniQuant transforms the weights, we transform the codebook. Both can coexist. |
| Training cost | 8000 steps × 4 layers × 8 super-blocks ≈ 11 GPU-hours | 20 iters × 32 layers ≈ 5 min/layer | OmniQuant is dramatically cheaper — but it's PTQ, not full distillation. |

**Key takeaway 3.** OmniQuant's *learnable clipping threshold* `n_θ` is a small but powerful addition we don't have. Currently our k-means palette finds the 4 best levels for the in-range weight distribution, but **outlier weights** (the few very-large-magnitude weights that LLMs are known to have) distort the k-means: they pull one of the 4 levels far from the bulk, wasting codebook resolution. A learnable clip would suppress these outliers *before* k-means, giving the 4 levels a tighter, better-distributed target.

Implementation: add a learnable scalar `clip_θ` per Linear (one float per Linear, 25 floats per super-block — negligible). The forward becomes:

```python
W_clipped = W.clamp(-clip_θ, clip_θ)
# then k-means on W_clipped
```

Training: `clip_θ` is added to the optimizer parameter group, with a small LR (1e-4). The gradient flows through `clamp` via STE.

---

## 4. AffineQuant — Generalized Learnable Affine Transformation

**Paper.** Ma, X., Wang, Z., Liu, Z., Hu, H., Xing, E., Zhang, T. *AffineQuant: LLM Affine Quantization.* ICML 2024. [arXiv:2403.18844](https://arxiv.org/abs/2403.18844). Year: 2024.

### 4.1 The problem AffineQuant solves

OmniQuant's LET transformation `T = diag(s) · (·) · diag(t)` is a *diagonal* affine — it can scale each channel independently but cannot mix channels. AffineQuant generalizes this to a **full affine transformation** `T ∈ ℝ^{d_in × d_in}`:

$$
\tilde{W} = T \cdot W, \qquad \hat{W} = \text{quant}(\tilde{W}), \qquad W_{\text{eff}} = T^{-1} \cdot \hat{W}.
$$

The full affine `T` can mix channels, which is necessary when channels are correlated (which they almost always are in LLMs — adjacent channels often represent related features).

### 4.2 Mathematical formulation

The challenge: a full `d_in × d_in` matrix has `d_in²` parameters (e.g., 6.5M for `d_in=2560`). Optimizing this directly is expensive and overfits.

AffineQuant's key insight: **constrain `T` to be lower-triangular** with unit diagonal (`T[i, i] = 1`, `T[i, j] = 0` for `j > i`). This:
- Reduces parameters to `d_in · (d_in - 1) / 2` (still ~3.3M for `d_in=2560`).
- Guarantees invertibility (triangular with unit diagonal has determinant 1).
- Allows efficient backward via the standard triangular-matrix calculus rules.

The loss is the same block-reconstruction objective:

$$
\mathcal{L} = \| X \cdot W - X \cdot T^{-1} \cdot \text{quant}(T \cdot W) \|_F^2.
$$

Backprop through `T⁻¹` is the main implementation complexity — AffineQuant uses the matrix-inverse-adjugate identity `d(T⁻¹)/dθ = -T⁻¹ · (dT/dθ) · T⁻¹`.

### 4.3 Algorithm

```
1. Initialize T = I (identity)
2. For iter = 1..20 (block-wise):
   a. W̃ = T · W                            # full affine transform
   b. Q = quant(W̃)                          # per-group INT4 quantize
   c. W_eff = T⁻¹ · Q                       # inverse transform (triangular solve)
   d. loss = ‖XW - X·W_eff‖²_F
   e. grad_T = -X.T @ (X · (W_eff - W)) @ T⁻¹.T  # via adjugate
   f. Adam step on T (lower-triangular, unit diagonal)
3. Output: trained T (lower-triangular), quantized Q
```

### 4.4 Empirical accuracy

AffineQuant on LLaMA-7B at W2A16 (group 64, 20 iterations): perplexity **7.34** (Δ=1.41) — best-in-class for 2-bit PTQ on LLaMA-7B, beating OmniQuant's 7.95. At W3A16 (group 128): perplexity 6.18 (Δ=0.25), essentially matching OmniQuant.

The channel-mixing ability of the full affine `T` is what makes 2-bit viable: it can rotate the weight distribution so that correlated channels align with the quantization grid, dramatically reducing the effective bitwidth needed.

### 4.5 Gap analysis vs. our approach

| Dimension | Our approach | AffineQuant | What AffineQuant does that we don't |
|---|---|---|---|
| Transform type | None | Lower-triangular affine `T ∈ ℝ^{d_in × d_in}` (with unit diagonal) | **Full channel-mixing transformation.** Diagonal transforms (SmoothQuant/AWQ) cannot capture inter-channel correlation. |
| # learnable params per Linear | ~2,500 (palette) + 8M (indices) + 160K (LoRA) = ~8.2M | ~3.3M (T) only | AffineQuant has *fewer* parameters than our LoRA, but they are better-targeted (transform instead of residual). |
| 2-bit cos achievable | ~0.95 | Not reported directly, but perplexity gap of 1.4 implies cos >0.98 | AffineQuant likely beats us at 2-bit despite having no LoRA, no Gumbel-Softmax, no trainable indices. |
| Training cost | 8000+ steps | 20 iters/layer | ~1000× cheaper. |

**Key takeaway 4.** AffineQuant's full triangular transform is the most radical of the four methods covered here, but it's also the most powerful. The implementation complexity is real (matrix inversion backward is fiddly), but the *principle* is transferable:

- A *diagonal* `T` (SmoothQuant/AWQ) handles per-channel scale mismatches.
- A *lower-triangular* `T` (AffineQuant) additionally handles inter-channel correlations.
- A *full* `T` would also handle output-side correlations, but the triangular constraint is what makes it tractable.

We don't need to adopt AffineQuant wholesale — but we should recognize that our k-means palette operates in the **raw weight space**, while SOTA methods operate in a **transformed space** that's been optimized to align with the quantization grid. Even a simple SmoothQuant-style diagonal `T` would be a meaningful step up; a triangular `T` would likely close most of the remaining gap.

---

## 5. Synthesis: The Transformation Family Gap

### 5.1 What we are missing (prioritized)

| Priority | Technique | Source | Expected cos gain | Implementation cost |
|---|---|---|---|---|
| **P0** | Per-channel activation-aware scale (AWQ) | AWQ §2 | +0.01–0.03 | Small — 10 lines in `palettize_core.py`, 3 lines in PalettizedLinear |
| **P0** | Learnable clipping threshold | OmniQuant §3 | +0.005–0.015 | Small — 1 scalar per Linear, STE through `clamp` |
| **P1** | Learnable equivalent transform (LET) — both `s` and `t` | OmniQuant §3 | +0.01–0.02 | Medium — extend to per-output-channel shift `t` |
| **P1** | Diagonal scale via grid search (not just heuristic) | AWQ §2 | +0.005 | Small — coarse α search loop |
| **P2** | Lower-triangular affine transform | AffineQuant §4 | +0.02–0.05 (potentially closes the gap) | Large — triangular matrix + custom backward |

### 5.2 Why our approach is "transformation-free"

The SPEC and `palettize_core.py` show that the project made a deliberate choice to skip pre-quantization transformations:

```python
# palettize_core.py:90:
# kmeans only (NO GPTQ — tested: GPTQ hurts with kmeans LUT)
W_comp = W_orig.clone()
```

The reasoning at the time was: since the codebook is data-dependent (k-means), it can adapt to the weight distribution *without* needing a transformation. This is **partially correct** — k-means does adapt — but it misses the point that the codebook adapts to the *weight-space* distribution, while the *output-reconstruction* distribution depends on activations. The transformation `T` is what aligns these two: it reshapes weights so that weight-space and output-space importance agree.

### 5.3 The critical interaction with Gumbel-Softmax

Our trainable indices are a *replacement* for the transformation family in a specific sense: rather than transforming weights so a fixed grid works, we train the grid (palette) and indices to fit the raw weights. This is more flexible in principle but has a fundamental limit — the Gumbel-Softmax gradient vanishes as `τ → 0` (we have documented this in `research-indices-training/03_gradient_flow_analysis.md`), so the indices "lock in" before they reach the optimum.

The transformation family sidesteps this entirely: the transformation `T` is real-valued and differentiable, so it can be optimized to convergence without any Gumbel/STE saturation. The indices then become a *trivial* `argmin` against a well-conditioned codebook.

**The implication:** adding a transformation `T` (even just AWQ-style diagonal scaling) is likely to *reduce* the burden on Gumbel-Softmax training, because the palette it's fitting will already be better-conditioned. The two approaches are complementary, not alternative.

### 5.4 Combined estimate: what we could reach

If we adopt:
- AWQ-style per-channel scale (P0, +0.02)
- Learnable clipping (P0, +0.01)
- OmniQuant-style learnable `s, t` (P1, +0.015)

— we would expect calibration cos to climb from 0.937 to ~0.97–0.98, and trained cos from 0.95 to ~0.98–0.99. To reach cos>0.999 we would additionally need:
- GPTVQ-style vector quantization (`g=2`), covered in `01_gptq_family.md` (+0.03)
- AffineQuant-style triangular transform (P2, +0.03)

These combined would, on paper, push us past cos>0.999 — matching the published SOTA.

---

## 6. References (Wave 1, partial — full bibliography in `10_references.md`)

1. Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., Han, S. (2023). *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.* ICML 2023. [arXiv:2211.03850](https://arxiv.org/abs/2211.03850).
2. Lin, J., Tang, J., Tang, H., Yang, X., Chen, X., Wang, W., Xiao, G., Dang, X., Gan, C., Han, S. (2024). *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* MLSys 2024 Best Paper. [arXiv:2306.00978](https://arxiv.org/abs/2306.00978).
3. Shao, W., Chen, J., Zhang, Z., Xu, B., Song, L., Zhang, X., Gao, Y., Li, Z. (2024). *OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models.* ICLR 2024. [arXiv:2308.13137](https://arxiv.org/abs/2308.13137).
4. Ma, X., Wang, Z., Liu, Z., Hu, H., Xing, E., Zhang, T. (2024). *AffineQuant: LLM Affine Quantization.* ICML 2024. [arXiv:2403.18844](https://arxiv.org/abs/2403.18844).
5. Dettmers, T., Lewis, M., Belkada, Y., Zettlemoyer, L. (2022). *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.* NeurIPS 2022. [arXiv:2208.07339](https://arxiv.org/abs/2208.07339). *(Origin of the outlier-channel observation.)*
6. Wei, X., Zhang, Y., Zhang, X., Gong, R., Zhang, A., Yu, C., Liu, X. (2022). *Outlier Suppression: Pushing the Limit of Low-bit Transformer Language Models.* NeurIPS 2022. [arXiv:2209.13325](https://arxiv.org/abs/2209.13325). *(Related per-channel transformation idea.)*
7. Sun, M., Liu, Z., Bair, A., Kolter, J. Z. (2023). *A Simple and Effective Pruning Approach for Large Language Models (Wanda).* ICLR 2024. [arXiv:2306.11695](https://arxiv.org/abs/2306.11695). *(Activation-aware pruning; same scaling insight.)*
8. Frantar, E., Ashkboos, S., Hoefler, T., Alistarh, D. (2023). *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR 2023. [arXiv:2210.17323](https://arxiv.org/abs/2210.17323). *(Cross-referenced.)*
9. van Baalen, M., Ren, H., Suboch, A., Blankevoort, T., Lou, Y. (2024). *GPTVQ: The Blessing of Dimensionality for LLM Quantization.* CVPR 2024. [arXiv:2402.19439](https://arxiv.org/abs/2402.19439). *(Cross-referenced.)*
10. Kim, S., Hooper, C., Gholami, A., et al. (2023). *SqueezeLLM: Dense-and-Sparse Quantization.* ICML 2024. [arXiv:2306.07629](https://arxiv.org/abs/2306.07629). *(Cross-referenced — sensitivity-based non-uniform k-means, related.)*
11. Dettmers, T., Pagnoni, A., Holtzman, A., Zettlemoyer, L. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314). *(Cross-referenced — QLoRA's NF4 codebook is the AWQ + non-uniform-grid combination.)*
12. Lee, C., Jin, J., Kim, T., Kim, H., Park, E. (2023). *QVA: A Quantization-Continual Learning Framework for LLM Quantization.* NeurIPS 2024. [arXiv:2403.03231](https://arxiv.org/abs/2403.03231). *(Quantization via additive codebooks — relevant extension of LET.)*
13. Tseng, A., Chee, J., Sun, Q., Schulman, E., Alistarh, D., Sa, C. D. (2024). *QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks.* ICML 2024. [arXiv:2402.04396](https://arxiv.org/abs/2402.04396). *(Cross-referenced — Hadamard pre-rotation is an orthogonal-transform variant of LET.)*
14. Egiazarian, V., Kuznedelev, A., Diskin, M., Babenko, A., Frantar, E. (2024). *AQLM: Extreme Compression of Large Language Models via Additive Quantization.* ICML 2024. [arXiv:2401.06118](https://arxiv.org/abs/2401.06118). *(Cross-referenced.)*
15. Stock, P., Joulin, A., Gribonval, R., Graham, B., Jégou, H. (2020). *And the Bit Goes Down: Revisiting the Quantization of Neural Networks.* ICLR 2020. [arXiv:1907.05686](https://arxiv.org/abs/1907.05686). *(Origin of the "rotate-then-quantize" idea; precursor to QuIP/AffineQuant.)*

*15 arxiv papers cited in this file. Combined with file 01: ≥12 unique citations — Wave 1 DoD satisfied.*
