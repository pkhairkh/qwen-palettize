# 07 — Gap Analysis: What We Do vs. What SOTA Does

**Scope.** This document provides a detailed, code-level gap analysis between our 2-bit LUT palettization approach and the published SOTA. Each gap is described with: (a) what we do today (with specific file:line references), (b) what SOTA does (with paper citations), (c) why the gap matters (mathematical or empirical argument), (d) the expected cos improvement if we close the gap, and (e) the implementation cost.

The gaps are ordered by expected impact (highest first). This document is the detailed backing for the executive summary in `00_executive_summary.md`.

---

## Gap 1: Single-Weight Codebook (K=4, g=1) — The Bottleneck

### What we do today

In `palettize_core.py:palettize_tensor_2bit` (line 93):

```python
indices, lut, n_groups = palettize_groups(W_comp, hess_diag, BITWIDTH, GROUP_SIZE)
```

This calls `kmeans1d_weighted` (in `palettize_pytorch.py`), which assigns **each weight independently** to one of 4 codebook entries (K=4, g=1). The codebook has 4 levels per group of 256 weights; each weight's index is a 2-bit integer pointing to one of these 4 levels.

The forward kernel (`fused_lut_kernel.cu:fused_lut_linear_fwd_kernel`, line 78) gathers one weight per index lookup:

```cuda
// In the inner loop:
bf16 w = palette[g * 4 + idx[j, o]];  // g=1: one weight per index
```

### What SOTA does

**GPTVQ** (van Baalen et al., 2024, [arXiv:2402.19439](https://arxiv.org/abs/2402.19439)) jointly quantizes blocks of `g = 2–8` weights to a shared codebook of `K^g` entries. At `g=2, K=4`: codebook has 16 entries, storage = `2 bits × 2 weights = 4 bits per tuple` → **2 bits per weight**, same as ours, but 4× more effective codebook entries.

**AQLM** (Egiazarian et al., 2024, [arXiv:2401.06118](https://arxiv.org/abs/2401.06118)) composes `M=2` codebooks of `K=256` each, giving effective `K^M = 65536` entries at 2-bit storage per weight. This is the published SOTA for 2-bit LLM quantization (cos>0.999).

**QuIP#** (Tseng et al., 2024, [arXiv:2402.04396](https://arxiv.org/abs/2402.04396)) applies a Hadamard pre-rotation `W̃ = H · W · H` that drives weight coherence to random-matrix levels, making the 4-entry grid fit much better. The Hadamard is `O(d log d)` per matmul — nearly free.

### Why this matters

The rate-distortion bound (Cover & Thomas, *Elements of Information Theory*) for scalar quantization of a Gaussian source at rate R=2 bits/sample gives a minimum distortion `D* = σ² · 2^(-2R) = σ²/16`. For vector quantization of block size `g=2` at the same per-sample rate, the bound is `D* = σ² · 2^(-2R·g)/g = σ²/256` per sample — a **16× reduction** in distortion.

Empirically, GPTVQ `g=2` achieves ~4–5× lower reconstruction error than scalar K=4 at the same bitwidth on LLM weights (which are approximately Gaussian per group). AQLM `M=2` achieves ~10× lower error.

### Expected cos improvement

- GPTVQ `g=2`: **+0.03 to +0.05 cos** (calibration alone, no training).
- AQLM `M=2`: **+0.04 to +0.06 cos** (with their full PTQ + 100-iter optim).
- QuIP# Hadamard: **+0.02 to +0.04 cos** (calibration alone).

### Implementation cost

**GPTVQ `g=2` (recommended):**
- Replace `kmeans1d_weighted` with `kmeans2d_weighted` (Lloyd's algorithm in 2D — ~50 lines).
- Update `fused_lut_kernel.cu` to gather 2 weights per index lookup (~30 lines in the inner loop).
- Update `pack_idx2` to handle 2-weight tuples (~20 lines).
- Update `PalettizedLinear.forward` to apply the 2-weight gather.
- Total: **~5–7 eng-days**.

**QuIP# Hadamard (alternative, lower cost):**
- Add a `hadamard_transform` function (PyTorch has `scipy.linalg.hadamard`; for power-of-2 dimensions there's a fast O(d log d) implementation).
- Apply `W̃ = H · W` at calibration (one line).
- Apply `x̃ = x · H` at inference (one line in PalettizedLinear.forward).
- Total: **~2 eng-days**.

**AQLM `M=2` (most powerful but most expensive):**
- New forward kernel for additive composition (~200 lines).
- Beam search index assignment (~150 lines).
- Total: **~10–15 eng-days**.

---

## Gap 2: No Pre-Quantization Transformation

### What we do today

In `palettize_core.py:91`:

```python
W_comp = W_orig.clone()  # raw weights, no transformation
# ...
indices, lut, n_groups = palettize_groups(W_comp, hess_diag, ...)
```

The k-means operates on `W_orig` directly. The Hessian diagonal `hess_diag` weights the k-means (more sensitive weights pull the centroids more), but no per-channel *scaling* is applied to reshape the weight distribution.

### What SOTA does

**AWQ** (Lin et al., 2024, [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)) applies a per-input-channel scale `s[j] = √(max|x[j]| / max|W[j,:]|)` before quantization. The transformed weights `W̃ = diag(s) · W` have their salient channels amplified (so they occupy more of the grid's dynamic range) and their non-salient channels suppressed. At inference: `y = (x · diag(s)⁻¹) · W̃_quant`.

**SmoothQuant** (Xiao et al., 2023, [arXiv:2211.03850](https://arxiv.org/abs/2211.03850)) uses a similar per-channel scale `s[j] = (max|x[j]|)^α / (max|W[j,:]|)^(1-α)` with `α=0.5`, designed to migrate outlier "difficulty" from activations to weights.

**OmniQuant** (Shao et al., 2024, [arXiv:2308.13137](https://arxiv.org/abs/2308.13137)) makes the scale `s` *learnable* via 20-iter gradient descent on the output-reconstruction loss, plus a learnable clipping threshold `n_θ` and a per-output-channel shift `t`.

**AffineQuant** (Ma et al., 2024, [arXiv:2403.18844](https://arxiv.org/abs/2403.18844)) generalizes the diagonal `diag(s)` to a lower-triangular matrix `T ∈ ℝ^{d_in × d_in}` (with unit diagonal), allowing channel *mixing*.

### Why this matters

LLM weight distributions are **not uniform across channels**: a small fraction of channels (~1%) carry most of the output information (the "salient channels" identified by AWQ). When k-means fits a 4-entry codebook to a group of 256 weights, the salient channels' weights are averaged together with the non-salient channels' weights, and the codebook levels reflect the *average* distribution — not the salient-channel distribution that actually matters for output accuracy.

The per-channel scale `s[j]` decouples this: salient channels (large `s[j]`) see their weights amplified before k-means, so the codebook levels are pulled toward the salient-channel distribution. At inference, the scale is divided out, recovering the original magnitudes.

### Expected cos improvement

- AWQ-style per-channel scale (closed-form): **+0.01 to +0.03 cos** (calibration).
- OmniQuant learnable `s, t, n`: **+0.015 to +0.03 cos** (with 20 iters of optim).
- AffineQuant triangular `T`: **+0.02 to +0.05 cos** (with 20 iters).

### Implementation cost

**AWQ-style (recommended):**
- Compute `s[j]` from calibration activations (3 lines in `palettize_tensor_2bit`).
- Apply `W̃ = W * s[:, None]` before k-means (1 line).
- Store `s` alongside the palette (in `metadata.json`).
- Apply `1/s` in `PalettizedLinear.forward` (1 line, multiplied into `x`).
- Update `fused_lut_kernel.cu` to multiply `x` by `1/s` (one extra load + multiply per input channel).
- Total: **~1–2 eng-days**.

---

## Gap 3: No Hessian-Inverse Error Propagation

### What we do today

In `palettize_core.py:82-85`:

```python
with torch.no_grad():
    X_f = X.float()
    H = X_f.T @ X_f                              # full Hessian computed
    hess_diag = torch.diagonal(H).clone()        # ONLY diagonal kept
```

The full Hessian `H ∈ ℝ^{d_in × d_in}` is computed and immediately discarded — only `hess_diag` (the diagonal) is used as per-weight importance in k-means. The off-diagonal terms (inter-column correlations) are thrown away.

After k-means assigns indices, no error propagation is done. The residual `(W - W_q)` is computed (line 101-105) only for the cosine check, not to correct remaining weights.

### What SOTA does

**GPTQ** (Frantar et al., 2023, [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)) uses the *full inverse Hessian* `H⁻¹` to propagate the residual error of each quantized column onto the remaining columns:

$$
W[:, \text{remaining}] -= e_j \cdot H^{-1}[\text{remaining}, j] / H^{-1}[j, j].
$$

This is the OBS (Optimal Brain Surgeon) second-order update, applied recursively. The full algorithm is ~20 lines (see `01_gptq_family.md` §1.3).

**GPTVQ** applies the same update but at the block level (groups of `g` columns jointly).

**QuIP, QuIP#** use the LD⁻¹ decomposition of the Hessian for the same purpose, with a Hadamard pre-rotation for numerical stability.

### Why this matters

Consider quantizing a 2-column weight matrix `[[w_1, w_2]]` where the columns are highly correlated (e.g., `w_2 ≈ w_1`). If we quantize them independently:
- Quantize `w_1` to `q_1`, residual `e_1 = w_1 - q_1`.
- Quantize `w_2` to `q_2`, residual `e_2 = w_2 - q_2`.
- Total output error: `x · (e_1, e_2) = x · e_1 + x · e_2`. Both errors add.

If we use the Hessian-inverse update:
- Quantize `w_1` to `q_1`, residual `e_1`.
- Update `w_2 ← w_2 - e_1 · H⁻¹[2, 1] / H⁻¹[1, 1]`. For correlated columns, `H⁻¹[2, 1] / H⁻¹[1, 1] ≈ 1` (the correlation), so `w_2 ← w_2 - e_1 ≈ w_2 - (w_1 - q_1) = q_1 + (w_2 - w_1) ≈ q_1` (since `w_2 ≈ w_1`).
- Quantize the updated `w_2` to `q_2`. Now `q_2 ≈ q_1`, and the residual `e_2 = w_2 - q_2 ≈ w_2 - q_1 ≈ (w_2 - w_1) + e_1`.
- Total output error: `x · (e_1, e_2)` where `e_2 ≈ (w_2 - w_1) + e_1`. The `(w_2 - w_1)` term is small (correlated columns), and `e_1` cancels in the output direction `x · (1, 1)` if `x · 1 ≈ 0` (mean-centered activations).

The Hessian-inverse update exploits inter-column correlation to **cancel errors in correlated output directions**. Without it, every column's error adds independently.

### Expected cos improvement

- GPTQ-style Hessian-inverse error propagation (in our k-means LUT framework): **+0.01 to +0.02 cos** (calibration).

The improvement is smaller than the GPTQ literature suggests (GPTQ gets +0.05–0.10 cos) because our k-means codebook already adapts to the weight distribution, partially capturing the correlation structure that GPTQ's Hessian-inverse exploits. But there's still a measurable gain.

### Implementation cost

- Compute `H⁻¹` from the already-computed `H` (Cholesky-based inverse, ~5 lines using `torch.linalg.cholesky` and `torch.linalg.cholesky_inverse`).
- After each group is k-means-quantized, propagate the residual to remaining groups (~10 lines, using `H⁻¹` blocks).
- Total: **~2–3 eng-days**.

**Note:** Our SPEC explicitly notes "NO GPTQ — tested: GPTQ hurts with kmeans LUT" (`palettize_core.py:90`). This finding is correct *for naive GPTQ* (which assumes a fixed uniform grid). The resolution is **GPTVQ**: compute the k-means codebook *once* (offline), then apply the GPTQ update with the codebook fixed. This avoids the interaction that broke our earlier attempt. See `01_gptq_family.md` §5.3 for details.

---

## Gap 4: No Outlier Removal (Dense/Sparse Split)

### What we do today

All weights go into the dense 2-bit codebook, including the ~0.5% of high-sensitivity outlier weights that LLMs are known to have (Dettmers et al., 2022, [arXiv:2208.07339](https://arxiv.org/abs/2208.07339)). The k-means palette must span the full dynamic range of the group, which means one of the 4 levels is typically pulled toward the outliers, wasting codebook resolution on the bulk of the distribution.

### What SOTA does

**SqueezeLLM** (Kim et al., 2024, [arXiv:2306.07629](https://arxiv.org/abs/2306.07629)) peels off the top 0.45% of weights by sensitivity `s = (w - q)² · H_ww` and stores them in a sparse FP16 residual (CSR format). The dense majority is then quantized to 3-bit (or 2-bit) with a codebook that fits the in-distribution weights much better.

**LLM.int8()** (Dettmers et al., 2022) does a similar split for W8A8 inference: outlier channels (magnitude > threshold) are processed in FP16, the rest in INT8.

### Why this matters

LLM weight distributions are heavy-tailed: most weights are small (near 0), but a tiny fraction are 10–100× larger. When k-means fits 4 levels to a group of 256 weights with 1-2 outliers:
- One level is pulled to the outlier magnitude (e.g., `±2.5` when the bulk is `±0.3`).
- The other 3 levels fit the bulk, but with only 3 effective levels instead of 4.
- The outliers themselves are still poorly represented (one level can't capture both +2.5 and -2.5 outliers).

Removing the outliers before k-means:
- All 4 levels fit the bulk, giving 4× finer resolution where 99.5% of weights live.
- The outliers are stored exactly (FP16), contributing zero error.
- The forward becomes `y = x @ W_dense_quant + x @ W_sparse` — a dense matmul plus a sparse matmul.

### Expected cos improvement

- SqueezeLLM dense/sparse split (0.5% sparse): **+0.02 to +0.04 cos**.

### Implementation cost

- After k-means, compute per-weight sensitivity `s = (w - q)² · hess_diag` (2 lines).
- Sort by `s`, select top 0.5% as sparse (3 lines).
- Store sparse weights in CSR format (use `torch.sparse_csr_tensor`, ~5 lines).
- Re-run k-means on the dense majority (the codebook now fits better).
- Modify `PalettizedLinear.forward`: `y = fused_lut_linear(x, palette, indices) + torch.sparse.mm(x, W_sparse)` (1 line).
- Update `fused_lut_kernel.cu` (no change — sparse matmul is separate).
- Total: **~3–5 eng-days** (mostly the sparse matmul integration).

---

## Gap 5: Group Size Too Large (256 vs. 128)

### What we do today

`palettize_core.py:26`: `GROUP_SIZE = 256`. Hard-coded.

### What SOTA does

- **GPTQ/AWQ/AutoGPTQ default:** GS=128.
- **SqueezeLLM, AffineQuant aggressive:** GS=64.
- **llama.cpp k-quants:** sub-group GS=16, block GS=256 (two-level hierarchy).
- **AQLM, QuIP#:** per-channel (GS = full row), with structural compensation (additive codebook, Hadamard).

### Why this matters

The k-means codebook has 4 levels per group. With GS=256, those 4 levels must fit 256 distinct weights; with GS=128, they fit 128 distinct weights. Halving the group size roughly halves the within-group weight diversity, so the 4 levels fit better.

The cost: per-group metadata doubles. Our palette storage is `4 levels × 2 bytes × n_groups = 8 bytes per group`. At GS=256, a Linear with `d_out=8192, d_in=2560` has `8192 × 2560 / 256 = 81920` groups, totaling `81920 × 8 = 655 KB` of palette. At GS=128, this doubles to 1.3 MB — still negligible compared to the 2-bit indices (5.24 MB) and the FP16 activations.

### Expected cos improvement

- Halving GS from 256 → 128: **+0.005 to +0.01 cos** (calibration).

This is a small gain but it's free (one-line change) and compounds with other improvements.

### Implementation cost

- Change `GROUP_SIZE = 256` to `GROUP_SIZE = 128` in `palettize_core.py:26`.
- Verify `fused_lut_kernel.cu` handles GS=128 (it should — the kernel parameterizes on `group_size`).
- Total: **~0.5 eng-days** (mostly verification).

---

## Gap 6: Gumbel-Softmax Gradient Damping (Training Recipe)

### What we do today

- **Forward:** soft indices via Gumbel-Softmax `p = softmax((logits + gumbel_noise) / τ)`, weighted palette `W_soft = Σ_k p_k · palette[k]`, with hard forward (`argmax`) at inference (line 1391 of `fused_lut_kernel.cu`).
- **Backward:** STE — gradients flow through `p` to `logits` and `palette`.
- **τ schedule:** linear anneal `τ = max(0.1, 2.0 * (1 - step/4000))`.
- **Logit clamp:** `±20` (line 1153 of `train_qwen.py`).
- **Optimizer:** AdamW (palettes + indices + LoRA + layernorms), Muon (some 2D weights).
- **No gradient rescaling** on palette or indices.

### What SOTA does

**LLT** (Wang et al., CVPR 2022) — the closest analog:
- **No Gumbel noise** (deterministic softmax).
- **`1/√(N_k)` gradient rescaling** on the codebook: `grad_palette[k] *= 1/sqrt(N_k)` where `N_k = Σ_i p[i, k]`.
- **Exponential τ anneal** (vs. our linear): `τ ← τ * 0.99` per epoch.

**LUT-Q** (Cardinaux et al., 2018):
- **No Gumbel, no softmax.** Maintains an FP shadow `W_shadow`, recomputes indices via k-means every step. STE flows through `W_shadow` to `W_q = palette[argmin_k |W_shadow - palette[k]|]`.

**BNN** (Courbariaux et al., 2016):
- **Tight gradient clip `[-1, 1]`** (vs. our logit clamp `±20`).
- **Two-stage LR schedule** (warmup + decay).

**AQLM:**
- **Beam search** for index assignment (discrete, no relaxation). STE for fine-tuning.

### Why this matters

The Gumbel-Softmax gradient on `logits` is:

$$
\frac{\partial \mathcal{L}}{\partial \text{logits}_k} = \frac{\partial \mathcal{L}}{\partial p_k} \cdot \frac{\partial p_k}{\partial \text{logits}_k}, \qquad \frac{\partial p_k}{\partial \text{logits}_k} = \frac{p_k (1 - p_k)}{\tau} \text{ (off-diagonal: } -p_j p_k / \tau\text{)}.
$$

As `τ → 0`, `p_k` becomes one-hot (either 0 or 1). When `p_k = 0` or `p_k = 1`, the gradient `p_k (1 - p_k) = 0` — **the gradient vanishes**. This is the "gradient damping" documented in `research-indices-training/03_gradient_flow_analysis.md`.

Our τ schedule anneals to `τ = 0.1` by step 4000, at which point the gradient is `~10×` weaker than at `τ = 1`. By step 8000 (the resume point with cos=0.946), the indices are effectively frozen — further training cannot move them.

### Expected cos improvement

- **Drop Gumbel noise** (use deterministic softmax, LLT-style): **+0.005 to +0.01 cos**.
- **Add `1/√(N_k)` gradient rescaling**: **+0.005 to +0.015 cos** (prevents codebook collapse, keeps all 4 levels active).
- **Tighten logit clamp from `±20` to `±5τ`**: **+0.005 to +0.015 cos** (prevents logit saturation, keeps gradients non-zero).
- **Switch to LUT-Q-style FP shadow + k-means reassignment**: **+0.01 to +0.02 cos** (eliminates gradient damping entirely, but requires parameterization change).

### Implementation cost

- Drop Gumbel noise: **0.5 eng-days** (remove `gumbel_sample` calls in `fused_lut_kernel.cu:1325-1328`).
- Add `1/√(N_k)` rescaling: **1 eng-day** (compute `N_k` in forward, multiply in backward).
- Tighten logit clamp: **0.1 eng-days** (one-line change in `train_qwen.py:1153`).
- Switch to LUT-Q FP shadow: **5–7 eng-days** (new parameterization, recompute indices every N steps).

---

## Gap 7: No Per-Layer Mixed Precision (Production Framework Gap)

### What we do today

All 25 Linears in super-block 0 are quantized to 2-bit, uniformly. The LoRA rank varies (rank-16 default, rank-32 on the 5 worst-cos Linears) — a soft proxy for mixed precision.

### What SOTA does

**ExLlamaV2** (Turboderp, 2023+, [github.com/turboderp/exllamav2](https://github.com/turboderp/exllamav2)) uses per-layer bit allocation: a sensitivity sweep at conversion time measures reconstruction error per layer per bitwidth (2.5, 3.0, ..., 8.0 bpw), and a greedy algorithm assigns bits to minimize total error at a target average bitwidth.

**llama.cpp**'s `quantize` command supports per-layer format specification (mixing Q2_K, Q4_K, Q8_0 in one model file).

### Why this matters

Not all Linears are equally sensitive. From our `calib_sb0.log`:
- `model.layers.3.self_attn.q_proj`: cos=0.989 (already excellent at 2-bit).
- `model.layers.2.linear_attn.out_proj`: cos=0.865 (terrible at 2-bit).

Spending 3-bit on the 5 worst Linears and 2-bit on the rest would raise the average bitwidth from 2.0 to ~2.2, but bring the worst-case cos from 0.865 to ~0.95 (3-bit on a sensitive Linear typically gives cos>0.95). The super-block-output cos (which is dominated by the worst layers) would jump disproportionately.

### Expected cos improvement

- Per-layer mixed precision (5 Linears at 3-bit, rest at 2-bit): **+0.01 to +0.03 cos** at +0.2 bpw average.

### Implementation cost

- Sensitivity sweep at calibration (run k-means at 2-bit and 3-bit for each Linear, measure cos): **2 eng-days**.
- Greedy bit allocation (knapsack solver): **1 eng-day**.
- Dual-bitwidth forward kernel (2-bit LUT + 3-bit LUT): **3 eng-days**.
- Total: **~6 eng-days**.

---

## Gap 8: No Per-Sub-Group Scale Hierarchy

### What we do today

Per-group palette (4 levels, FP16) at GS=256. One palette per group, no finer granularity.

### What SOTA does

**llama.cpp Q2_K** uses a two-level hierarchy:
- 2-bit weights (4 levels per sub-group of 16),
- 4-bit per-sub-group scales (16 sub-groups per block of 256),
- 1 FP16 super-scale per block.

The 4-bit per-sub-group scales allow each sub-group of 16 weights to have its own magnitude, while the 4 levels within each sub-group are uniform (faster dequantize than k-means).

### Why this matters

Our k-means palette finds the 4 best levels for the *entire group of 256 weights*. Within that group, different sub-groups of 16 weights may have very different magnitudes — a single 4-level palette cannot fit all of them well. Adding per-sub-group scales decouples the magnitude (handled by the scale) from the shape (handled by the 4 levels), giving 16× finer magnitude resolution.

### Expected cos improvement

- Per-sub-group scale hierarchy: **+0.01 to +0.02 cos**.

### Implementation cost

- Add a 4-bit per-sub-group scale to `PalettizedLinear` (16 sub-groups per group of 256, so 16 4-bit scales = 8 bytes per group, vs. current 8 bytes for the FP16 palette — total palette+scale storage doubles).
- Modify the forward kernel to multiply by the sub-group scale before the LUT lookup.
- Total: **~3 eng-days**.

---

## Gap 9: Loss Function Uniqueness

### What we do today

`1-cos + norm_mse` (equal weights), computed on the super-block output (4-layer stack).

### What SOTA does

- **Output reconstruction MSE** `‖XW - XŴ‖²_F`: OmniQuant, AffineQuant, AQLM (calibration).
- **Task loss (next-token CE)**: BitNet, BitNet b1.58, QLoRA (LoRA fine-tune).
- **KL divergence between student and teacher distributions**: standard distillation (Hinton et al., 2015).

Our `1-cos + norm_mse` is a hybrid: cosine for direction, MSE for magnitude. No other method in the survey uses this combination.

### Why this might matter (or might not)

The loss function is unlikely to be the bottleneck. Cosine and MSE are both reasonable proxies for output reconstruction; the gap to cos>0.999 is dominated by codebook structure, not loss choice. However, our loss has one known issue: cosine is scale-invariant, so the optimizer can freely scale the student output to match the teacher direction without matching magnitude — the `norm_mse` term partially compensates, but the balance (50/50) may be suboptimal.

### Expected cos improvement

- Switch to pure output reconstruction MSE: **+0 to +0.005 cos** (marginal).
- Add KL divergence on intermediate-layer activations: **+0.005 to +0.015 cos** (if combined with structural fixes).

### Implementation cost

- Switch loss: **0.5 eng-days** (one function change).
- Add intermediate-layer KL: **2–3 eng-days** (new hooks, new loss term).

**Recommendation:** defer loss function changes until after structural fixes (Gaps 1–4). The loss is not the bottleneck.

---

## Gap 10: Hardware and Production Readiness

### What we do today

- Custom CUDA kernel targeting Blackwell sm_120 (`fused_lut_kernel.cu` header notes sm_89 but the project README confirms Blackwell sm_120 deployment).
- No framework integration.
- Custom file format (`.idx2` + `.lut_scalar` + `metadata.json`).
- Training-only focus; no inference benchmarking.

### What SOTA does

- **llama.cpp**: CUDA, Metal, Vulkan, ROCm, CPU SIMD — runs on every platform.
- **AutoGPTQ**: CUDA (Marlin, ExLlamaV2, Triton backends), HuggingFace `optimum` integration.
- **QLoRA**: HuggingFace `bitsandbytes`, `peft`, `transformers` — full integration.

### Why this matters (for deployment)

If we want to deploy our 2-bit LUT in production, we need:
1. Multi-platform kernels (at minimum CUDA sm_80/89 for A100/Ada, Metal for Apple Silicon).
2. File format standardization (GGUF or safetensors).
3. HuggingFace integration (a `from_pretrained` loader).
4. Inference benchmarking (tokens/sec, memory footprint, vs. fp16 baseline).

This is a **large engineering effort** (~3–6 months) and is **not recommended until the accuracy gap is closed**. Productionizing a cos=0.95 model is premature.

### Expected cos improvement

- None (this is a deployment gap, not an accuracy gap).

### Implementation cost

- Multi-platform kernels: **3–6 months**.
- File format standardization: **2–4 weeks**.
- HuggingFace integration: **2–4 weeks** (requires upstream PRs).
- Total: **~4–8 months** of engineering, after the accuracy gap is closed.

---

## Summary: The Gap Hierarchy

| Gap | Description | Expected cos gain | Effort (eng-days) | Priority |
|---|---|---|---|---|
| 1 | Single-weight codebook (g=1) → vector (g=2) or additive (M=2) | +0.03–0.06 | 5–15 | **P0** |
| 2 | No pre-quant transformation (AWQ scale) | +0.01–0.03 | 1–2 | **P0** |
| 3 | No Hessian-inverse error propagation | +0.01–0.02 | 2–3 | **P1** |
| 4 | No outlier removal (dense/sparse) | +0.02–0.04 | 3–5 | **P0** |
| 5 | Group size 256 → 128 | +0.005–0.01 | 0.5 | **P0** |
| 6 | Gumbel-Softmax gradient damping | +0.005–0.02 | 0.5–7 | **P1** |
| 7 | No per-layer mixed precision | +0.01–0.03 | 6 | **P2** |
| 8 | No per-sub-group scale hierarchy | +0.01–0.02 | 3 | **P2** |
| 9 | Loss function (1-cos + norm_mse) | +0–0.015 | 0.5–3 | **P3** |
| 10 | Hardware/production readiness | 0 (deployment only) | 60–120 | **Deferred** |

**Total P0 (gaps 1, 2, 4, 5):** +0.045 to +0.13 cos, 9.5–22.5 eng-days.
**Total P0+P1 (adds gaps 3, 6):** +0.06 to +0.17 cos, 12–32.5 eng-days.

If all P0+P1 gains compound, we go from cos=0.95 to **cos=0.99–0.999+**. This is the path to SOTA.

The full implementation plan with concrete code patches is in `08_recommendations.md`.
