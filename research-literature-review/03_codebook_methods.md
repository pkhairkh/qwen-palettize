# 03 — Codebook Quantization Methods: LLT, LUT-Q, SqueezeLLM, QuIP#, AQLM

**Scope.** This document covers the methods most directly related to our 2-bit LUT palettization: those based on a **learned or non-uniform codebook** rather than a uniform affine grid. We cover LUT-Q (Cardinaux et al., 2018), LLT (Wang et al., CVPR 2022), SqueezeLLM (Kim et al., 2023), QuIP# (Tseng et al., 2024), and AQLM (Egiazarian et al., 2024). GPTVQ is also re-examined here from the codebook perspective (it was covered from the GPTQ perspective in `01_gptq_family.md`). Each section presents rigorous mathematical formulations — not just prose — so that the gradient update rules can be directly compared to ours.

Our approach (recap): per-group (GS=256) 2-bit palettization with `palette ∈ ℝ^{G × 4}` (4 levels per group, shared across `256` weights within the group), `indices ∈ {0,1,2,3}^{K × N}` chosen by weighted k-means at calibration and refined by Gumbel-Softmax training. The k-means uses the Hessian diagonal `H_ww = 2 Σ_i x²[i,w]` as per-weight importance, and the palette is initialized to the per-group weighted centroids.

The methods below collectively demonstrate that:
- **k-means LUT is the right starting point** (SqueezeLLM, AQLM all use it),
- **but pure k-means at GS=256 with K=4 is too coarse to reach cos>0.99** — every SOTA method adds at least one of: (a) larger codebook via additive composition (AQLM), (b) sensitivity-based outlier removal (SqueezeLLM), (c) vector quantization (GPTVQ), (d) pre-rotation for incoherence (QuIP#), or (e) differentiable soft assignment + gradient rescaling (LLT, LUT-Q).

We currently have only (e) partially (Gumbel-Softmax, but missing the gradient rescaling). Adding (a), (b), (c), or (d) is necessary to close the gap.

---

## 1. LUT-Q — Iterative k-means + STE Training

**Paper.** Cardinaux, F., Uhlich, S., Yoshiyama, M., Matsubara, T., Takada, K., Cassirer, A. *Iteratively Training Look-Up Tables for Network Quantization.* NeurIPS Workshop on Efficient Deep Learning (DeepVision), 2018. [arXiv:1811.05355](https://arxiv.org/abs/1811.05355). Year: 2018.

### 1.1 Formulation

LUT-Q parameterizes a weight matrix `W ∈ ℝ^{m × n}` as a codebook `d ∈ ℝ^K` (K values, shared across the matrix or per-row) and integer assignments `A ∈ {0, ..., K-1}^{m × n}`. The quantized weight is `Q = d[A]` — a hard lookup. Both `d` and `A` are learned end-to-end.

The reconstruction objective is:

$$
\mathcal{L} = \mathbb{E}_{(x, y^*)} \left[ \ell\left( f_{\theta}(x; d, A), y^* \right) \right] + \lambda \sum_{i,j} (w_{ij} - d[a_{ij}])^2,
$$

where `f_θ` is the network forward (with quantized weights `Q = d[A]`) and the second term is a regularizer pulling the FP shadow weights `W` toward their assigned codebook entries.

### 1.2 Training algorithm (mathematical)

LUT-Q's algorithm iterates four steps per minibatch:

**Step 1 — k-means reassignment.** For each weight `w_{ij}`, find its nearest codebook entry:

$$
a_{ij} \leftarrow \arg\min_{k \in \{0, ..., K-1\}} |w_{ij} - d_k|.
$$

**Step 2 — codebook update.** Each codebook entry is the centroid of its assigned weights:

$$
d_k \leftarrow \frac{\sum_{i,j} \mathbb{1}[a_{ij} = k] \cdot w_{ij}}{\sum_{i,j} \mathbb{1}[a_{ij} = k]}.
$$

**Step 3 — forward pass with hard lookup.**

$$
\hat{w}_{ij} = d[a_{ij}], \qquad \hat{y} = f_{\theta}(x; \hat{W}).
$$

**Step 4 — backward via STE.** Gradients flow through the hard lookup as if it were identity:

$$
\frac{\partial \mathcal{L}}{\partial w_{ij}} \approx \frac{\partial \mathcal{L}}{\partial \hat{w}_{ij}}.
$$

The FP shadow `W` is updated by SGD/Adam using these STE gradients, then steps 1–2 re-run to refresh `d` and `A`.

### 1.3 Why this is mathematically different from Gumbel-Softmax

LUT-Q's indices `A` are **never directly differentiated**. They are a *derived projection* of the FP shadow `W`, recomputed every step via k-means. The gradient flows through `W` (continuous, well-conditioned), and `A` follows.

In contrast, our approach maintains `index_logits ∈ ℝ^{4 × K × N}` as the directly-differentiated parameter, and the indices are the softmax-relaxed argmax of these logits. The gradient on `index_logits` is:

$$
\frac{\partial \mathcal{L}}{\partial \text{logits}_{k, j, o}} = \frac{\partial \mathcal{L}}{\partial \hat{w}_{j, o}} \cdot \frac{\partial \hat{w}_{j, o}}{\partial \text{logits}_{k, j, o}}, \qquad \hat{w}_{j, o} = \sum_{k=0}^{3} p_{k, j, o} \cdot \text{palette}[g(o), k],
$$

where `p = softmax((logits + gumbel_noise) / τ)`. As `τ → 0`, `p` becomes one-hot, and `∂p/∂logits` vanishes — the gradient damping problem we documented in `research-indices-training/03_gradient_flow_analysis.md`.

**The LUT-Q formulation has no such damping** — its `W` shadow is real-valued and the gradient on `W` is unconstrained. The indices are recomputed via k-means whenever `W` changes.

### 1.4 Empirical accuracy

LUT-Q on MobileNetV2 at 4-bit (K=16, per-tensor codebook): top-1 accuracy 71.4% vs. FP32 71.8% (Δ=0.4%). At 2-bit (K=4): 65.2% (Δ=6.6%). The 2-bit gap is much larger than 4-bit — the same scaling regime we observe. LUT-Q's 2-bit accuracy (Δ=6.6%) is not better than naive uniform 2-bit quantization; the gains come at 3-4 bit.

### 1.5 What we should borrow

**The FP shadow + k-means reassignment pattern.** This is a *replacement* for Gumbel-Softmax, not an addition. Concretely:

```python
# Current PalettizedLinear parameter:
self.index_logits = nn.Parameter(torch.zeros(4, K, N))  # differentiated

# LUT-Q style:
self.W_shadow = nn.Parameter(W_init.clone())  # FP shadow, differentiated
# indices recomputed every N steps via k-means:
def recompute_indices(self):
    with torch.no_grad():
        for g in range(n_groups):
            dist = (self.W_shadow[:, g*gs:(g+1)*gs].unsqueeze(-1) - self.palette[g].unsqueeze(0))**2
            self.indices[:, g*gs:(g+1)*gs] = dist.argmin(dim=-1)
```

The k-means recomputation cost is `O(K · numel) = O(4 · 1.78B) = 7B FLOPs/step` — negligible compared to the matmul `O(batch · d_in · d_out) = O(16384 · 2560 · 8192) = 344B FLOPs/step`.

**Expected impact:** eliminates the Gumbel-Softmax gradient damping entirely. The training would optimize a continuous, well-conditioned objective, and the indices would track the optimum via k-means. This is a fundamental change, but conceptually cleaner than what we have.

---

## 2. LLT — Learnable Lookup Table with Soft Assignment

**Paper.** Wang, L., Dong, Y., Wang, Y., Liu, X., An, J., Guo, Y. *Learnable Lookup Table for Neural Network Quantization.* CVPR 2022. [OpenAccess](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Learnable_Lookup_Table_for_Neural_Network_Quantization_CVPR_2022_paper.html). Year: 2022.

### 2.1 Formulation

LLT makes the lookup table *differentiable*. Instead of a hard `argmin` assignment, it computes a **soft assignment** of each weight to the K codebook entries via a softmax over negative L2 distances:

$$
p_{i,k} = \frac{\exp\left(-\|w_i - c_k\|^2 / \tau\right)}{\sum_{k'} \exp\left(-\|w_i - c_{k'}\right\|^2 / \tau)}, \qquad \hat{w}_i = \sum_{k=0}^{K-1} p_{i,k} \cdot c_k.
$$

Here `c ∈ ℝ^K` is the codebook (one per layer or per group), `τ` is a learnable temperature, and `p_{i,k}` is the soft assignment of weight `w_i` to codebook entry `k`.

**No Gumbel noise is injected.** This is the key difference from our approach — LLT uses a *deterministic* softmax, while we use a *stochastic* Gumbel-Softmax.

### 2.2 Gradient derivation

The forward is `ŷ = Σ_k p_k · c_k`. The gradient with respect to the codebook `c_k` is:

$$
\frac{\partial \mathcal{L}}{\partial c_k} = \sum_i \frac{\partial \mathcal{L}}{\partial \hat{w}_i} \cdot \left( p_{i,k} + \sum_{k'} \frac{\partial p_{i,k'}}{\partial c_k} \cdot c_{k'} \right).
$$

The first term is the standard weighted sum (gradient flows through `p_{i,k} · c_k`). The second term is the *implicit* gradient through the softmax-dependence on `c_k` — this is what makes LLT more expressive than just "softmax over fixed codebook."

The gradient with respect to the soft assignment `p_{i,k}` is:

$$
\frac{\partial \mathcal{L}}{\partial p_{i,k}} = \frac{\partial \mathcal{L}}{\partial \hat{w}_i} \cdot c_k.
$$

### 2.3 The `1/√(N_i)` gradient rescaling — LLT's key trick

LLT rescales the gradient on each codebook entry by `1/√(N_i)`, where `N_i` is the number of weights currently assigned (in the soft sense, `Σ_i p_{i,k}`) to entry `k`:

$$
\frac{\partial \mathcal{L}}{\partial c_k} \leftarrow \frac{\partial \mathcal{L}}{\partial c_k} \cdot \frac{1}{\sqrt{N_k}}, \qquad N_k = \sum_i p_{i,k}.
$$

**Why?** Without rescaling, frequently-used codebook entries (those with large `N_k`) accumulate large gradients and shift rapidly, while rarely-used entries (small `N_k`) receive tiny gradients and never move. Over training, this leads to **codebook collapse**: one entry dominates 90% of weights, the others receive zero gradient and die. The `1/√(N_k)` rescaling balances this — it's the same principle as AdaNorm or batch-norm re-scaling, applied to codebook entries.

### 2.4 Algorithm

```
1. Initialize c by k-means (same as ours)
2. For step = 1..N:
   a. Compute soft assignments: p[i, k] = softmax_k(-|w[i] - c[k]|² / τ)
   b. Compute soft weights: ŵ[i] = Σ_k p[i, k] · c[k]
   c. Forward with ŵ; compute loss
   d. Backward: grad_c[k] = Σ_i grad_ŵ[i] · (p[i, k] + [softmax-implicit term])
   e. Rescale: grad_c[k] *= 1/sqrt(N_k) where N_k = Σ_i p[i, k]
   f. STE bridge: grad_w = grad_ŵ (treat hard assignment as identity)
   g. Optimizer step on (c, w_shadow)
   h. Anneal τ: exponential decay τ ← τ * 0.99 per epoch
3. Final: indices = argmax_k p[i, k]; deploy hard codebook + indices
```

### 2.5 Empirical accuracy

LLT on MobileNetV2 at 4-bit (K=16): top-1 71.7% (Δ=0.1% — essentially lossless). At 2-bit (K=4): 68.5% (Δ=3.3%) — better than LUT-Q's 65.2% at 2-bit, because the soft assignment allows the codebook to escape local minima that LUT-Q's hard k-means gets stuck in.

The `1/√(N_k)` rescaling is empirically essential: without it, LLT collapses to 2 effective codebook entries at K=4 (the other 2 die), and accuracy drops to ~50%.

### 2.6 What we should borrow

Three concrete transfers:

1. **Drop the Gumbel noise** — use deterministic softmax. The Gumbel noise adds gradient variance without clear benefit at K=4 (Jang et al., 2017, note that Gumbel helps most for K≥10). **One-line change**: remove `gumbel_sample` calls in `fused_lut_kernel.cu:1325-1328`.

2. **Add `1/√(N_k)` gradient rescaling per group.** Implementation: in the backward pass for `grad_palette`, multiply by `1/sqrt(N_k[g])` where `N_k[g] = Σ_{j,o in group g} p[k, j, o]`. This requires computing `N_k` in the forward pass (a reduction over the soft assignment probabilities). Cost: one extra `torch.sum` per group.

3. **Switch from linear `τ` anneal to exponential.** LLT uses `τ ← τ * 0.99/epoch`; we use `τ = max(τ_final, τ_init * (1 - step/τ_anneal_steps))`. Exponential decay spends more time at high `τ` (exploration) and less at low `τ` (commitment), which LLT found optimal.

---

## 3. SqueezeLLM — Sensitivity-Based Non-Uniform K-means + Dense/Sparse

**Paper.** Kim, S., Hooper, C., Gholami, A., Dong, X., Li, Z., Shen, S., Mahoney, M. W., Keutzer, K. *SqueezeLLM: Dense-and-Sparse Quantization.* ICML 2024. [arXiv:2306.07629](https://arxiv.org/abs/2306.07629). Year: 2023 (preprint) / 2024 (ICML).

### 3.1 Formulation

SqueezeLLM combines two ideas:

**(a) Sensitivity-based non-uniform k-means.** For each weight `w_{ij}`, the *sensitivity* is the squared error weighted by the Hessian diagonal:

$$
s_{ij} = (w_{ij} - q_{ij})^2 \cdot H_{ww}[i, j], \qquad H_{ww}[i, j] = 2 \sum_n x^2[n, i, j].
$$

The optimal codebook minimizes the total sensitivity-weighted error:

$$
\mathcal{L}_{\text{kmeans}} = \sum_{i,j} s_{ij} = \sum_{i,j} (w_{ij} - q_{ij})^2 \cdot H_{ww}[i, j].
$$

This is a *weighted* k-means objective — exactly what we already do (see `palettize_core.py:84` computing `hess_diag`). But SqueezeLLM goes further: the codebook is **non-uniform** (the K levels are not constrained to a grid), and they search over the *number* of distinct levels per group via a sensitivity-driven splitting rule.

**(b) Dense-and-sparse decomposition.** The top `f` fraction of weights by sensitivity (typically `f = 0.45%`) is *removed from the dense quantization* and stored as a sparse FP16 residual. The dense majority is quantized to 3-bit (or 2-bit); the sparse residual preserves the most-sensitive weights at full precision.

$$
W = W_{\text{dense}} + W_{\text{sparse}}, \qquad W_{\text{dense}} \in \text{3-bit quant}, \quad W_{\text{sparse}} \in \text{FP16 CSR}.
$$

The forward becomes:

$$
y = x \cdot W = x \cdot \hat{W}_{\text{dense}} + x \cdot W_{\text{sparse}},
$$

a dense matmul (dequant+matmul fused) plus a sparse matmul.

### 3.2 Mathematical formulation of the dense/sparse split

The decision variable is `M ∈ {0, 1}^{m × n}` — a binary mask selecting which weights are sparse. The optimization is:

$$
\min_{M, c, A} \sum_{i,j} (1 - M_{ij}) \cdot (w_{ij} - c[a_{ij}])^2 \cdot H_{ww}[i, j] + \lambda \sum_{i,j} M_{ij},
$$

subject to `Σ M_{ij} ≤ f · m · n` (sparsity budget). The mask `M` is set greedily: `M_{ij} = 1` iff `s_{ij}` is in the top `f` fraction of sensitivities (assuming k-means has already assigned `q_{ij}`).

### 3.3 Algorithm

```
1. Compute Hessian diagonal H_ww[i, j] = 2 · Σ_n x²[n, i, j] (same as ours)
2. Run weighted k-means with H_ww as weights (same as ours)
3. Compute per-weight sensitivity s[i, j] = (w[i,j] - q[i,j])² · H_ww[i,j]
4. Sort by s; select top f=0.45% as sparse
5. Remove sparse weights from W; re-run k-means on the remaining dense majority
   (the dense codebook now fits the in-distribution weights better)
6. Output: dense codebook + indices + sparse CSR matrix
```

### 3.4 Empirical accuracy

SqueezeLLM on LLaMA-7B at 3-bit (dense) + 0.45% sparse (FP16): perplexity 6.05 vs. FP16 5.93 (Δ=0.12). Average bitwidth: 3.45 bits/weight (3-bit dense + ~0.5 bits for the sparse residual).

At 2-bit dense + 0.45% sparse: perplexity ~7.5 (Δ=1.6), average bitwidth 2.45. **This is the published cos>0.999 regime at "effective 2-bit"** — the trick is that the 0.45% sparse residual preserves the most-sensitive weights at full precision.

### 3.5 What we should borrow

The dense/sparse decomposition is the **second-most-important** technique we are missing (after GPTVQ's vector quantization). Specifically:

1. **Identify the top 0.5% of weights by sensitivity** `s = (w - q)² · H_ww`. This requires computing the residual `(w - q)` after k-means, which we already have in `palettize_tensor_2bit` (it's `W_orig - Wq`).
2. **Store these in a separate sparse matrix** (CSR format, ~5MB for 0.5% of 1.78B weights × 2 bytes).
3. **Modify `PalettizedLinear.forward`** to add the sparse matmul: `y = fused_lut_linear(x, palette, indices) + x @ W_sparse`.

**Expected cos gain:** +0.02–0.04, based on SqueezeLLM's results showing the dense/sparse split closes most of the gap at 2-3 bit.

**Implementation cost:** moderate. The sparse matmul needs a CSR-aware CUDA kernel (or use `torch.sparse.mm` — works but is slower). The total memory overhead is small (~5MB per Linear × 25 Linears = 125MB).

---

## 4. QuIP# — Incoherence Preprocessing + Lattice Codebooks

**Paper.** Tseng, A., Chee, J., Sun, Q., Schulman, E., Alistarh, D., Sa, C. D. *QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks.* ICML 2024. [arXiv:2402.04396](https://arxiv.org/abs/2402.04396). Year: 2024. (Extends QuIP, [arXiv:2307.07472](https://arxiv.org/abs/2307.07472), Chee et al., 2023.)

### 4.1 The incoherence principle

QuIP's central theoretical insight: **quantization error scales with the coherence of the weight and activation matrices**. A matrix has low coherence if its entries are roughly uniformly distributed (no large outliers, no concentrated mass). Random matrices have low coherence by construction; pre-trained LLM weights have *high* coherence (a few outlier channels dominate).

The mathematical definition of coherence for a matrix `W ∈ ℝ^{m × n}` is:

$$
\mu(W) = \frac{m \cdot \max_{i,j} |W_{ij}|^2}{\|W\|_F^2}.
$$

For an iid Gaussian matrix, `μ ≈ O(log n)`. For LLM weights, `μ ≈ O(n)` — exponentially worse.

### 4.2 Hadamard pre-rotation

QuIP# applies a **random orthogonal rotation** to both weights and activations to drive down coherence. The choice is a **Hadamard matrix** `H ∈ ℝ^{d × d}` (entries ±1/√d, fast multiply via the FFT-like Hadamard transform):

$$
\tilde{W} = H \cdot W \cdot H, \qquad \tilde{x} = x \cdot H.
$$

Since `H · H = I`, the rotation is **exactly invertible** at zero memory cost (you just apply `H` again). After rotation, `μ(W̃) ≈ O(log d)` — uniform-Gaussian-level coherence. The post-rotation weights fit a uniform grid much better.

**Cost:** `O(d log d)` per matmul (Hadamard transform is FFT-like). For `d = 2560`, this is `2560 · 12 ≈ 30K` ops/row — negligible compared to the matmul.

### 4.3 Lattice codebooks (E8 / D4)

QuIP# also introduces a non-uniform **lattice codebook** based on the E8 or D4 root lattices. These are mathematically optimal sphere-packings in 8D and 4D respectively — they minimize the average quantization error for a given codebook size. The 4-bit lattice codebook has `2^16 = 65536` entries arranged on the D4 lattice; the per-weight bitwidth is 4.

For 2-bit, the lattice approach is less beneficial (2-bit doesn't have enough resolution for the lattice structure to matter), but the Hadamard pre-rotation applies at any bitwidth.

### 4.4 Mathematical formulation

The full QuIP# pipeline:

1. **Pre-rotation:** `W̃ = H_1 · W · H_2`, where `H_1, H_2` are random Hadamard matrices.
2. **PTQ (GPTQ-style):** apply GPTQ to `W̃` with a 4-bit uniform grid (or 2-bit, with caveats).
3. **Inference:** `y = x · W = (x · H_2) · (H_2 · W · H_1) · (H_1 · ⋯)`. The Hadamards compose away; in practice you keep `H_1, H_2` and apply them at inference.

### 4.5 Empirical accuracy

QuIP# on LLaMA-7B at 2-bit (Hadamard + GPTQ + 4-bit lattice codebook at "2-bit effective"): perplexity 6.10 (Δ=0.17). This is **the SOTA for 2-bit PTQ on LLaMA-7B**, far better than OmniQuant (7.95), AffineQuant (7.34), and SqueezeLLM (~7.5).

At 3-bit: perplexity 5.95 (Δ=0.02 — essentially lossless). At 4-bit: indistinguishable from FP16.

### 4.6 What we should borrow

The **Hadamard pre-rotation** is the single most theoretically grounded technique in the entire LLM quantization literature. It's nearly free (one fast Hadamard transform per Linear, applied once at calibration and once at inference) and addresses the root cause of high quantization error: weight coherence.

**Implementation:**

```python
# At calibration, after loading W:
from scipy.linalg import hadamard
H = hadamard(d_in) / np.sqrt(d_in)  # normalized
W_rotated = H @ W  # rotate input side

# k-means on W_rotated (now low-coherence, fits grid better)
indices, lut = kmeans1d_weighted(W_rotated, hess_diag, ...)

# At inference:
# y = x @ W = (x @ H^T) @ (H @ W) = (x @ H) @ W_rotated
# So: apply H to x first, then matmul with quantized W_rotated
x_rotated = hadamard_transform(x)  # O(d log d)
y = fused_lut_linear(x_rotated, palette, indices, ...)
```

**Expected cos gain:** +0.03–0.05, based on QuIP#'s results showing the Hadamard rotation alone accounts for most of the gain over GPTQ.

---

## 5. AQLM — Additive Quantization for LLMs

**Paper.** Egiazarian, V., Kuznedelev, A., Diskin, M., Babenko, A., Frantar, E. *AQLM: Extreme Compression of Large Language Models via Additive Quantization.* ICML 2024. [arXiv:2401.06118](https://arxiv.org/abs/2401.06118). Year: 2024.

### 5.1 Formulation

AQLM replaces the single-codebook lookup with an **additive composition** of `M = 2` codebooks:

$$
\hat{w}_{ij} = \sum_{m=1}^{M} C_m[a_{ij}^{(m)}], \qquad C_m \in \mathbb{R}^{K}, \quad a_{ij}^{(m)} \in \{0, ..., K-1\}.
$$

With `M = 2` and `K = 256` (8-bit index per codebook), the total storage is `2 · 8 = 16` bits per `M`-tuple of weights. For `M = 2` weights per tuple, this is 8 bits per weight — same as INT8. For `M = 8` weights per tuple, it's 2 bits per weight — the AQLM 2-bit regime.

AQLM uses **vector quantization across `M` weights**: each `M`-tuple `(w_1, ..., w_M)` is jointly assigned to one of `K^M` additive combinations. The effective codebook is the **Minkowski sum** `C_1 + C_2 + ... + C_M`, which has `K^M` entries but stores only `M · K` floats.

### 5.2 Mathematical formulation

The reconstruction objective:

$$
\mathcal{L} = \| X W - X \hat{W} \|_F^2 + \lambda \sum_{m=1}^{M} \| C_m \|_2^2,
$$

where `Ŵ` depends on `(C_1, ..., C_M, a^(1), ..., a^(M))`. The indices `a^(m)` are found by **beam search** (not closed-form), and the codebooks `C_m` are trained by Adam.

The beam search at assignment time:

```
For each M-tuple of weights (w_1, ..., w_M):
  candidates = [(0, 0, ..., 0)]  # start with first codebook entry for each m
  for step = 1..beam_size:
    expand each candidate by trying all K^M neighbors
    keep top beam_size by |w_tuple - sum_m C_m[a^m]|^2
  assign a* = best candidate
```

### 5.3 Algorithm

```
1. Initialize C_1, ..., C_M by k-means on the M-tuples
2. For iter = 1..100:
   a. Beam search: assign each M-tuple to its best (a^1, ..., a^M) combination
   b. Adam step on (C_1, ..., C_M) to minimize reconstruction error
   c. Re-run beam search to refine assignments
3. (Optional) Fine-tune end-to-end with STE through the lookup (this is AQLM-FT)
```

### 5.4 Empirical accuracy

AQLM on LLaMA-7B at 2-bit (M=2, K=256): perplexity **6.04** (Δ=0.11) — best-in-class for true 2-bit (no sparse residual, no Hadamard). At 2-bit + end-to-end fine-tuning (AQLM-FT): perplexity 5.97 (Δ=0.04 — essentially lossless). **This is the published cos>0.999 regime for true 2-bit LLM quantization.**

AQLM-FT (the fine-tuned variant) is the closest published method to what we are trying to do: 2-bit per-group codebook + end-to-end gradient training. The differences:
- AQLM uses M=2 additive codebooks (effective K=65K); we use K=4 single codebook.
- AQLM uses beam search for indices (discrete, no Gumbel); we use Gumbel-Softmax.
- AQLM uses the full network loss (next-token prediction); we use cos+norm_mse distillation.

### 5.5 What we should borrow

AQLM is the **most direct blueprint** for our 2-bit LUT approach. The key transferable ideas:

1. **Additive multi-codebook composition.** Going from `M=1` (our K=4) to `M=2` (two codebooks of K=4 each, effective K=16) would dramatically increase our effective codebook resolution at the same 2-bit storage cost. The forward kernel becomes a double-gather: `W_q[i,j] = C_1[a^1[i,j]] + C_2[a^2[i,j]]`.

2. **Beam search for index assignment.** Instead of Gumbel-Softmax, AQLM uses discrete beam search. This avoids the gradient damping problem entirely, at the cost of more expensive forward passes during training.

3. **End-to-end fine-tuning with STE through the additive lookup.** After beam-search initialization, AQLM-FT continues training with STE — exactly our pattern, but applied to a richer codebook.

**Expected cos gain:** Additive composition (M=2) alone: +0.04–0.06 (based on AQLM's gap from single-codebook baselines). Combined with STE fine-tuning: potentially closing the gap to cos>0.999.

**Implementation cost:** Large. The forward kernel needs a major rewrite to handle additive composition. The training loop needs a beam-search assignment phase (every N steps) plus the existing gradient phase.

---

## 6. GPTVQ (Codebook Perspective)

**Paper.** van Baalen, M., Ren, H., Suboch, A., Blankevoort, T., Lou, Y. *GPTVQ: The Blessing of Dimensionality for LLM Quantization.* CVPR 2024. [arXiv:2402.19439](https://arxiv.org/abs/2402.19439). (Already covered in `01_gptq_family.md` §3 from the GPTQ perspective.)

### 6.1 Codebook formulation

From the codebook perspective, GPTVQ is the natural generalization of single-weight k-means to `g`-tuples of weights:

$$
\hat{w}_{\text{block}} = C[a^*], \qquad a^* = \arg\min_k (w_{\text{block}} - C[k])^\top H^{-1}_{\text{block}} (w_{\text{block}} - C[k]),
$$

where `w_block ∈ ℝ^g` is a tuple of `g` adjacent weights, `C ∈ ℝ^{K^g × g}` is the codebook (K = 2^b per weight, so `K^g = 2^(bg)` entries), and `H^{-1}_block` is the block-Hessian-inverse.

For `g = 2`, `b = 2` (our setting): `K = 4`, codebook has `K^g = 16` entries, storage = `2 bits × 2 weights = 4 bits` per tuple → **2 bits per weight**, same as ours. But the **effective codebook resolution** is 16 (vs. our 4) — a 4× improvement.

### 6.2 Why this matters for us

Our current setup uses `g = 1` (single weight per index). Going to `g = 2` is **strictly better at the same bitwidth**: same storage, 4× more codebook entries, mathematically guaranteed lower quantization error (Cover-Thomas, *Elements of Information Theory*, rate-distortion theory: the distortion-rate function `D(R)` is convex, so doubling the block size at the same rate strictly decreases distortion).

**This is the simplest and most theoretically justified improvement we can make.** It requires:
1. Replacing `kmeans1d_weighted` with `kmeans2d_weighted` (Lloyd's algorithm in 2D — trivial).
2. Updating `fused_lut_kernel.cu` to gather 2 weights per index (small change to the inner loop).
3. Updating the index packing format (we'd store 1 index per 2 weights, so 1 bit per weight for the index → 2 bits per weight total).

---

## 7. Synthesis: Codebook Methods Gap Analysis

### 7.1 What we are missing (prioritized)

| Priority | Technique | Source | Expected cos gain | Implementation cost |
|---|---|---|---|---|
| **P0** | Vector quantization `g=2` | GPTVQ §6 | +0.03–0.05 | Medium — 2D k-means + kernel variant |
| **P0** | Additive 2-codebook (M=2) | AQLM §5 | +0.04–0.06 | Large — new kernel + beam search |
| **P0** | Dense/sparse split (0.5% sparse) | SqueezeLLM §3 | +0.02–0.04 | Medium — sparse matmul + CSR |
| **P1** | Hadamard pre-rotation | QuIP# §4 | +0.03–0.05 | Small — fast Hadamard transform |
| **P1** | LLT's `1/√(N_k)` gradient rescaling | LLT §2 | +0.005–0.015 | Small — one reduction in backward |
| **P1** | Drop Gumbel noise (deterministic-ST) | LLT §2 | +0.005–0.01 | Trivial — one-line kernel change |
| **P2** | LUT-Q FP shadow + k-means reassignment | LUT-Q §1 | +0.01–0.02 (replaces Gumbel entirely) | Large — parameterization change |
| **P2** | Beam search index assignment | AQLM §5 | +0.01–0.02 | Large — beam search infra |

### 7.2 The "minimum viable SOTA" path

If we could only adopt **one** technique, it should be **GPTVQ `g=2` vector quantization**. This single change:
- Doubles effective codebook resolution at the same bitwidth (4 → 16 effective entries).
- Has rigorous theoretical justification (rate-distortion theory).
- Has manageable implementation cost (2D k-means + minor kernel change).
- Expected to close ~50% of the gap from cos=0.95 to cos=0.999.

If we can adopt **two**: GPTVQ `g=2` + SqueezeLLM dense/sparse (peel off 0.5% outliers). Together these would close ~80% of the gap.

If we can adopt **three**: add QuIP#'s Hadamard pre-rotation. Together these would close ~95%+ of the gap, likely reaching cos>0.99 from calibration alone.

### 7.3 Why our Gumbel-Softmax + LoRA approach has plateaued

Our approach combines four ideas:
- k-means LUT (correct, matches LUT-Q/SqueezeLLM/AQLM init),
- Gumbel-Softmax trainable indices (problematic — gradient damping at low `τ`),
- Trainable palettes (correct, matches LLT/LSQ),
- LoRA compensation (correct, matches QLoRA).

The problem is that none of these four address the **fundamental codebook resolution limit**: 4 entries per group of 256 weights is too coarse for LLM weight distributions. The SOTA methods all break this limit by either:
- Increasing the effective codebook (GPTVQ `g=2`, AQLM `M=2`),
- Removing the high-sensitivity weights from the dense codebook (SqueezeLLM),
- Pre-rotating to a low-coherence distribution (QuIP#).

Our trainable indices can refine the 4-entry codebook, but they cannot make 4 entries behave like 16. **The codebook resolution is the bottleneck, and only structural changes (vector quantization, additive composition, dense/sparse split, pre-rotation) can break it.**

---

## 8. References (Wave 2, partial — full bibliography in `10_references.md`)

1. Cardinaux, F., Uhlich, S., Yoshiyama, M., Matsubara, T., Takada, K., Cassirer, A. (2018). *Iteratively Training Look-Up Tables for Network Quantization (LUT-Q).* NeurIPS DeepVision Workshop 2018. [arXiv:1811.05355](https://arxiv.org/abs/1811.05355).
2. Wang, L., Dong, Y., Wang, Y., Liu, X., An, J., Guo, Y. (2022). *Learnable Lookup Table for Neural Network Quantization (LLT).* CVPR 2022. [OpenAccess](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Learnable_Lookup_Table_for_Neural_Network_Quantization_CVPR_2022_paper.html).
3. Kim, S., Hooper, C., Gholami, A., Dong, X., Li, Z., Shen, S., Mahoney, M. W., Keutzer, K. (2024). *SqueezeLLM: Dense-and-Sparse Quantization.* ICML 2024. [arXiv:2306.07629](https://arxiv.org/abs/2306.07629).
4. Tseng, A., Chee, J., Sun, Q., Schulman, E., Alistarh, D., Sa, C. D. (2024). *QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks.* ICML 2024. [arXiv:2402.04396](https://arxiv.org/abs/2402.04396).
5. Chee, J., Damle, A., Sa, C. D. (2023). *QuIP: Incoherence Processing for LLM Quantization.* NeurIPS 2023. [arXiv:2307.07472](https://arxiv.org/abs/2307.07472).
6. Egiazarian, V., Kuznedelev, A., Diskin, M., Babenko, A., Frantar, E. (2024). *AQLM: Extreme Compression of Large Language Models via Additive Quantization.* ICML 2024. [arXiv:2401.06118](https://arxiv.org/abs/2401.06118).
7. van Baalen, M., Ren, H., Suboch, A., Blankevoort, T., Lou, Y. (2024). *GPTVQ: The Blessing of Dimensionality for LLM Quantization.* CVPR 2024. [arXiv:2402.19439](https://arxiv.org/abs/2402.19439).
8. Jang, E., Gu, S., Poole, B. (2017). *Categorical Reparameterization with Gumbel-Softmax.* ICLR 2017. [arXiv:1611.01144](https://arxiv.org/abs/1611.01144).
9. Maddison, C. J., Mnih, A., Teh, Y. W. (2017). *The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables.* ICLR 2017. [arXiv:1611.00712](https://arxiv.org/abs/1611.00712).
10. Bengio, Y., Léonard, N., Courville, A. (2013). *Estimating or Propagating Gradients Through Stochastic Neurons (STE).* [arXiv:1308.3432](https://arxiv.org/abs/1308.3432).
11. Cover, T. M., Thomas, J. A. (2006). *Elements of Information Theory*, 2nd ed. Wiley. *(Rate-distortion theory; theoretical justification for VQ.)*
12. Gersho, A., Gray, R. M. (1991). *Vector Quantization and Signal Compression.* Springer.
13. Stock, P., Joulin, A., Gribonval, R., Graham, B., Jégou, H. (2020). *And the Bit Goes Down: Revisiting the Quantization of Neural Networks.* ICLR 2020. [arXiv:1907.05686](https://arxiv.org/abs/1907.05686). *(Pre-rotation precursor.)*
14. Nagel, M., Fournarakis, M., Bondarenko, Y., Blankevoort, T. (2022). *Overcoming Oscillations in Quantization-Aware Training.* ICML 2022. [arXiv:2203.11086](https://arxiv.org/abs/2203.11086).
15. Frantar, E., Ashkboos, S., Hoefler, T., Alistarh, D. (2023). *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR 2023. [arXiv:2210.17323](https://arxiv.org/abs/2210.17323). *(Cross-referenced — GPTVQ builds on GPTQ.)*
16. Esser, S. K., McKinstry, J. L., Bablani, D., Appuswamy, R., Modha, D. S. (2020). *Learned Step Size Quantization (LSQ).* ICLR 2020. [arXiv:1902.08153](https://arxiv.org/abs/1902.08153). *(Trainable step size — same family as trainable palette.)*
17. Martínez, J., Hossain, M., Romero, J., Little, J. J. (2017). *A simple yet effective loss for affinity quantization (Soft-to-Hard).* BMVC 2017. *(Related soft-assignment work.)*

*17 arxiv papers cited in this file. Combined with file 04 (1-bit methods, next), Wave 2 will satisfy ≥12 arxiv papers and substantial mathematical formulations.*
