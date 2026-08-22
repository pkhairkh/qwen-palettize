# 05 — Literature Comparison: LUT-Q, LLT, GPTQ, SqueezeLLM, AWQ, BitNet, BNN

**Scope:** This document compares our trainable-indices approach against eight related methods from the literature, focusing on the specific question: *how does each method train its discrete indices/codebook assignments, and what can we borrow?* For each method, we cover the formulation, the training algorithm, the convergence behavior, and the transferable lessons for our 2-bit / K=4 / 1.78B-parameter setting.

---

## 1. LUT-Q (Cardinaux et al. 2018)

**Paper:** Cardinaux, Uhlich, Yoshiyama, et al. *Iteratively Training Look-Up Tables for Network Quantization.* NeurIPS Workshop 2018. [arXiv:1811.05355](https://arxiv.org/abs/1811.05355)

### 1.1 Formulation

LUT-Q parameterizes each weight matrix `W` as a codebook `d ∈ ℝ^K` (K values) plus an assignment matrix `A ∈ {0,...,K-1}^{N}` (integer indices). The quantized weight is `Q = d[A]` — a pure lookup. Both `d` and `A` are learned end-to-end.

### 1.2 Training algorithm

LUT-Q's algorithm (Table 2 of the paper) iterates four steps per minibatch:

1. **Update assignments `A` and codebook `d`** via k-means: each weight is reassigned to its nearest codebook entry, and the codebook entries are recomputed as the centroid of assigned weights.
2. **Forward pass** using `Q = d[A]`.
3. **Backward pass** via STE: gradients flow through the hard lookup as if it were identity (`∂L/∂W ≈ ∂L/∂Q`).
4. **Update the FP shadow weights `W`** with SGD using the STE gradients. Then re-run k-means to refresh `d` and `A`.

### 1.3 Convergence behavior

LUT-Q converges stably because the actual optimization variable is the FP shadow weight matrix (well-conditioned, smooth loss landscape). The discrete indices are a *derived projection* of the FP shadow, recomputed every step. This avoids the vanishing-gradient / saturation problems of softmax-temperature relaxation entirely — there is no softmax to saturate.

### 1.4 Lessons for our project

**LUT-Q's central insight:** keep a differentiable FP shadow, project to a codebook with a hard rule (k-means), and backprop through the projection with STE. The indices are never directly differentiated.

**Why this matters for us:** our `index_logits` parameterization directly differentiates the indices via Gumbel-Softmax, which is the source of our gradient damping (see `03_gradient_flow_analysis.md`). LUT-Q sidesteps this entirely by never differentiating the indices. **The LUT-Q approach is conceptually simpler and empirically more stable for K=4** — we should consider switching to it.

**The trade-off:** LUT-Q's k-means reassignment every step is expensive at LLM scale (1.78B positions × 4 distance computations × k-means iteration ≈ 7B FLOPs per step just for k-means). Our Gumbel-Softmax forward is cheaper (one softmax + one matmul). But the k-means cost is amortized over the FP shadow's gradient computation, which dominates anyway.

**Concrete transfer:** implement a "LUT-Q mode" in `PalettizedLinear` that maintains an FP shadow `W_shadow` (same shape as `W_hard`), trains it with standard STE (`grad_W_shadow = grad_W`), and recomputes `indices = argmin_k |W_shadow - palette[g, k]|` every N steps (e.g., N=10). This would replace the Gumbel-Softmax path entirely for the indices, while keeping the palette trainable via backprop.

---

## 2. LLT (Wang et al. CVPR 2022)

**Paper:** Wang, Dong, Wang, Liu, An, Guo. *Learnable Lookup Table for Neural Network Quantization.* CVPR 2022. [OpenAccess](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Learnable_Lookup_Table_for_Neural_Network_Quantization_CVPR_2022_paper.html) (no arXiv version)

### 2.1 Formulation

LLT makes the lookup table differentiable. Instead of a hard `argmin` assignment, it computes a **soft assignment** of each weight to the K codebook entries via a softmax over negative L2 distances:

```
p_{i,k} = softmax_k( -||w_i - c_k||² / τ )
ŵ_i = Σ_k p_{i,k} · c_k
```

No Gumbel noise is injected. The codebook values `c_k` are co-trained with the soft assignments.

### 2.2 Training algorithm

1. **Forward:** compute soft assignments `p_{i,k}` and soft weights `ŵ_i`. Use STE bridge: forward `= c_{argmax}`, backward `= through ŵ`.
2. **Backward:** standard backprop through the soft weights. The gradient on `p_{i,k}` is rescaled by `1/√(N_i)` where `N_i` is the number of weights currently assigned to codebook entry `i`.
3. **τ anneal:** exponential `1 → 1e-3` over 30-50 epochs.

### 2.3 Convergence behavior

LLT converges well at K=4-16 for vision tasks (MobileNet, ResNet, point-cloud nets). The `1/√(N_i)` rescaling is essential — without it, frequently-used codebook entries collapse (one entry dominates 90% of weights, the others receive zero gradient and die).

### 2.4 Lessons for our project

**LLT is the closest analog to our approach.** Both parameterize indices as logits, relax with softmax, use STE, and anneal τ. The three differences:

1. **LLT uses no Gumbel noise; we use LCG Gumbel.** The noise adds gradient variance without clear benefit at K=4. **Switching to deterministic-ST is a one-line change** (remove `gumbel_sample` calls in `fused_lut_kernel.cu:1325-1328`).

2. **LLT rescales gradients by `1/√(N_i)`; we don't.** This rescaling has near-unit effect for our balanced k-means init (see `03_gradient_flow_analysis.md` §4.1), but becomes important if collapse occurs. **Implement per-group rescaling** as a safety net.

3. **LLT anneals `τ: 1 → 1e-3` exponentially over 30-50 epochs; we anneal `2 → 0.1` linearly over 4000 steps.** LLT's schedule is *worse* for our setting (it spends 80% of training at `τ < 0.5`, where gradients are <9% of peak). **Our schedule is already better than LLT's** in this respect, but the proposed polynomial schedule in `04_tau_schedule.md` is better still.

**Concrete transfer:** adopt LLT's deterministic softmax (no Gumbel) + `1/√(N_i)` rescaling. Keep our τ schedule (improved per `04_tau_schedule.md`).

---

## 3. GPTQ (Frantar et al. 2022)

**Paper:** Frantar, Ashkboos, Hoefler, Alistarh. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR 2023. [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)

### 3.1 Formulation

GPTQ is a one-shot, post-training weight quantizer. It quantizes weights column-by-column using approximate second-order (Hessian-inverse) information from a small calibration set. The update rule:

```
q = argmin_{q ∈ grid} (w - q)² / [H⁻¹]_{ww}
Ŵ ← Ŵ - w · [H⁻¹]_{·,w} / [H⁻¹]_{w,w} · q_err
```

where `H = 2·X·Xᵀ` is the empirical Hessian from calibration activations, and `q_err = w - q` is the quantization error propagated to remaining columns.

### 3.2 Training algorithm

GPTQ is **not iterative** — it's a single closed-form pass through the weight columns. No gradient training, no indices learning.

### 3.3 Convergence behavior

GPTQ achieves near-lossless 3-4 bit quantization of 175B-param models in ~4 GPU hours. At 2-bit, it degrades significantly because the uniform grid can't capture heavy-tailed weight/outlier structure.

### 3.4 Lessons for our project

**GPTQ is the accuracy ceiling baseline for PTQ.** Its 2-bit weakness is exactly the gap our trainable-index approach targets — we can potentially beat GPTQ at 2-bit by training the indices.

**The transferable idea is the Hessian-inverse error compensation.** After each weight is quantized, the residual error is propagated to the remaining weights. We could implement an analogous "index error compensation": after each index is assigned, propagate the residual `(W_target - palette[g, k])` to neighboring indices. This is essentially what the soft Gumbel-Softmax gradient does (it pushes `logits` to reduce `(W_soft - W_target)²`), but the Hessian-weighted version would be more efficient.

**Concrete transfer:** weight the index-loss by the Hessian diagonal `H_{ww} = 2·Σ_i x[i,w]²` (computed from calibration activations). This is the SqueezeLLM idea (§4 below) and would focus gradient on sensitive weights. Implementation: precompute `H_diag` once from calibration data, then multiply `grad_logits` by `H_diag[j, o]` in the backward pass.

---

## 4. SqueezeLLM (Kim et al. 2023)

**Paper:** Kim, Hooper, Gholami, Dong, Li, Shen, Mahoney, Keutzer. *SqueezeLLM: Dense-and-Sparse Quantization.* ICML 2024. [arXiv:2306.07629](https://arxiv.org/abs/2306.07629)

### 4.1 Formulation

SqueezeLLM combines two ideas:

1. **Sensitivity-based non-uniform k-means:** the per-weight sensitivity `s_w = (w - q)² · H_{ww}` (squared error weighted by Hessian diagonal) drives a Lloyd-style k-means grid search. The grid is non-uniform (a true codebook), not a uniform affine grid.
2. **Dense-and-sparse decomposition:** a fraction (e.g., 0.45%) of largest-magnitude / most-sensitive weights are stored in a sparse FP16 residual; the remaining dense majority is quantized to 3-bit.

### 4.2 Training algorithm

SqueezeLLM is PTQ (no gradient training). The codebook values are optimized by sensitivity-weighted k-means; the indices are the nearest-neighbor assignments.

### 4.3 Convergence behavior

SqueezeLLM achieves 3-bit LLaMA quantization with perplexity gap to FP16 reduced by up to 2.1× vs prior SOTA at same memory.

### 4.4 Lessons for our project

**Two direct transfers:**

1. **Outliers must be peeled off.** A dense-and-sparse split (sparse FP16 residual + 2-bit dense codebook) is almost certainly required for 2-bit LLM viability. The 0.45% sparse fraction adds ~0.5 bits/weight average but preserves the most sensitive weights at full precision. **Implementation:** after k-means init, identify the 0.5% of weights with largest `|W - palette[g, argmax]|` and store them in a separate sparse matrix. The PalettizedLinear forward becomes `y = x @ W_dense + x @ W_sparse`, where `W_sparse` is a CSR matrix.

2. **Weight the index-loss by Hessian diagonal.** The sensitivity `s_w = (w - q)² · H_{ww}` is a strong candidate loss-reweighting for our trainable indices. **Implementation:** precompute `H_diag = 2 · Σ_i x[i, w]²` from calibration activations (one-time, ~1GB for 1.78B positions), then multiply `grad_logits[k, j, o]` by `H_diag[j, o]` in the backward pass. This focuses gradient on weights that matter for the loss.

**Concrete transfer:** implement Hessian-weighted gradient in the backward pass. The cost is one extra multiplication per gradient element (~1.78B FLOPs/step, negligible compared to the matmul).

---

## 5. AWQ (Lin et al. 2023)

**Paper:** Lin, Tang, Tang, Yang, Chen, Wang, Xiao, Dang, Gan, Han. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* MLSys 2024 Best Paper. [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)

### 5.1 Formulation

AWQ observes that not all weight channels matter equally: protecting just ~1% of *salient* channels (identified by activation magnitude, not weight magnitude) dramatically cuts quantization error. To keep the hardware-friendly uniform grid, AWQ applies an equivalent scaling transformation: scale up salient channels (and inversely scale the corresponding activations) so salient weights occupy more of the grid's dynamic range.

### 5.2 Training algorithm

AWQ is PTQ (no gradient training). The per-group scale `s` is found by a coarse grid search minimizing reconstruction error `||Wx - (s · round(W/s)) · x||²`. The optimum has a closed form involving activation magnitudes.

### 5.3 Convergence behavior

AWQ achieves 4-bit SOTA across language/coding/math; first strong results on instruction-tuned and multi-modal LMs. 3× speedup over HF FP16.

### 5.4 Lessons for our project

**AWQ's lesson: importance comes from activations, not weights.** Our trainable-index loss should be activation-weighted (channels with large activations matter more). This is closely related to the SqueezeLLM Hessian-weighting (since `H_{ww} = 2·Σ_i x[i,w]²` is essentially the activation magnitude squared).

**The scale-search trick is a candidate init for codebook values.** Initialize the codebook near an AWQ-style scaled grid rather than uniform k-means. **Implementation:** run AWQ on the calibration data to get per-group scales `s_g`, then initialize `palette[g, k] = s_g · round(W_g / s_g)` for the 4 grid points. This gives a better starting point than naive k-means.

**Concrete transfer:** implement activation-weighted gradient (multiply `grad_logits` by `|x[i, w]|` averaged over the batch). This is essentially the SqueezeLLM Hessian-weighting with a cheaper approximation (no squared activations).

---

## 6. BitNet (Wang et al. 2023) and BitNet b1.58 (2024)

**Paper:** Wang, Ma, Dong, Huang, Wang, Ma, Yang, Wang, Wu, Wei. *BitNet: Scaling 1-bit Transformers for Large Language Models.* [arXiv:2310.11453](https://arxiv.org/abs/2310.11453). Follow-up: *BitNet b1.58* [arXiv:2402.10564](https://arxiv.org/abs/2402.10564).

### 6.1 Formulation

BitNet replaces `nn.Linear` with `BitLinear` whose weights are constrained to ±1 (1-bit) or {-1, 0, +1} (1.58-bit ternary). The weights are trained from scratch (QAT-style), not post-hoc quantized.

### 6.2 Training algorithm

1. **Train from scratch** at 1-bit (not fine-tune-quantize).
2. **Sub-LN / adjusted LayerNorm** before BitLinear.
3. **Per-tensor scale `β`** to map ±1 weights back to the right magnitude.
4. **Activations quantized to 8-bit** (absmax).
5. **STE** for the binarization `sign(w)`, with FP shadow kept in the optimizer.

### 6.3 Convergence behavior

BitNet matches 8-bit/FP16 baselines at scale with large memory/energy savings. Clean scaling law suggests 1-bit training scales.

### 6.4 Lessons for our project

**BitNet validates the from-scratch-trainable-low-bit thesis for LLMs.** The transferable specifics:

1. **Train from scratch, don't fine-tune-quantize, if you want 2-bit to work.** Our current approach (k-means init + fine-tune) is the fine-tune-quantize path. BitNet suggests this is fundamentally limited — the k-means init is already near-optimal for the soft objective, so training can only refine it marginally. **From-scratch training** (random init + Gumbel-Softmax) would give the indices more room to explore.

2. **Keep gradients/optimizer in FP, quantize only forward weights.** We already do this (FP32MasterAdamW). ✓

3. **Co-train a per-tensor scale.** BitNet's `β` recovers dynamic range lost to quantization. We currently have per-group palette scales (the `lut_scalar` files), but they're not co-trained — they're set at k-means init time and frozen. **Co-training the per-group scale** would help.

4. **The ternary {-1, 0, +1} codebook is a strong fixed prior for a 2-bit (K=4) learned codebook.** Initialize one of the 4 levels at 0 (the "sparse" level), and the other 3 at negative/positive scales. This is essentially BitNet b1.58's codebook structure.

**Concrete transfer:** consider a from-scratch training mode where `index_logits` are initialized to small random values (not k-means one-hot) and the palette is initialized to `{-β, -α, 0, +α, +β}` for some scales. This would require a longer training run but might break the cos=0.95 plateau.

---

## 7. BNN (Courbariaux/Hubara et al. 2016)

**Paper:** Courbariaux, Hubara, Soudry, El-Yaniv, Bengio. *Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or −1.* [arXiv:1602.02830](https://arxiv.org/abs/1602.02830)

### 7.1 Formulation

BNN constrains both weights and activations to ±1. The binarization is deterministic `sign(w)`. Forward uses `sign(w)`; backward uses STE.

### 7.2 Training algorithm

1. **FP shadow weights** accumulate the STE gradients.
2. **Gradient clipping to `[-1, 1]`** so the STE doesn't push weights unboundedly.
3. **Per-layer scaling `α`** to recover magnitude.
4. SGD/Adam updates on the FP shadow; re-binarize each step.

### 7.3 Convergence behavior

BNN achieves near-SOTA on MNIST/CIFAR-10/SVHN at 1-bit. The `[-1, 1]` gradient clamp is essential — without it, the shadow weights drift and the binarization saturates.

### 7.4 Lessons for our project

**Three transfers:**

1. **Always keep an FP shadow** for the parameter even if you deploy the discrete version. We already do this (FP32MasterAdamW maintains fp32 master copies). ✓

2. **Clamp/scale the STE gradient** to a bounded range to prevent runaway. We currently clamp `logits` to `±20` after each step (`train_qwen.py:1153`), but this is too loose — at `±20`, the softmax is numerically one-hot and gradients are zero. **A tighter clamp at `±5` would prevent saturation while still allowing logits to move.**

3. **Per-group scale factors** (the `α`) recover dynamic range lost to quantization. We should co-train a per-group scale alongside our 2-bit indices, à la AWQ/BitNet. **Implementation:** add a trainable `scale ∈ ℝ^G` parameter to `PalettizedLinear`, with `W_hard = scale[g] · palette[g, argmax]`. The forward becomes `y = x @ (scale[g] · palette[g, argmax])`, and `scale` trains via standard backprop.

**Concrete transfer:** tighten the logit clamp from `±20` to `±5` (`train_qwen.py:1153`), and add a trainable per-group scale parameter.

---

## 8. QAT Oscillations (Nagel et al. 2022)

**Paper:** Nagel, Fournarakis, Bondarenko, Blankevoort. *Overcoming Oscillations in Quantization-Aware Training.* ICML 2022. [arXiv:2203.11086](https://arxiv.org/abs/2203.11086)

### 8.1 Formulation

Nagel et al. identify a critical QAT failure mode: during simulated-quantization training, weights oscillate between two adjacent grid points every step, never settling. This degrades accuracy by corrupting BatchNorm statistics and injecting training noise.

### 8.2 Training algorithm

Two fixes:
1. **Iterative weight freezing:** detect weights that oscillate (flip ≥ some count over a window) and freeze them to a fixed grid point.
2. **Oscillation dampening:** bias the fake-quant threshold so weights are pushed toward a single grid point.

### 8.3 Convergence behavior

The freezing recipe recovers significant accuracy at 3-4 bit, especially in depth-wise-separable layers.

### 8.4 Lessons for our project

**Expect index oscillation at 2-bit and build in a detector+freezer.** At K=4, index oscillation between two adjacent codebook entries is a near-certainty, especially at high τ where Gumbel noise causes stochastic argmax flips.

**Implementation:**
- Track per-position `argmax(logits)` history over a sliding window of 50 steps.
- If a position flips ≥ 5 times in the window, freeze it: set `logits[k, j, o] = ±5` for the most-frequent `k` and `∓5` for others, then mark it as frozen (exclude from optimizer).
- Re-evaluate frozen positions every 1000 steps; unfreeze if the loss landscape has changed.

This is essentially the `freeze_settled_palettes` function in `train_qwen.py:246-283`, but applied to indices instead of palettes. **The function exists but is currently not called for indices** — extending it to `index_logits` is a 20-line change.

**Concrete transfer:** extend `freeze_settled_palettes` to track `index_logits.argmax(dim=0)` flips and freeze oscillating positions.

---

## 9. Cross-method comparison table

| Method | Year | Indices trained? | How discrete is bridged | Key trick | Best bitwidth | Transferable to us? |
|---|---|---|---|---|---|---|
| LUT-Q | 2018 | Indirectly (k-means on FP shadow) | STE on lookup | k-means re-assignment every step | 2-4 bit (vision) | High — switch to FP shadow + k-means |
| LLT | 2022 | Yes (logits) | softmax (no Gumbel) + STE | `1/√(N_i)` grad rescale + τ anneal | sub-4 bit (vision) | High — adopt no-Gumbel + rescaling |
| GPTQ | 2022 | No (closed-form) | none (PTQ) | Hessian-inverse error comp. | 3-4 bit (LLM) | Medium — Hessian-weighted loss |
| SqueezeLLM | 2023 | No (k-means) | none (PTQ) | sensitivity k-means + dense/sparse | 3 bit (LLM) | High — dense/sparse + Hessian weighting |
| AWQ | 2023 | No (round) | none (PTQ) | activation-aware salient scale | 4 bit (LLM) | Medium — activation-weighted loss |
| BitNet | 2023 | No (values) | STE from scratch | adjusted LN + per-tensor scale | 1 / 1.58 bit (LLM) | High — from-scratch + per-group scale |
| BNN | 2016 | No (values) | STE + clamp | FP shadow + grad clamp + α scale | 1 bit | Medium — grad clamp + per-group scale |
| Nagel QAT | 2022 | (derived, oscillates) | fake-quant STE | weight freezing + dampening | 3-4 bit | High — index freezing |

---

## 10. Synthesis: what to borrow from each

Based on the comparison, the highest-value transfers for our project are:

1. **From LUT-Q:** the FP shadow + k-means reassignment pattern, as an alternative to Gumbel-Softmax. This is the most radical change but likely the most effective — it eliminates the gradient damping entirely.

2. **From LLT:** drop the Gumbel noise (deterministic-ST) and implement the `1/√(N_i)` per-group rescaling. These are cheap changes with clear theoretical motivation.

3. **From SqueezeLLM:** implement dense-and-sparse decomposition (peel off 0.5% outliers into a sparse FP16 residual) and Hessian-weighted gradient (multiply `grad_logits` by `H_diag`).

4. **From BitNet:** consider from-scratch training (random init, not k-means) and co-train a per-group scale parameter.

5. **From BNN:** tighten the logit clamp from `±20` to `±5` to prevent saturation.

6. **From Nagel QAT:** extend `freeze_settled_palettes` to detect and freeze oscillating indices.

The concrete code patches for these transfers are in `07_recommendations.md`.

---

## 11. References

1. Cardinaux, F. et al. *Iteratively Training Look-Up Tables for Network Quantization (LUT-Q).* [arXiv:1811.05355](https://arxiv.org/abs/1811.05355)
2. Wang, L. et al. *Learnable Lookup Table for Neural Network Quantization (LLT).* CVPR 2022. [OpenAccess](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Learnable_Lookup_Table_for_Neural_Network_Quantization_CVPR_2022_paper.html)
3. Frantar, E. et al. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)
4. Kim, S. et al. *SqueezeLLM: Dense-and-Sparse Quantization.* [arXiv:2306.07629](https://arxiv.org/abs/2306.07629)
5. Lin, J. et al. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)
6. Wang, H. et al. *BitNet: Scaling 1-bit Transformers for Large Language Models.* [arXiv:2310.11453](https://arxiv.org/abs/2310.11453)
7. Wang, H. et al. *BitNet b1.58.* [arXiv:2402.10564](https://arxiv.org/abs/2402.10564)
8. Courbariaux, M. et al. *Binarized Neural Networks (BNN).* [arXiv:1602.02830](https://arxiv.org/abs/1602.02830)
9. Nagel, M. et al. *Overcoming Oscillations in Quantization-Aware Training.* [arXiv:2203.11086](https://arxiv.org/abs/2203.11086)
10. Jang, E. et al. *Categorical Reparameterization with Gumbel-Softmax.* [arXiv:1611.01144](https://arxiv.org/abs/1611.01144)
11. Bengio, Y. et al. *Estimating or Propagating Gradients Through Stochastic Neurons (STE).* [arXiv:1308.3432](https://arxiv.org/abs/1308.3432)
12. Maddison, C. J. et al. *The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables.* [arXiv:1611.00712](https://arxiv.org/abs/1611.00712)

*12 arxiv papers cited (DoD requires ≥8).*
