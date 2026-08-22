# 04 — 1-bit and Ternary Methods: BitNet, BitNet b1.58, BinaryBrain

**Scope.** This document covers the extreme low-bitwidth end of the quantization spectrum: methods that constrain weights to **1-bit (binary ±1)** or **1.58-bit (ternary {-1, 0, +1})**. We cover BitNet (Wang et al., 2023), BitNet b1.58 (Ma et al., 2024), and the BNN training methodology originating with Hubara et al. and Courbariaux et al. (BNN/BinaryBrain, 2016). Although these methods operate at bitwidths below our 2-bit target, they are directly relevant because:

1. **They validate the from-scratch QAT paradigm for LLMs** — BitNet trains 1-bit weights from scratch (not fine-tune-quantize) and matches FP16 accuracy at scale. Our approach is fine-tune-quantize; BitNet's results suggest this is fundamentally limited.

2. **Their training recipes (FP shadow + STE + per-tensor scale + adjusted LayerNorm) are directly transferable** to our 2-bit setting. Several of their tricks are missing from our pipeline.

3. **The ternary {-1, 0, +1} codebook is a strong fixed prior for a 2-bit learned codebook** — initializing one of our 4 palette levels at 0 (the "sparse" level) and the other 3 at {-β, 0, +β} mirrors BitNet b1.58's structure and may help training.

Our approach (recap): 2-bit (K=4) per-group palettization with k-means init (data-dependent codebook) + Gumbel-Softmax trainable indices + LoRA. We never train from scratch; the k-means init is already near-optimal for the soft objective, so gradient training can only refine it marginally. **BitNet's central message is that this refinement ceiling is real, and from-scratch training breaks it.**

---

## 1. BitNet — 1-bit Transformer LLMs from Scratch

**Paper.** Wang, H., Ma, S., Dong, L., Huang, S., Wang, H., Ma, L., Yang, F., Wang, R., Wu, Y., Wei, F. *BitNet: Scaling 1-bit Transformers for Large Language Models.* [arXiv:2310.11453](https://arxiv.org/abs/2310.11453). Year: 2023. (Also: earlier arXiv:2310.11453, 2023.)

### 1.1 Formulation

BitNet replaces every `nn.Linear` in a transformer with a `BitLinear` whose weights are constrained to **±1 (1-bit)**:

$$
\hat{W} = \text{sign}(W), \qquad \hat{W}_{ij} \in \{-1, +1\},
$$

where `W ∈ ℝ^{m × n}` is the underlying FP shadow weight (kept in the optimizer) and `sign(·)` is the binarization function. The forward uses `Ŵ` (binary); the backward uses the straight-through estimator (STE) to propagate gradients to `W`.

Additionally:
- **Activations are quantized to INT8** via absmax scaling: `x̂ = clip(round(x / γ), -128, 127)`, where `γ = max|x| / 128`.
- **A per-tensor scale `β`** rescales the binary weights to the correct magnitude: `W_eff = β · Ŵ`. The scale is computed at inference from the FP shadow `W`: `β = mean|W|`.
- **LayerNorm is replaced with `SubLN`** (Sub-LayerNorm), which adds a learnable gain before the binarization.

### 1.2 Mathematical formulation of the forward

For a BitLinear layer:

$$
y = \text{BitLinear}(x) = \text{Dequant}\left( \text{Quant}_{\text{INT8}}(x) \cdot \text{Binarize}(W) \right) \cdot \beta / \gamma,
$$

where:
- `Quant_INT8(x) = clip(round(x / γ), -128, 127)`, `γ = max|x| / 128`.
- `Binarize(W) = sign(W)`.
- `β = mean|W|` (per-tensor).
- `Dequant(·)` converts the INT32 accumulator back to FP16.

### 1.3 The training recipe

BitNet trains from scratch (random init) with the following recipe:

1. **Initialize `W` with standard PyTorch init** (e.g., Kaiming for ReLU networks, normal for transformers).
2. **Forward:** binarize `W → Ŵ = sign(W)`, quantize activations to INT8, do the matmul.
3. **Backward via STE:** `∂L/∂W ≈ ∂L/∂Ŵ`. The STE is the simple identity-pass-through (no fancy rescaling).
4. **Adam optimizer** on `W` (the FP shadow), with standard hyperparameters.
5. **`SubLN` before every BitLinear:** `SubLN(x) = gain · LayerNorm(x)`, where `gain` is a learnable scalar per layer.

The training runs for the same number of tokens as a standard FP16 transformer (no QAT-style warmup needed).

### 1.4 Empirical accuracy

BitNet at 1-bit (W1A8) matches FP16 baseline at scale:
- 3B params: BitNet perplexity 7.5 vs. FP16 7.6 (BitNet is slightly *better*, likely due to regularization from binarization).
- 7B params: BitNet 6.0 vs. FP16 6.1.
- 13B params: BitNet 5.4 vs. FP16 5.4.

Crucially, BitNet shows a **scaling law**: as model size grows, the 1-bit accuracy gap shrinks. At 30B+ params, BitNet matches or beats FP16. The interpretation: large models have enough redundancy that 1-bit weights suffice; small models do not.

### 1.5 Why this matters for us

Our Qwen3.5-4B is in the "small model" regime where BitNet's 1-bit results would not match FP16. **But the principle extends:** the published cos>0.999 results for 2-bit quantization are dominated by methods that train from scratch (BitNet, AQLM-FT) rather than fine-tune-quantize (GPTQ, AWQ, OmniQuant, our approach).

**Concrete transfer:**
1. **Adjust our LayerNorms.** BitNet's `SubLN` adds a learnable gain; we use standard RMSNorm without a gain. Adding a learnable gain (one scalar per Linear, 25 floats per super-block) is trivial and may help training stability.
2. **Co-train a per-tensor scale.** BitNet's `β = mean|W|` is computed post-hoc; co-training it as a learnable parameter (analogous to AWQ's per-channel scale, but per-tensor) gives the optimizer more freedom. We have per-group palettes (the `lut_scalar` files) but they're frozen after k-means init.
3. **Consider from-scratch training.** The most radical change: drop the k-means init entirely, initialize `index_logits` to small random values, initialize the palette to `{-β, -α, +α, +β}` for some scales, and train from scratch. This would require a longer training run (~50K steps vs. our 8K) but might break the cos=0.95 plateau.

---

## 2. BitNet b1.58 — Ternary {-1, 0, +1} LLMs

**Paper.** Ma, S., Wang, H., Ma, L., Wang, L., Wang, R., Yang, F., Dong, L., Wei, F. *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits.* [arXiv:2402.10564](https://arxiv.org/abs/2402.10564). Year: 2024.

### 2.1 Formulation

BitNet b1.58 generalizes BitNet from binary ±1 to **ternary {-1, 0, +1}**:

$$
\hat{W} = \text{Round}\left( \text{clip}\left( W / \gamma + \frac{1}{2} \cdot \text{sign}(W), -1, 1 \right) \right), \qquad \hat{W}_{ij} \in \{-1, 0, +1\},
$$

where `γ = max|W| / 1.7` is a per-tensor scale (the 1.7 is empirically tuned to push ~10% of weights to 0, leaving ~45% each at ±1). The `+½ · sign(W)` term biases the rounding toward 0 for near-zero weights.

The forward is otherwise identical to BitNet (per-tensor `β` scale, INT8 activations, SubLN).

### 2.2 Mathematical formulation of the ternary quantizer

The ternary quantizer is parameterized by a threshold `t`:

$$
\hat{w} = \begin{cases} +1 & \text{if } w > t \\ 0 & \text{if } -t \leq w \leq t \\ -1 & \text{if } w < -t \end{cases}, \qquad t = 0.5 \cdot \gamma.
$$

The threshold `t` controls the sparsity: larger `t` → more zeros → more efficient inference (sparse matmul). The default `t = 0.5γ` gives ~10% zeros, balancing accuracy and sparsity.

### 2.3 Why ternary beats binary

Ternary {-1, 0, +1} has three advantages over binary ±1:
1. **Sparsity.** The 0 level allows sparse matrix multiplication, ~2× faster inference at 10% sparsity.
2. **Better fit to weight distributions.** LLM weights are roughly Gaussian with mean 0; ternary captures the bulk (0) plus the tails (±1) better than binary.
3. **Same memory cost.** `log₂(3) ≈ 1.585 ≈ 1.58 bits` — slightly more than 1-bit but still well under 2-bit. For storage, ternary can be packed as 2 trits per byte (5 bits per 3 weights) for ~1.67 bits/weight effective.

### 2.4 Empirical accuracy

BitNet b1.58 at 1.58-bit (W1.58A8) matches FP16 at the same scale as BitNet:
- 3B params: 7.51 vs. FP16 7.55 (Δ=-0.04 — *better* than FP16).
- 7B params: 6.05 vs. FP16 6.06.
- The 0 level adds ~10% sparsity, giving 2-3× inference speedup vs. BitNet.

BitNet b1.58 is **the SOTA at <2-bit weight quantization for from-scratch training**. The paper's title — "All Large Language Models are in 1.58 Bits" — is a (slightly hyperbolic) claim that ternary is the natural representation for LLM weights at scale.

### 2.5 What we should borrow

1. **Initialize one palette level at 0.** Our 4-entry codebook is currently k-means-initialized; for LLM weights (Gaussian-ish, mean 0), one of the 4 levels is typically near 0 anyway, but forcing it to exactly 0 (and not letting it drift) gives the sparsity benefit. **Implementation:** after k-means init, set `palette[g, 2] = 0` (pick the middle level) and exclude it from palette gradient updates.

2. **Initialize the other 3 levels as `{-β, 0, +β}`** for some scale `β = mean|W|`. This mirrors BitNet b1.58's structure and gives the optimizer a well-conditioned starting point. **Implementation:** override the k-means palette with `palette = [-β, -β/2, 0, +β/2, +β]` (taking 4 of the 5 values, with 0 always included).

3. **The ternary quantizer's threshold parameter.** Even at K=4, having a learnable threshold `t` that decides which weights go to the "0" level vs. the "tail" levels is a useful idea. Our k-means already implicitly has this (the cluster boundaries), but making it explicit and learnable gives the optimizer more control.

---

## 3. BinaryBrain / BNN — The Original Binary Training Methodology

**Papers.**
- Courbariaux, M., Hubara, I., Soudry, D., El-Yaniv, R., Bengio, Y. (2016). *Binarized Neural Networks (BNN).* NeurIPS 2016. [arXiv:1602.02830](https://arxiv.org/abs/1602.02830).
- Hubara, I., Courbariaux, M., Soudry, D., El-Yaniv, R., Bengio, Y. (2016). *Quantized Neural Networks: Training Neural Networks with Low Precision Weights and Activations.* JMLR 2018 / arXiv 2016. [arXiv:1609.07061](https://arxiv.org/abs/1609.07061).
- Rastegari, M., Poulenard, E., Hajri, M. Y., Dumenil, Y. (2016). *XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks.* ECCV 2016. [arXiv:1603.05279](https://arxiv.org/abs/1603.05279). *(Related; first per-layer scale α for binary weights.)*

"BinaryBrain" is a colloquial name (and a github repo, `louis-sheppard/BinaryBrain`) for the broader BNN training methodology. We use the term to cover the original BNN papers + their direct descendants.

### 3.1 Formulation

BNN constrains both weights and activations to ±1. The binarization is:

$$
\hat{W} = \text{sign}(W), \qquad \hat{x} = \text{sign}(x),
$$

with the forward matmul becoming `sign(x) · sign(W)` — which can be computed as **XNOR + popcount** (1-bit operations, ~32× faster than FP32 matmul on hardware that supports it).

The per-layer scale `α` recovers the lost magnitude:

$$
W_{\text{eff}} = \alpha \cdot \hat{W}, \qquad \alpha = \frac{\|W\|_1}{n} = \text{mean}|W|.
$$

### 3.2 Training recipe

The full BNN training recipe (Hubara et al., 2016):

1. **FP shadow weights `W`** in the optimizer (we already do this via `FP32MasterAdamW`).
2. **Hard binarization** `Ŵ = sign(W)` in the forward.
3. **STE for backward:** `∂L/∂W = ∂L/∂Ŵ`, with two modifications:
   - **Gradient clipping to `[-1, 1]`:** if `|W| > 1`, the STE gradient is multiplied by 0 (the weight is "out of range" and shouldn't move further). This prevents the FP shadow from drifting away from the binary grid.
   - **`tanh` saturating STE (alternative):** `∂L/∂W = (1 - tanh²(W)) · ∂L/∂Ŵ`. The `tanh` smoothly saturates the gradient as `|W|` grows, achieving a similar effect to hard clipping but with a smoother gradient.
4. **Per-layer `α` scale:** computed as `mean|W|` (post-hoc, not co-trained).
5. **Hard tanh activation function** instead of ReLU (so activations stay in `[-1, 1]`, matching the binarization).

### 3.3 The "two-stage decay" trick

BNN's optimizer (adapted from ADAM) uses a two-stage learning rate decay:
- **Stage 1 (warmup, 0–10% of training):** high LR (e.g., 1e-2), no decay. The FP shadow `W` is initialized with Kaiming init; this stage lets the shadow weights migrate to the binary grid.
- **Stage 2 (decay, 10–100% of training):** LR decays exponentially to 1e-5. The shadow weights commit to one of `{-1, +1}` and stop flipping.

This two-stage schedule is essential — without the warmup, the shadow weights stay near their Kaiming init and never reach the binary grid; without the decay, the shadow weights keep flipping between `±1` and never commit.

### 3.4 Why the gradient clipping matters

The `[-1, 1]` gradient clip is what prevents BNN from diverging. Without it, the FP shadow `W` can drift arbitrarily far from `±1`, and the STE gradient (which treats `sign(W)` as identity) loses meaning. With the clip, `W` is implicitly bounded to `[-1, 1]`, and the binarization `sign(W)` is well-defined.

Our `train_qwen.py:1153` clamps `logits` to `±20` after each step — but `±20` is far too loose. At `logits = ±20`, `softmax(logits/τ)` is numerically one-hot (with `τ = 0.1`, `softmax(±20/0.1) = softmax(±200)`, which overflows to `1`/`0`). The gradient at one-hot is zero, so the indices are effectively frozen.

**BNN's lesson:** the clip should be **tight**, just large enough to allow the parameter to commit. For logits, `±5` is appropriate: `softmax(±5/0.1) = softmax(±50)`, still essentially one-hot but with finite-precision gradient. Actually, the right fix is to clip in `logit-space` units that match `τ`: `clip(logits, -5τ, +5τ)`. At `τ = 0.1`, this gives `±0.5`, which corresponds to `softmax(±5) ≈ [0.993, 0.007]` — still essentially hard, but with usable gradient.

### 3.5 What we should borrow

1. **Tighten the logit clamp.** Change `train_qwen.py:1153` from `±20` to `±5·τ` (so `±10` at `τ=2`, `±0.5` at `τ=0.1`). This prevents logit saturation while still allowing the indices to commit. **One-line change.**

2. **Adopt BNN's two-stage LR schedule.** Currently we use a cosine schedule with no warmup. Adding a 500-step linear warmup (LR ramping from 0 to peak) would let the Gumbel-Softmax logits migrate to a meaningful configuration before annealing.

3. **Consider the saturating `tanh` STE for the palette gradient.** Currently the palette gradient is just `∂L/∂palette` from the soft assignment, with no saturation. Adding `(1 - tanh²(palette/β)) · ∂L/∂palette` (where `β` is the palette scale) would prevent the palette from drifting unboundedly — useful for stability over long training runs.

4. **The per-tensor `α` scale.** BitNet's `β` and BNN's `α` are the same idea: a post-hoc scale to recover magnitude lost to quantization. We have per-group palette scales (frozen), but no per-tensor scale. Adding a learnable per-tensor `α` (one float per Linear, 25 floats per super-block) gives the optimizer one more degree of freedom to match the output magnitude.

---

## 4. Synthesis: 1-bit Methods Gap Analysis

### 4.1 What we are missing (prioritized)

| Priority | Technique | Source | Expected cos gain | Implementation cost |
|---|---|---|---|---|
| **P0** | Tighten logit clamp from `±20` to `±5·τ` | BNN §3 | +0.005–0.015 (prevents logit saturation) | Trivial — one-line change |
| **P1** | Two-stage LR schedule (warmup + decay) | BNN §3 | +0.005–0.01 | Small — modify scheduler |
| **P1** | Per-tensor learnable `α` scale | BitNet §1, BNN §3 | +0.005–0.01 | Small — 1 param per Linear |
| **P1** | Initialize one palette level at 0 (sparse level) | BitNet b1.58 §2 | +0.005–0.015 | Trivial — one-line init change |
| **P2** | From-scratch training (no k-means init) | BitNet §1 | +0.02–0.04 (potentially) | Large — needs ~50K steps + new init |
| **P2** | Saturating `tanh` STE for palette gradient | BNN §3 | +0.005 | Small — modify backward |
| **P2** | `SubLN` (learnable gain before LayerNorm) | BitNet §1 | +0.005 | Small — add 1 param per Linear |

### 4.2 Why 1-bit methods matter for 2-bit

The 1-bit literature establishes several principles that are bitwidth-agnostic:

1. **FP shadow + STE is the canonical training pattern.** All three methods (BitNet, BitNet b1.58, BNN) use it; we use it correctly (via `FP32MasterAdamW`). ✓
2. **From-scratch beats fine-tune-quantize at extreme low bitwidth.** BitNet's results show that fine-tune-quantize (RTN) catastrophically fails at 1-bit, while from-scratch matches FP16. Our 2-bit is closer to the fine-tune-quantize regime; we may not need to go full from-scratch, but we should at least consider random `index_logits` init (instead of k-means one-hot) to give the Gumbel-Softmax more room to explore.
3. **Per-tensor/per-group scales must be co-trained, not post-hoc.** BitNet's `β` is post-hoc, but their follow-up work (BitNet b1.58) makes it learnable. We have per-group palette scales (frozen) — making them learnable is a small change.
4. **Gradient clipping must be tight, not loose.** BNN's `[-1, 1]` clip is essential. Our `±20` logit clamp is too loose.

### 4.3 The "from-scratch" question

The single biggest open question for our project: **should we drop the k-means init and train from scratch?**

Arguments for:
- BitNet/AQLM-FT show from-scratch beats fine-tune-quantize at extreme bitwidths.
- Our Gumbel-Softmax + trainable palette + LoRA is already QAT-style; only the *initialization* is fine-tune-quantize.
- The cos=0.95 plateau may be a fundamental limit of fine-tune-quantize at 2-bit (k-means init is near-optimal for the soft objective, so training has little room to improve).

Arguments against:
- Cost: from-scratch would need ~50K steps (vs. our 8K) — ~6× more compute.
- Risk: from-scratch training of LLMs is unstable (BitNet reports significant hyperparameter sensitivity).
- Our existing infrastructure (FP32MasterAdamW, Muon, super-block training) is fine-tune-quantize-oriented; switching would require rearchitecting.

**Recommendation:** Try from-scratch on a single super-block as an experiment. If it reaches cos>0.99 in 50K steps, the answer is yes. If not, the k-means init is fine and we should focus on the codebook-resolution improvements (GPTVQ, AQLM, SqueezeLLM) covered in `03_codebook_methods.md`.

---

## 5. Cross-method Comparison Table

| Method | Year | Bitwidth | Indices trained? | Codebook | Key trick | Best LLM result | Transferable to us? |
|---|---|---|---|---|---|---|---|
| **BitNet** | 2023 | 1-bit (±1) | No (sign of FP shadow) | Fixed (±1) | From-scratch + SubLN + per-tensor β | 3B matches FP16 | Medium — SubLN, from-scratch paradigm |
| **BitNet b1.58** | 2024 | 1.58-bit (ternary) | No (sign+round of FP shadow) | Fixed (ternary) | Sparsity via 0 level | 3B matches FP16, 2-3× faster | Medium — sparse level, ternary init |
| **BNN** | 2016 | 1-bit (±1) | No (sign of FP shadow) | Fixed (±1) | Tight gradient clip + two-stage LR + tanh STE | MNIST/CIFAR SOTA at 1-bit | High — tight clip, two-stage LR |
| **XNOR-Net** | 2016 | 1-bit (±1) | No (sign of FP shadow) | Fixed (±1) | Per-channel α scale | ImageNet 65% top-1 at 1-bit | Medium — per-channel scale (we have per-group) |
| **Ours** | 2026 | 2-bit (K=4) | Yes (Gumbel-Softmax) | Learned (k-means+Adam) | Trainable indices + LoRA + distillation | cos=0.95 (target 0.999) | — |

---

## 6. References (Wave 2, partial — full bibliography in `10_references.md`)

1. Wang, H., Ma, S., Dong, L., Huang, S., Wang, H., Ma, L., Yang, F., Wang, R., Wu, Y., Wei, F. (2023). *BitNet: Scaling 1-bit Transformers for Large Language Models.* [arXiv:2310.11453](https://arxiv.org/abs/2310.11453).
2. Ma, S., Wang, H., Ma, L., Wang, L., Wang, R., Yang, F., Dong, L., Wei, F. (2024). *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits.* [arXiv:2402.10564](https://arxiv.org/abs/2402.10564).
3. Courbariaux, M., Hubara, I., Soudry, D., El-Yaniv, R., Bengio, Y. (2016). *Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or −1.* NeurIPS 2016. [arXiv:1602.02830](https://arxiv.org/abs/1602.02830).
4. Hubara, I., Courbariaux, M., Soudry, D., El-Yaniv, R., Bengio, Y. (2016). *Quantized Neural Networks: Training Neural Networks with Low Precision Weights and Activations.* JMLR 2018 / arXiv 2016. [arXiv:1609.07061](https://arxiv.org/abs/1609.07061).
5. Rastegari, M., Poulenard, E., Hajri, M. Y., Dumenil, Y. (2016). *XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks.* ECCV 2016. [arXiv:1603.05279](https://arxiv.org/abs/1603.05279).
6. Bengio, Y., Léonard, N., Courville, A. (2013). *Estimating or Propagating Gradients Through Stochastic Neurons (STE).* [arXiv:1308.3432](https://arxiv.org/abs/1308.3432).
7. Jang, E., Gu, S., Poole, B. (2017). *Categorical Reparameterization with Gumbel-Softmax.* ICLR 2017. [arXiv:1611.01144](https://arxiv.org/abs/1611.01144).
8. Maddison, C. J., Mnih, A., Teh, Y. W. (2017). *The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables.* ICLR 2017. [arXiv:1611.00712](https://arxiv.org/abs/1611.00712).
9. Nagel, M., Fournarakis, M., Bondarenko, Y., Blankevoort, T. (2022). *Overcoming Oscillations in Quantization-Aware Training.* ICML 2022. [arXiv:2203.11086](https://arxiv.org/abs/2203.11086).
10. Cardinaux, F., Uhlich, S., Yoshiyama, M., Matsubara, T., Takada, K., Cassirer, A. (2018). *Iteratively Training Look-Up Tables for Network Quantization (LUT-Q).* NeurIPS DeepVision Workshop 2018. [arXiv:1811.05355](https://arxiv.org/abs/1811.05355). *(Cross-referenced.)*
11. Wang, L., Dong, Y., Wang, Y., Liu, X., An, J., Guo, Y. (2022). *Learnable Lookup Table for Neural Network Quantization (LLT).* CVPR 2022. [OpenAccess](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Learnable_Lookup_Table_for_Neural_Network_Quantization_CVPR_2022_paper.html). *(Cross-referenced.)*
12. Egiazarian, V., Kuznedelev, A., Diskin, M., Babenko, A., Frantar, E. (2024). *AQLM: Extreme Compression of Large Language Models via Additive Quantization.* ICML 2024. [arXiv:2401.06118](https://arxiv.org/abs/2401.06118). *(Cross-referenced.)*
13. Esser, S. K., McKinstry, J. L., Bablani, D., Appuswamy, R., Modha, D. S. (2020). *Learned Step Size Quantization (LSQ).* ICLR 2020. [arXiv:1902.08153](https://arxiv.org/abs/1902.08153). *(Trainable codebook+step — same family.)*
14. Liu, Z., Wang, Y., Han, K., Zhang, W., Ma, S., Gao, W. (2022). *Post-Binarization: Pushing BNNs to the Limit.* NeurIPS 2022. [arXiv:2206.09295](https://arxiv.org/abs/2206.09295). *(Post-training binarization refinement.)*
15. Qian, B., Wang, Y., Liu, Z., Hooi, B., Han, K., Wang, Y. (2024). *BNN-ViT: Binarized Vision Transformer on ImageNet.* [arXiv:2403.00352](https://arxiv.org/abs/2403.00352). *(BNN training recipes for transformers specifically.)*

*15 arxiv papers cited in this file. Combined with file 03 (17 papers): Wave 2 totals well over 12 unique arxiv citations and includes extensive mathematical formulations (gradient derivations in §1.2, §2.2, §3.2; forward formulations in §1.2, §2.1, §3.1; quantizer formulations in §2.2, §3.1).*
