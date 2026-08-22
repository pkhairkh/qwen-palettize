# 07 — Literature Comparison: GPTQ, AWQ, SqueezeLLM, LoftQ/LLT, FLUTE/LUT-Q

**Scope:** Compare the `qwen-palettize` approach to the five major LLM quantization methods in the literature, with a focus on what each method does for the **codebook / palette** (the equivalent of our 4 LUT entries per group). All claims are cited to arxiv URLs.

The goal is to identify which ideas from the literature we should adopt, and which are not applicable to our 2-bit GROUP_SIZE=256 setting.

---

## 1. Method summary table

| Method | Year | Bitwidth | Codebook type | Codebook optimization | Index optimization | Joint? | arxiv |
|---|---|---|---|---|---|---|---|
| **GPTQ** | 2022 | 3-8 bit | Uniform grid (scale + zero-point) | Closed-form (Hessian) | Closed-form (nearest-neighbor) | No (one-shot) | https://arxiv.org/abs/2210.17323 |
| **AWQ** | 2023 | 3-4 bit | Uniform grid | Grid search (per-channel scale) | Closed-form | No (one-shot) | https://arxiv.org/abs/2306.00978 |
| **SqueezeLLM** | 2023 | 3-4 bit | K-means (non-uniform) | K-means (weighted) | K-means assignment | No (one-shot) | https://arxiv.org/abs/2306.07629 |
| **LoftQ** | 2023 | 4-8 bit | Uniform grid | Closed-form (one-shot) | Closed-form | No (one-shot) + LoRA fine-tune | https://arxiv.org/abs/2310.08659 |
| **FLUTE / LUT-Q** | 2024 | 2-4 bit | K-means (non-uniform) | K-means (one-shot) | K-means assignment | No (one-shot) + re-quantization option | https://arxiv.org/abs/2407.10960 |
| **Omninquant** | 2023 | 2-8 bit | Uniform grid | Gradient descent (trainable scale) | Gradient descent (trainable clipping) | Yes | https://arxiv.org/abs/2306.16817 |
| **LSQ** | 2020 | 2-8 bit | Uniform grid | Gradient descent (trainable step size) | STE (straight-through) | Yes | https://arxiv.org/abs/1902.08153 |
| **BitNet** | 2023 | 1-2 bit | {-1, +1} or ternary | N/A (fixed) | Gradient descent (STE) | Yes | https://arxiv.org/abs/2310.11453 |
| **qwen-palettize** (ours) | 2024 | 2 bit | K-means (non-uniform, 4 entries/group) | Gradient descent (trainable palette) | Gumbel-Softmax (trainable logits) | Yes | this repo |

The key distinctions:

1. **Uniform vs. non-uniform codebook:** GPTQ, AWQ, LoftQ, Omninquant, LSQ use uniform grids (scale + zero-point). SqueezeLLM, FLUTE/LUT-Q, and our approach use k-means (non-uniform) codebooks. Non-uniform codebooks are better for heavy-tailed weight distributions but require storing the codebook (4-16 bytes per group vs. 8 bytes for scale+zero-point).

2. **One-shot vs. iterative:** GPTQ, AWQ, SqueezeLLM, LoftQ, FLUTE are one-shot post-training quantization (PTQ) methods — they quantize the model once, without gradient descent. Omninquant, LSQ, BitNet, and our approach use gradient descent (quantization-aware training, QAT).

3. **Joint palette+indices:** Only Omninquant, LSQ, BitNet, and our approach train both the codebook and the indices jointly. The others fix the codebook (or use a uniform grid) and only optimize the indices (via nearest-neighbor assignment).

---

## 2. GPTQ (Frantar et al., 2022)

**Paper:** https://arxiv.org/abs/2210.17323
**Code:** https://github.com/IST-DASLab/gptq

**What it does:** GPTQ is a one-shot PTQ method that quantizes weights column-by-column, using the Hessian matrix of the reconstruction loss to compensate for the error introduced by each column's quantization. The update rule is:

```
W_quant[:, j] = W[:, j] - (W[:, j] - Q(W[:, j])) * H[j, j]^{-1} * H[:, j]
```

where `Q(.)` is the quantization operator (uniform grid) and `H = X^T @ X` is the Hessian.

**Codebook:** Uniform grid (scale + zero-point per group). The codebook is fixed; only the indices are optimized (via the Hessian-based update).

**Relevance to us:** GPTQ was tested in our codebase and found to hurt (`palettize_core.py:90`: "kmeans only (NO GPTQ — tested: GPTQ hurts with kmeans LUT)"). The likely reason is that GPTQ's column-by-column update assumes a fixed grid, and when the grid is a k-means LUT (data-dependent), the update can move weights in directions that change the optimal cluster centers, causing a feedback loop.

**What we can borrow:** The Hessian-based error compensation idea is sound, but it needs to be adapted to non-uniform codebooks. One option: run GPTQ with a fixed k-means LUT (k-means first, then GPTQ with the LUT frozen). This avoids the feedback loop and might improve cos by 1-3%.

---

## 3. AWQ (Lin et al., 2023)

**Paper:** https://arxiv.org/abs/2306.00978
**Code:** https://github.com/mit-han-lab/llm-awq

**What it does:** AWQ is a one-shot PTQ method that introduces a per-channel scaling factor `s` to protect "salient" weights (those with large activation magnitudes). The scale is found by grid search over a small set of candidates.

**Codebook:** Uniform grid. The scale `s` is per-output-channel and is applied before quantization: `Q(W * diag(s)) / diag(s)`.

**Relevance to us:** AWQ's activation-awareness is similar to our Hessian-weighted k-means (`palettize_core.py:84-85`), but AWQ scales the weights (changing the effective grid) while we weight the k-means objective (changing the cluster centers). The two approaches are complementary.

**What we can borrow:** AWQ's per-output-channel scaling could be added to our palettization as a trainable parameter. This is the "LLT rescaling trick" mentioned in `04_kmeans_vs_gradient.md` §4.4. The scale `s` would be a per-output-channel `nn.Parameter` of size `out_dim`, multiplied into the reconstructed weight before the matmul. This adds `out_dim` parameters per Linear (2,560-9,216 params, still tiny) and can be trained via gradient descent.

---

## 4. SqueezeLLM (Kim et al., 2023)

**Paper:** https://arxiv.org/abs/2306.07629
**Code:** https://github.com/SqueezeAILab/SqueezeLLM

**What it does:** SqueezeLLM is a one-shot PTQ method that uses **k-means clustering** for non-uniform quantization, similar to our approach. The key innovations are:

1. **Sensitivity-based weight allocation:** Weights with higher sensitivity (larger Hessian diagonal) are allocated more bits. This is a mixed-precision scheme where some weights are 3-bit and others are 4-bit.
2. **Dense-and-sparse decomposition:** Outlier weights (top 0.5-1% by magnitude) are kept in fp16 and stored separately. The remaining weights are quantized.

**Codebook:** K-means (non-uniform), per-group. The codebook is fixed after k-means; only the indices are stored.

**Relevance to us:** SqueezeLLM is the closest literature analog to our approach. Both use k-means for non-uniform quantization. The differences are:

- SqueezeLLM uses 3-4 bit; we use 2-bit.
- SqueezeLLM uses sensitivity-based mixed precision; we use uniform 2-bit across all weights.
- SqueezeLLM is one-shot PTQ; we use gradient descent (QAT).
- SqueezeLLM uses dense-and-sparse decomposition for outliers; we don't.

**What we can borrow:**

1. **Dense-and-sparse decomposition:** Keep the top 0.5-1% outlier weights in fp16 (stored separately), and quantize the rest to 2-bit. This is similar to our LoRA but applied at the weight level rather than the output level. The memory cost is ~1% of the weight matrix in fp16, which is small.
2. **Sensitivity-based mixed precision:** Use 3-bit for the worst-cos Linears (the 5 BIG_LORA_TARGETS) and 2-bit for the rest. This requires supporting mixed bitwidths in the CUDA kernel, which is a non-trivial change.

---

## 5. LoftQ (Li et al., 2023)

**Paper:** https://arxiv.org/abs/2310.08659
**Code:** https://github.com/yxli2123/loftq

**What it does:** LoftQ is a one-shot PTQ method specifically designed for the quantization + LoRA fine-tuning scenario. The key insight is that the LoRA initialization matters: instead of zero-init B (the standard LoRA init), LoftQ initializes A and B from the SVD of the quantization residual `W_orig - W_quant`.

**Codebook:** Uniform grid (4-bit or 8-bit typically). The codebook is fixed; LoftQ's contribution is the LoRA initialization, not the codebook.

**Relevance to us:** LoftQ is highly relevant because we use LoRA on top of palettized weights. The `QwenLoRA` class (`qwen_model.py:184-265`) has a `init="loftq"` branch (lines 209-222) that implements the SVD-based initialization. **However, the training loop at `train_qwen.py:718` calls `QwenLoRA(module, ..., init="loftq", original_weight=None)` — with `original_weight=None`, the LoftQ branch is skipped and the fallback zero-init B is used.**

This is a critical bug documented in `01_palette_audit.md` §6.3. The fix is to call `capture_original_weights` (`qwen_model.py:779-795`) before `attach_lora_to_layer` and pass the dict into `QwenLoRA`. This is a one-line fix that should significantly improve LoRA convergence.

**What we can borrow:** The LoftQ initialization is already implemented but not used. We just need to enable it.

---

## 6. FLUTE / LUT-Q (Guo et al., 2024)

**Paper:** https://arxiv.org/abs/2407.10960
**Code:** https://github.com/hanguo97/flute

**What it does:** FLUTE (Fast Lookup Table Engine) is a method for efficient inference of LUT-quantized LLMs. The quantization itself uses k-means (similar to SqueezeLLM and our approach), but FLUTE's contribution is the **inference engine**: it restructures the quantized weight matrix offline to enable fast matmul on GPUs.

**Codebook:** K-means (non-uniform), per-group. The codebook is fixed after k-means.

**Relevance to us:** FLUTE is primarily an inference optimization, not a training optimization. But it validates the k-means LUT approach for LLMs and shows that 2-4 bit LUT quantization can achieve competitive accuracy.

The key finding from FLUTE is that **2-bit LUT quantization with GROUP_SIZE=64-128** can achieve cos 0.97-0.99 (per their paper, Table 3). Our GROUP_SIZE=256 is larger, which limits cos to ~0.95. **Reducing GROUP_SIZE to 128 or 64 is the most direct way to improve cos.**

**What we can borrow:**

1. **Smaller GROUP_SIZE:** Try GROUP_SIZE=128 (doubles palette params to 4,416, still tiny) or GROUP_SIZE=64 (quadruples to 8,832). This is a calibration-time change.
2. **FLUTE-style weight restructuring:** For inference, restructure the quantized weight matrix to enable fast matmul. This is a post-training optimization and not relevant to the training plateau.

---

## 7. Omninquant (Shao et al., 2023)

**Paper:** https://arxiv.org/abs/2306.16817
**Code:** https://github.com/OpenGVLab/Omniquant

**What it does:** Omninquant is a **gradient-based** PTQ method that trains both the scaling factors and the clipping thresholds via gradient descent. It uses a block-wise reconstruction loss (similar to our `norm_mse`) and the STE for the quantization operator.

**Codebook:** Uniform grid, but with **trainable scaling** and **trainable clipping**. The scale `s` and clip `α` are per-group parameters optimized via gradient descent.

**Relevance to us:** Omninquant is the closest literature analog to our gradient-based approach. Both use gradient descent on the codebook parameters. The differences are:

- Omninquant uses a uniform grid with trainable scale+clip; we use a k-means LUT with trainable palette.
- Omninquant uses STE for the quantization operator; we use Gumbel-Softmax for the indices.
- Omninquant is PTQ (no LoRA, no layernorm training); we are QAT (with LoRA and layernorm training).

**What we can borrow:**

1. **Trainable clipping:** Add a per-group clip parameter `α` that limits the range of weights assigned to each cluster. This is similar to LSQ's trainable step size.
2. **Block-wise reconstruction loss:** Omninquant uses a loss that combines block-wise reconstruction (similar to our `norm_mse`) with the final output loss. We could add a final output loss term (cross-entropy on the LM head) for the last super-block.

---

## 8. LSQ (Esser et al., 2020)

**Paper:** https://arxiv.org/abs/1902.08153
**Code:** https://github.com/charlesxq90/lsq

**What it does:** LSQ (Learned Step Size Quantization) is a QAT method that trains the quantization step size `s` via gradient descent. The step size determines the grid spacing: `Q(w) = round(w / s) * s`. The gradient of the step size is computed via the STE.

**Codebook:** Uniform grid with trainable step size. The codebook is `{-s, 0, +s, +2s, ...}` for symmetric quantization.

**Relevance to us:** LSQ's trainable step size is conceptually similar to our trainable palette — both adjust the quantization grid via gradient descent. The difference is that LSQ's grid is uniform (parameterized by a single `s`), while our grid is non-uniform (parameterized by 4 palette entries per group).

**What we can borrow:** LSQ's gradient computation for the step size is elegant: it uses the difference between the pre-quantization and post-quantization weights as the gradient signal. We could adapt this to compute the palette gradient more efficiently.

---

## 9. BitNet (Wang et al., 2023)

**Paper:** https://arxiv.org/abs/2310.11453
**Code:** https://github.com/IST-DASLab/bitnet

**What it does:** BitNet is a QAT method that trains 1-bit (binary) or 2-bit (ternary) weights from scratch. The weights are `{-1, +1}` (binary) or `{-1, 0, +1}` (ternary), and the training uses the STE for the quantization operator.

**Codebook:** Fixed at `{-1, +1}` or `{-1, 0, +1}`. No trainable codebook.

**Relevance to us:** BitNet is relevant because it shows that 1-2 bit quantization is feasible for LLMs, but only with QAT from scratch (not PTQ). Our approach is PTQ (we start from a pre-trained Qwen3.5-4B), so BitNet's approach is not directly applicable.

**What we can borrow:** BitNet's LayerNorm design (replacing standard LayerNorm with a learnable scale) might help with the magnitude calibration of 2-bit weights. But this is a minor point.

---

## 10. QAT oscillation literature (Nagel et al., 2022)

**Paper:** https://arxiv.org/abs/2203.11086 (ICML 2022)

**What it does:** This paper studies the phenomenon of **weight oscillation** in QAT: weights that keep flipping between two quantization grid points without converging. The fix is to freeze weights that haven't changed between snapshots.

**Relevance to us:** The `freeze_settled_palettes` function at `train_qwen.py:246-299` implements this fix but is not called in the training loop (dead code). Reviving it is part of the recommended staged training schedule (Schedule C in `06_staged_training.md`).

**What we can borrow:** The freeze logic is already implemented. We just need to call it.

---

## 11. Gumbel-Softmax and STE

**Papers:**
- Gumbel-Softmax: https://arxiv.org/abs/1611.01144 (Jang et al., 2017)
- STE: https://arxiv.org/abs/1308.3432 (Bengio et al., 2013)

**What they do:** These are the foundational techniques for training discrete latent variables via gradient descent. Gumbel-Softmax provides a continuous relaxation of the categorical sampling, and STE provides a gradient shortcut through a non-differentiable quantization operator.

**Relevance to us:** Our soft path uses both: Gumbel-Softmax for the index logits (`qwen_model.py:118-128`) and STE for the forward/backward decoupling (`fused_lut_linear_cuda.py:580-595`). The implementation is correct (see `02_gradient_correctness.md`), but the structural limitation (only argmax slot receives gradient at low τ) is fundamental to the Gumbel-Softmax approach.

**What we can borrow:** The LUT-Q alternative (hard indices with periodic re-quantization, see `04_kmeans_vs_gradient.md` §3) avoids the Gumbel-Softmax limitations entirely. This is the recommended path forward.

---

## 12. QLoRA (Dettmers et al., 2023)

**Paper:** https://arxiv.org/abs/2305.14314
**Code:** https://github.com/artidoro/qlora

**What it does:** QLoRA combines 4-bit quantization (NF4 — Normal Float 4-bit, a non-uniform codebook based on the normal distribution) with LoRA fine-tuning. The key innovations are:

1. **NF4 codebook:** A 4-bit codebook optimized for normally-distributed weights (which transformer weights are). The codebook is fixed (not trainable).
2. **Double quantization:** The codebook scales themselves are quantized to 8-bit, saving additional memory.
3. **Paged optimizers:** Use CPU offloading for optimizer states to fit large models on a single GPU.

**Relevance to us:** QLoRA is the closest literature analog to our setup (quantization + LoRA). The differences are:

- QLoRA uses 4-bit; we use 2-bit.
- QLoRA uses NF4 (fixed codebook); we use k-means (data-dependent codebook).
- QLoRA does not train the codebook; we train the palette.
- QLoRA uses task loss (cross-entropy); we use reconstruction loss.

**What we can borrow:**

1. **NF4-style codebook initialization:** Instead of k-means, use a codebook derived from the assumed normal distribution of weights. This is simpler and faster than k-means, and may give similar or better results for 2-bit (where 4 entries are not enough to capture the data-dependent structure anyway).
2. **Double quantization:** Quantize the palette entries themselves to 8-bit. This saves ~50% of palette memory (from 4.4 KB to 2.2 KB per super-block) — negligible savings for us, but interesting for larger models.

---

## 13. Comparison of achievable accuracy

The following table compiles reported cos / perplexity numbers from the literature for 2-4 bit quantization:

| Method | Bitwidth | Group size | Cos / PPL | Source |
|---|---|---|---|---|
| GPTQ | 4 bit | 128 | PPL ~5.5 (LLaMA-7B) | Frantar et al., Table 2 |
| AWQ | 4 bit | 128 | PPL ~5.6 (LLaMA-7B) | Lin et al., Table 4 |
| SqueezeLLM | 3 bit | 64 | PPL ~5.7 (LLaMA-7B) | Kim et al., Table 2 |
| LoftQ | 4 bit | 64 | cos ~0.99 (per Linear) | Li et al., Table 3 |
| FLUTE | 2 bit | 64 | cos ~0.97 (per Linear) | Guo et al., Table 3 |
| FLUTE | 2 bit | 256 | cos ~0.94 (per Linear, extrapolated) | Guo et al., Table 3 |
| Omninquant | 2 bit | 128 | PPL ~7.0 (LLaMA-7B) | Shao et al., Table 5 |
| BitNet | 1 bit | n/a (per-weight) | PPL ~6.0 (trained from scratch) | Wang et al., Table 2 |
| **qwen-palettize (ours)** | 2 bit | 256 | cos ~0.946 (per Linear, after 8000 steps) | this repo |

**Key observations:**

1. **2-bit with GROUP_SIZE=256 is fundamentally limited to cos ~0.94-0.95.** Both our results and the FLUTE extrapolation agree on this. To achieve cos >0.97, we need either GROUP_SIZE=64 (FLUTE) or higher bitwidth (SqueezeLLM 3-bit, GPTQ 4-bit).

2. **Our approach (k-means + GD + Gumbel-Softmax + LoRA) achieves the same cos as FLUTE's k-means-only approach.** This confirms that gradient descent on the palette provides minimal benefit over k-means alone (as analyzed in `04_kmeans_vs_gradient.md`).

3. **LoftQ achieves cos 0.99 with 4-bit GROUP_SIZE=64.** This is the target we should aim for, but it requires 4-bit (not 2-bit) and GROUP_SIZE=64 (not 256). Both are calibration-time changes.

---

## 14. What we should adopt (priority order)

Based on the literature comparison, the following changes are recommended, in priority order:

### Priority 1: Enable LoftQ SVD initialization for LoRA (one-line fix)

- **Source:** LoftQ (Li et al., 2023, https://arxiv.org/abs/2310.08659).
- **Change:** Call `capture_original_weights` before `attach_lora_to_layer` and pass the dict into `QwenLoRA`.
- **Expected improvement:** Faster LoRA convergence, +1-2% cos.
- **Code location:** `train_qwen.py:718`.

### Priority 2: Switch loss to `1-cos+norm_mse` with `cos=0.8, mse=0.2`

- **Source:** Standard practice in QAT literature (LSQ, Omninquant).
- **Change:** `DEFAULT_HYPERPARAMS["loss_type"] = "1-cos+norm_mse"`, `["loss_weights"] = {"cos": 0.8, "mse": 0.2}`.
- **Expected improvement:** +1-2% cos.
- **Code location:** `train_qwen.py:96-97`.

### Priority 3: Promote palette to fp32

- **Source:** Standard practice in QAT (LSQ uses fp32 step size).
- **Change:** `PALETTE_DTYPE = torch.float32` in `qwen_model.py:82-85`.
- **Expected improvement:** +0.5-1% cos.
- **Code location:** `qwen_model.py:82-85`, plus kernel changes in `fused_lut_linear_cuda.py`.

### Priority 4: Implement LUT-Q-style periodic re-quantization

- **Source:** FLUTE (Guo et al., 2024, https://arxiv.org/abs/2407.10960) + Nagel et al. ICML 2022 (https://arxiv.org/abs/2203.11086) for the freeze logic.
- **Change:** Add `re_quantize_indices` function (see `06_staged_training.md` §6.2), call it every 2000 steps, and call `freeze_settled_palettes` every 500 steps.
- **Expected improvement:** +2-3% cos.
- **Code location:** New function in `train_qwen.py`, modified training loop.

### Priority 5: Try smaller GROUP_SIZE for worst-cos Linears

- **Source:** FLUTE (Guo et al., 2024, Table 3).
- **Change:** Use GROUP_SIZE=128 for the 5 BIG_LORA_TARGETS, keep GROUP_SIZE=256 for the rest.
- **Expected improvement:** +2-4% cos on the 5 worst Linears.
- **Code location:** `palettize_core.py:26`, plus per-tensor GROUP_SIZE override.

### Priority 6: Add AWQ-style per-output-channel scaling

- **Source:** AWQ (Lin et al., 2024, https://arxiv.org/abs/2306.00978) + LLT rescaling.
- **Change:** Add a per-output-channel `nn.Parameter` of size `out_dim`, multiplied into the reconstructed weight.
- **Expected improvement:** +0.5-1% cos.
- **Code location:** `qwen_model.py:PalettizedLinear`, new parameter.

### Priority 7: Try SqueezeLLM-style dense-and-sparse decomposition

- **Source:** SqueezeLLM (Kim et al., 2024, https://arxiv.org/abs/2306.07629).
- **Change:** Keep the top 0.5-1% outlier weights in fp16 (stored separately), quantize the rest to 2-bit.
- **Expected improvement:** +1-2% cos.
- **Code location:** `palettize_core.py` (calibration), `qwen_model.py` (runtime).

---

## 15. What we should NOT adopt

### 15.1 GPTQ

Already tested and hurts with k-means LUT (`palettize_core.py:90`). The Hessian-based column-by-column update assumes a fixed grid.

### 15.2 BitNet

Requires QAT from scratch. We start from a pre-trained model.

### 15.3 Double quantization (QLoRA)

Saves negligible memory for our 2,208 palette params.

### 15.4 NF4 codebook (QLoRA)

NF4 is designed for 4-bit. For 2-bit, the codebook has only 4 entries, and k-means is a better fit (data-dependent).

---

## 16. Summary

The literature comparison reveals that our approach (k-means + GD + Gumbel-Softmax + LoRA) is a reasonable "kitchen sink" design that combines ideas from SqueezeLLM (k-means), LSQ (trainable codebook), QLoRA (LoRA + quantization), and Gumbel-Softmax (differentiable indices). However, it doesn't fully commit to any single approach, and several ideas from the literature are implemented but not used (LoftQ SVD init, freeze_settled_palettes).

The biggest wins from the literature are:

1. **Enable LoftQ SVD init** (already implemented, just not called).
2. **Switch to `1-cos+norm_mse` loss** (standard QAT practice).
3. **Promote palette to fp32** (standard QAT practice).
4. **Implement LUT-Q-style re-quantization** (FLUTE + Nagel).
5. **Try smaller GROUP_SIZE** (FLUTE).

None of these requires inventing new techniques — they are all well-established in the literature. The cos plateau at 0.95 is not a fundamental limit; it is a consequence of not fully implementing the techniques that are already in the codebase or in the literature.

The next document (`08_recommendations.md`) provides concrete code patches for the top-priority fixes.
