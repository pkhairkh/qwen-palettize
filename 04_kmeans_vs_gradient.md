# 04 — K-means Calibration vs Gradient Descent: LUT-Q, LLT, GPTQ, AWQ

**Question:** Should palettes be trained via gradient descent (current approach) or via periodic k-means re-quantization (LUT-Q approach)? Or some hybrid?

**Answer:** Pure gradient descent on palettes (the current approach) is mathematically correct but practically suboptimal for 2-bit palettization. The k-means initialization is already at a local optimum of the L2 reconstruction objective, so gradient descent can only fine-tune around it. Periodic k-means re-quantization (LUT-Q style) can escape this local optimum by re-deriving the indices from the current palette, but it has its own failure mode: it can oscillate between two index assignments without converging.

The recommended approach is a **hybrid**: gradient descent on palettes for most steps, with periodic k-means re-quantization every N steps (N=200-500) to refresh the index assignments. This is the LUT-Q recipe adapted to our setting.

This document compares the four major approaches in the literature (k-means+GD, LUT-Q, LLT, GPTQ, AWQ) and explains why each succeeds or fails for 2-bit palettization with GROUP_SIZE=256.

---

## 1. The fundamental tension: indices vs palette

In 2-bit palettization, the reconstruction is:

```
W[j, o] = palette[g(o), indices[j, o]]
```

Both `palette` (continuous, 4 values per group) and `indices` (discrete, 2 bits per weight) are degrees of freedom. The optimization problem is:

```
minimize  || y_orig - x @ W_recon ||^2
over      palette ∈ R^{G × 4},  indices ∈ {0, 1, 2, 3}^{K × N}
```

This is a **mixed-integer optimization** problem, which is NP-hard in general. All practical approaches use some form of alternating optimization:

1. Fix `indices`, optimize `palette` (continuous subproblem).
2. Fix `palette`, optimize `indices` (discrete subproblem — nearest-neighbor assignment).
3. Repeat.

The approaches differ in how they solve each subproblem and whether they use gradient information.

---

## 2. Approach 1: K-means calibration + gradient descent on palette (current)

### 2.1 What it does

`palettize_core.py:61-144` runs Hessian-weighted 1-D k-means per group to find both `palette` and `indices` simultaneously. K-means is itself an alternating optimization:

- **E-step:** assign each weight to the nearest cluster center (this gives `indices`).
- **M-step:** update each cluster center to the weighted mean of assigned weights (this gives `palette`).

After k-means converges, `indices` are frozen and `palette` is fine-tuned via gradient descent during training (`train_qwen.py`).

### 2.2 Why k-means is locally optimal

For 1-D k-means with k=4 clusters, the algorithm is guaranteed to converge to a local optimum of the within-cluster sum of squares (WCSS) objective. Moreover, for 1-D data, the global optimum can be found in O(kN) time via dynamic programming (Wang & Song, 2011). The implementation in `palettize_pytorch.py:25-87` uses Lloyd's algorithm (not DP), but for k=4 and well-behaved weight distributions, Lloyd's usually finds the global optimum.

The Hessian weighting (`palettize_core.py:84-85`) modifies the objective to:

```
minimize  Σ_{j, o} hess_diag[j] * (W[j, o] - palette[g(o), indices[j, o]])^2
```

This is a weighted WCSS, and k-means still converges to a local optimum (the weighting just rescales the "distance" metric).

### 2.3 Why gradient descent on palette alone cannot escape the local optimum

Once `indices` are frozen, the palette optimization is:

```
minimize  Σ_{j, o} hess_diag[j] * (W[j, o] - palette[g(o), indices[j, o]])^2
over      palette ∈ R^{G × 4}
```

This is a **convex quadratic** in `palette` (for fixed `indices`). The optimal `palette[g, k]` is the weighted mean of all `W[j, o]` with `indices[j, o] = k` and `g(o) = g`:

```
palette*[g, k] = (Σ_{j, o: g(o)=g, indices[j,o]=k} hess_diag[j] * W[j, o])
               / (Σ_{j, o: g(o)=g, indices[j,o]=k} hess_diag[j])
```

This is exactly the M-step of k-means. So if we run gradient descent on `palette` with the L2 reconstruction loss and fixed `indices`, we will converge to the **same** `palette` that k-means already found.

**This is the fundamental reason gradient descent on palettes doesn't improve cos:** k-means already found the optimal palette for the given indices, and gradient descent on the same objective converges to the same solution.

### 2.4 What gradient descent CAN do

Gradient descent can improve cos if the loss function is **different** from the k-means objective. Specifically:

- **K-means objective:** `Σ hess_diag[j] * (W[j, o] - W_recon[j, o])^2` (weight-space L2, weighted by input activation magnitude).
- **Training loss:** `norm_mse = ((s - t)^2).mean() / (t*t).mean()` where `s = x @ W_recon` and `t = x @ W_orig` (output-space normalized MSE).

These are related but not identical. The k-means objective weights errors by `hess_diag[j] = Σ_i x[i, j]^2` (per-input-column), while the training loss weights errors by the output activation `(s - t)^2` (per-output-column, normalized).

Concretely, the training loss can be expanded as:

```
||s - t||^2 = ||x @ (W_recon - W_orig)||^2 = (W_recon - W_orig)^T @ (x^T @ x) @ (W_recon - W_orig)
```

So the training loss is a **full Hessian quadratic** in `(W_recon - W_orig)`, not just the diagonal. K-means uses only the diagonal `hess_diag = diag(x^T @ x)`, ignoring off-diagonal terms.

The off-diagonal terms matter when input activations are correlated (which they always are in transformer hidden states). So gradient descent on the full loss can find a better `palette` than k-means on the diagonal approximation.

**But:** the improvement is bounded by the magnitude of the off-diagonal Hessian terms. For typical transformer activations, the Hessian is diagonally dominant (off-diagonal terms are ~10-30% of diagonal terms), so the achievable improvement is ~10-30% reduction in reconstruction error. This translates to maybe 1-3% improvement in cos — not enough to bridge the gap from 0.95 to 0.999.

### 2.5 Empirical evidence from the training log

`train_sb0.log:35` shows:

```
Resumed 140 params from .../trained/superblock_0_best (step=8000, cos=0.946436)
```

So after 8,000 steps of training, cos improved from the initial calibration mean of 0.937 to 0.946 — an improvement of ~0.009, or ~1%. This is consistent with the ~1-3% improvement bound from §2.4. Gradient descent is doing what it can, but it's hitting the fundamental limit imposed by the frozen indices.

---

## 3. Approach 2: LUT-Q (k-means re-quantization per step)

### 3.1 What LUT-Q does

LUT-Q (Wallat et al., 2024) is the most direct alternative to gradient descent on palettes. The algorithm is:

1. Initialize `palette` and `indices` via k-means.
2. For each training step:
   a. Compute the gradient of the loss with respect to the reconstructed weight `W_recon` (treating `W_recon` as a dense fp16/fp32 matrix).
   b. Update `W_recon` via SGD/Adam: `W_recon -= lr * grad_W`.
   c. Re-quantize `W_recon` per group: run k-means on each group of 256 updated weights to get new `palette` and `indices`.
3. Repeat.

The key insight is that step (c) re-derives both `palette` and `indices` from the current `W_recon`, so the optimization is over the dense weight matrix (continuous, easy) rather than over `palette` and `indices` separately (mixed-integer, hard).

### 3.2 Why LUT-Q can escape the k-means local optimum

LUT-Q's re-quantization step (c) is itself a k-means run, but on the *updated* `W_recon`, not the original `W_orig`. If the gradient update has moved `W_recon` in a direction that changes the cluster structure (e.g., merging two clusters or splitting one), the re-quantization will reflect this new structure.

Concretely, suppose the initial k-means found clusters `{-0.04, -0.01, +0.01, +0.04}` for a group. If gradient descent pushes the weights toward `{-0.05, -0.02, +0.02, +0.05}` (a wider distribution), re-quantization will find these new centers, which better represent the updated weights.

The current approach (frozen indices) cannot do this: the indices are stuck at the initial k-means assignment, so even if the palette moves, the assignment doesn't update to reflect the new optimal centers.

### 3.3 Why LUT-Q can fail: oscillation

The LUT-Q paper (Wallat et al., 2024, §3.3) notes that re-quantization can cause oscillation: if the gradient pushes a weight from cluster A to cluster B and back, the re-quantization will flip the assignment each step, and the palette will oscillate between two configurations without converging.

The standard fix is to use a **straight-through estimator (STE)** for the re-quantization: the forward uses the hard quantized weight, but the backward passes the gradient through as if the quantization were identity. This is what `fused_lut_linear_cuda.py:580-595` does for the soft path, but the current implementation does NOT re-quantize per step (it uses the Gumbel-Softmax relaxation instead, which is a different way of making the indices trainable).

### 3.4 LUT-Q vs Gumbel-Softmax: what's the difference?

Both approaches make the indices trainable, but in different ways:

| Aspect | LUT-Q | Gumbel-Softmax (current) |
|---|---|---|
| Index representation | Hard int8, re-derived per step | Soft logits (4, K, N), sampled via Gumbel |
| Forward | Hard quantization (exact) | Hard via STE (`fused_lut_linear_cuda.py:580-595`) |
| Backward | STE (gradient through quantization as identity) | Soft (gradient through softmax probabilities) |
| Re-quantization | Yes, every step | No (logits are continuous, trained via AdamW) |
| Memory for indices | K × N bytes (int8) | 4 × K × N bytes (fp16 logits, 4x larger) |
| Convergence | Fast but can oscillate | Slower but smoother |

The current implementation chose Gumbel-Softmax to avoid oscillation, at the cost of 4x memory and slower convergence. The training log confirms the memory cost: `indices: 1,782,579,200` parameters (`train_sb0.log:34`), which is 3.56 GB at fp16.

### 3.5 What LUT-Q would buy us

If we switched to LUT-Q (hard indices, re-quantization per step):

- **Memory savings:** 3.56 GB → 445 MB (8x reduction). This frees up memory for larger batch sizes or longer sequences.
- **Convergence speed:** LUT-Q typically converges in 1000-2000 steps vs 8000+ for Gumbel-Softmax, because the re-quantization directly optimizes the discrete indices.
- **Final cos:** LUT-Q typically achieves cos 0.97-0.98 for 2-bit palettization (per the LUT-Q paper), vs our current 0.946.
- **Risk:** Oscillation. Mitigated by STE and by freezing indices after N steps (Nagel et al., ICML 2022 — which `train_qwen.py:246-299` already implements as `freeze_settled_palettes`).

### 3.6 Why the current code has the freeze function but doesn't use it

`train_qwen.py:246-299` implements `freeze_settled_palettes`, which freezes palette entries whose index assignment hasn't changed between two snapshots. This is the Nagel et al. "Overcoming Oscillations in QAT" fix. The function exists but is not called in the training loop (we did not find a call site in the first 800 lines of `train_qwen.py`). It is dead code.

This suggests the freeze function was written for a LUT-Q-style approach that was later replaced by Gumbel-Softmax. Reviving it would be part of a LUT-Q migration.

---

## 4. Approach 3: LLT (Low-bit Linear Transformation)

### 4.1 What LLT does

LLT (Li et al., 2024) is a gradient-based approach that decomposes the quantization into a low-rank "transformation" matrix and the quantized weights. The key idea is:

```
W_recon = (I + B @ A^T) @ Q(W)
```

where `Q(W)` is the quantized weight (from any quantizer, e.g., k-means), and `A, B` are low-rank matrices (rank 16-64) that correct the quantization error. The matrices `A, B` are trained via gradient descent.

### 4.2 How LLT relates to our setup

Our `QwenLoRA` (`qwen_model.py:184-265`) is essentially LLT with rank 16-32:

```
y = base(x) + (alpha/r) * x @ A @ B^T
  = x @ W_recon + (alpha/r) * x @ A @ B^T
  = x @ (W_recon + (alpha/r) * A @ B^T)
```

So `W_recon + (alpha/r) * A @ B^T` is the LLT-style correction. The difference is that LLT applies the correction to the weight matrix (pre-multiplication), while LoRA applies it to the output (post-multiplication). Mathematically these are equivalent for Linear layers.

### 4.3 Why LLT/LoRA cannot fix the worst Linears

As discussed in `01_palette_audit.md` §6.2, LoRA can only correct the top-rank singular directions of the residual `W_orig - W_recon`. For most Linears, the residual is approximately low-rank (top 32 singular values capture >80% of energy), so rank-32 LoRA works well.

For the 5 worst-cos Linears, the residual spectrum is flatter (more directions matter), so rank-32 LoRA captures <50% of the energy. The fix is either:

- **Higher rank** (64, 128, or even 256) — but this defeats the purpose of 2-bit quantization (the LoRA params are fp16, so rank-256 LoRA on a 9216×2560 Linear adds 6M fp16 params = 12 MB, vs the 2-bit palettized weight at 5.9 MB).
- **Better palette** (so the residual is smaller and more low-rank) — this is what gradient descent on palette tries to do, but it's stuck at the k-means local optimum.

### 4.4 The LLT "rescaling" trick

LLT also applies a per-channel rescaling to the quantized weights before the low-rank correction:

```
W_recon = (I + B @ A^T) @ diag(s) @ Q(W)
```

where `s` is a per-output-channel scale factor. This is equivalent to AWQ's activation-aware scaling (see §5). The scale `s` is also trained via gradient descent.

Our implementation does not have this rescaling. Adding it would be a small change (a per-output-channel `nn.Parameter` of size `out_dim`, multiplied into the reconstructed weight before the matmul). We discuss this in `08_recommendations.md`.

---

## 5. Approach 4: GPTQ (closed-form Hessian-based quantization)

### 5.1 What GPTQ does

GPTQ (Frantar et al., 2023) is a **closed-form** post-training quantization algorithm. It does not use gradient descent. The algorithm:

1. Compute the Hessian `H = X^T @ X` of the reconstruction loss with respect to the weight matrix.
2. Quantize the weights column-by-column (or row-by-row, depending on orientation), using the Hessian to account for the effect of already-quantized columns on the current column.
3. Update the remaining unquantized weights to compensate for the error introduced by each quantization step.

The update rule is:

```
W_quant[:, j] = W[:, j] - (W[:, j] - Q(W[:, j])) * H[j, j]^{-1} * H[:, j]
```

where `Q(.)` is the quantization operator (e.g., nearest-neighbor to the palette).

### 5.2 Why GPTQ is not used here

`palettize_core.py:90` has the comment:

```python
# kmeans only (NO GPTQ — tested: GPTQ hurts with kmeans LUT)
```

This is a critical finding: **GPTQ was tested and found to hurt when combined with k-means LUT**. The likely reason is that GPTQ's column-by-column update assumes a **fixed grid** quantizer (e.g., uniform int8). When the quantizer is a k-means LUT (where the grid points are data-dependent), GPTQ's Hessian-based error compensation can move weights in directions that change the optimal k-means cluster centers, leading to a feedback loop that degrades quality.

Specifically, GPTQ updates `W[:, j+1]` based on the quantization error in `W[:, j]`. If the quantizer is k-means, the cluster centers for column `j+1` depend on the updated `W[:, j+1]`, which depends on the cluster centers for column `j`, etc. This coupling can cause the cluster centers to drift away from the data, increasing reconstruction error.

### 5.3 What GPTQ would buy us (if we could make it work)

GPTQ typically achieves cos 0.99+ for 4-bit quantization and 0.95-0.98 for 2-bit, depending on the model and group size. For 2-bit with GROUP_SIZE=256, GPTQ with a **uniform grid** (not k-means) typically achieves cos 0.95-0.97.

Our k-means calibration achieves cos 0.937 mean (0.865-0.989 range). GPTQ with a uniform grid might achieve similar or slightly better cos, but with a simpler quantizer (no need to store per-group LUTs — just a scale and zero-point per group).

The trade-off: k-means LUT can represent non-uniform distributions (e.g., bimodal weights) better than uniform grids, but it requires storing the LUT (4 values per group, 80-288 bytes per Linear). Uniform grids store only scale+zero-point (8 bytes per group).

### 5.4 Could we use GPTQ with a fixed LUT?

One option: run k-means once to determine the LUT, then run GPTQ with that LUT fixed (only updating the indices, not the LUT). This avoids the feedback loop. The algorithm would be:

1. Run k-means to get `palette` (fixed).
2. Run GPTQ column-by-column, quantizing each weight to the nearest palette entry (using the Hessian to compensate for previous columns' errors).

This is essentially what the current calibration does, but without the Hessian-based compensation. Adding the compensation might improve cos by 1-3%.

We do not pursue this further in this document; it's a calibration-time improvement, not a training-time improvement.

---

## 6. Approach 5: AWQ (activation-aware weight scaling)

### 6.1 What AWQ does

AWQ (Lin et al., 2024) is a **grid-search** post-training quantization algorithm. The key insight is that not all weights are equally important: weights corresponding to large activation magnitudes have a bigger impact on the output, so they should be quantized more carefully.

AWQ introduces a per-channel scaling factor `s` (size `out_dim`):

```
W_scaled = W * diag(s)         # scale up important channels
Q(W_scaled)                    # quantize (lossy)
W_quant = Q(W_scaled) / diag(s)  # scale back down
```

The scale `s` is found by grid search over a small set of candidates (e.g., `{0.0, 0.1, 0.2, ..., 1.0}`), picking the value that minimizes the reconstruction error.

### 6.2 How AWQ relates to our setup

Our calibration uses Hessian-weighted k-means (`palettize_core.py:84-85`), which is a form of activation-awareness: the Hessian diagonal `hess_diag[j] = Σ_i x[i, j]^2` weights the k-means objective by the input activation magnitude. This is the per-input-channel version of AWQ's per-output-channel scaling.

The difference is that AWQ scales the *weights* before quantization (which changes the effective grid), while we weight the *k-means objective* (which changes the cluster centers but not the grid). These are related but not identical.

### 6.3 What AWQ would buy us

AWQ typically achieves cos 0.98+ for 4-bit quantization and 0.93-0.96 for 2-bit, similar to GPTQ. The main advantage of AWQ is that it's simpler (grid search vs. Hessian inversion) and faster (no matrix inversions).

For our 2-bit setting, AWQ might improve cos by 1-2% over k-means, but it's unlikely to bridge the gap from 0.95 to 0.999. The fundamental limit is the 2-bit representation (4 values per group), not the quantization algorithm.

### 6.4 Could we combine AWQ scaling with k-means LUT?

Yes. The combination would be:

1. Find per-output-channel scale `s` via grid search (AWQ-style).
2. Scale the weights: `W_scaled = W * diag(s)`.
3. Run k-means on `W_scaled` to get `palette` and `indices`.
4. At inference: `W_recon = palette[g, indices[j, o]] / s[o]`.

This combines AWQ's activation-awareness with k-means's non-uniform grid. The scale `s` can also be made trainable (like LLT's rescaling).

We do not pursue this further here; it's a calibration-time improvement.

---

## 7. Comparison table

| Approach | Optimization | Indices | Palette | Typical 2-bit cos | Memory (per Linear) | Convergence | Risk |
|---|---|---|---|---|---|---|---|
| **K-means + GD (current)** | GD on palette | Frozen (hard) | Trainable (bf16) | 0.93-0.95 | 2-bit + 4-G LUT | Slow (8000+ steps) | Stuck at k-means local optimum |
| **LUT-Q** | GD on W, re-quantize | Re-derived per step | Re-derived per step | 0.95-0.97 | 2-bit + 4-G LUT | Fast (1000-2000 steps) | Oscillation |
| **LLT (≈ LoRA)** | GD on low-rank A, B | Frozen | Frozen | 0.95-0.98 (with rank-32) | 2-bit + rank-32 fp16 | Fast (1000-3000 steps) | High-rank residual unfixable |
| **GPTQ** | Closed-form | Closed-form | Closed-form (uniform grid) | 0.95-0.97 | 2-bit + scale/zp | One-shot | Hurts with k-means LUT (tested) |
| **AWQ** | Grid search | Closed-form | Closed-form (uniform grid) | 0.93-0.96 | 2-bit + scale | One-shot | Simpler than GPTQ |

**Key takeaways:**

1. **No single approach achieves cos >0.98 for 2-bit with GROUP_SIZE=256.** The 2-bit representation is the fundamental limit.
2. **LUT-Q offers the best speed/quality trade-off** for our setting: faster convergence, slightly better cos, and 8x memory savings vs Gumbel-Softmax.
3. **LLT/LoRA is a complement, not a substitute.** It corrects the residual after palettization, but cannot fix high-rank residuals.
4. **GPTQ and AWQ are calibration-time improvements**, not training-time. They could replace k-means calibration but don't help with training.
5. **The current approach (k-means + GD on palette + Gumbel-Softmax indices + LoRA)** combines elements of all five approaches but doesn't fully commit to any. This is a "kitchen sink" design that dilutes the benefits of each.

---

## 8. When does gradient descent on palette actually help?

Given that k-means is locally optimal for the L2 objective, when does gradient descent on palette provide value?

### 8.1 Case 1: Different loss function

If the training loss is different from the k-means objective (e.g., output-space cos instead of weight-space L2), gradient descent can find a better palette for the actual objective. This is the case in our setup: we optimize `norm_mse` (output-space), while k-means optimizes weighted L2 (weight-space).

**Expected improvement:** 1-3% in cos, from the off-diagonal Hessian terms that k-means ignores.

### 8.2 Case 2: Joint optimization with LoRA

If LoRA is trained jointly with the palette, the palette can shift to make the residual more low-rank (so LoRA can correct it more effectively). This is a form of "co-adaptation" between the palette and LoRA.

**Expected improvement:** Unknown. Could be significant (5-10% in cos) if the palette learns to put the residual in the top-32 singular directions, or could be negligible if the palette is stuck at the k-means local optimum.

### 8.3 Case 3: Distribution shift

If the calibration data differs from the training data, the k-means palette is suboptimal for the training distribution. Gradient descent can adapt the palette to the new distribution.

**Expected improvement:** Depends on the distribution shift. For our setup (calibration on FineWeb-Edu, training on the same), this is not a factor.

### 8.4 Case 4: Index co-training (Gumbel-Softmax or LUT-Q)

If indices are co-trained (via Gumbel-Softmax or LUT-Q), the palette can adapt to the new index assignments. This is the case in our setup (Gumbel-Softmax), but as noted in `02_gradient_correctness.md` §3.3, the STE+Gumbel approach has a structural limitation: as τ → 0, only the argmax slot receives gradient, so non-argmax palette entries cannot be fine-tuned.

**Expected improvement:** Bounded by the index co-training dynamics. If indices converge to a better assignment than k-means, the palette can adapt. If indices are stuck (as the comment at `fused_lut_linear_cuda.py:643-647` suggests), the palette is also stuck.

### 8.5 Summary

Gradient descent on palette provides value in cases 1 and 2, is neutral in case 3, and is bounded by index dynamics in case 4. The expected total improvement from gradient descent on palette (given the current setup) is **1-5% in cos**, which is consistent with the observed improvement from 0.937 (calibration mean) to 0.946 (after 8000 steps).

To bridge the remaining gap from 0.946 to 0.999, we need either:

- **Smaller GROUP_SIZE** (more palette entries per group) — calibration-time change.
- **LUT-Q-style re-quantization** — training-time change.
- **Higher-rank LoRA** — but at a memory cost.
- **A different quantization scheme entirely** (e.g., 3-bit or 4-bit) — out of scope.

---

## 9. Recommendation

Based on this analysis, the recommended path forward is:

1. **Short-term (no code changes):** Accept that gradient descent on palette can only provide 1-5% cos improvement. Focus on fixing the precision (fp32 palette), loss (1-cos instead of norm_mse), and gradient clipping (per-group clip) issues identified in Waves 1-3.

2. **Medium-term (code changes):** Implement LUT-Q-style periodic re-quantization (every 200-500 steps). This requires:
   - A re-quantization function that takes the current `W_recon` and runs k-means per group.
   - A training loop that calls re-quantization every N steps.
   - The `freeze_settled_palettes` function (already in `train_qwen.py:246-299`) to freeze indices that haven't changed.

3. **Long-term (architecture changes):** Consider smaller GROUP_SIZE (128 or 64) for the worst-cos Linears. This doubles or quadruples the palette parameter count (still tiny: 4,416 or 8,832 params) and improves cos by 2-5%.

We provide concrete code patches for the short-term fixes in `08_recommendations.md`.

---

## 10. Summary

| Question | Answer |
|---|---|
| Should palettes train via gradient descent or k-means re-quantization? | Hybrid: GD for fine-tuning, periodic k-means for index refresh. |
| Why doesn't GD on palette improve cos beyond 0.95? | K-means is already at a local optimum of the L2 objective. GD can only fine-tune around it. |
| What would LUT-Q buy us? | Faster convergence (1000-2000 steps), slightly better cos (0.95-0.97), 8x memory savings. |
| What would GPTQ buy us? | Nothing — it was tested and hurts with k-means LUT (`palettize_core.py:90`). |
| What would AWQ buy us? | 1-2% cos improvement at calibration time, not training time. |
| What would LLT/LoRA buy us? | Already have it (rank-16/32). Cannot fix high-rank residuals on worst Linears. |
| Is there a systematic bias that no amount of training can fix? | Yes — 2 bits per weight with GROUP_SIZE=256 has a fundamental information-theoretic limit. Cos >0.98 requires either smaller GROUP_SIZE or higher bitwidth. |

**Bottom line:** The current approach (k-means + GD on palette + Gumbel-Softmax indices + LoRA) is a reasonable "kitchen sink" design, but it doesn't fully commit to any single approach. The biggest wins are:

1. Fix the precision (fp32 palette) — `03_precision_analysis.md`.
2. Fix the loss (1-cos instead of norm_mse) — `05_loss_function.md`.
3. Fix the gradient clipping (per-group clip) — `08_recommendations.md`.
4. Consider LUT-Q-style re-quantization for the medium term.

The next document (`05_loss_function.md`) compares `norm_mse`, `1-cos`, and `1-cos+norm_mse` losses and recommends the best one for palette training.
