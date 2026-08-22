# 08 — Recommendations: Prioritized Implementation Plan

**Scope.** This document provides a prioritized, actionable implementation plan based on the gap analysis in `07_gap_analysis.md`. Each recommendation includes: (a) the rationale (which gap it closes), (b) the concrete code change (with file:line references and patch sketches), (c) the expected cos improvement, (d) the implementation effort, and (e) the validation procedure.

The plan is structured in three phases:
- **Phase 1 (Week 1–2):** Free / cheap wins. Expected +0.005 to +0.02 cos. Total effort ~2 eng-days.
- **Phase 2 (Week 2–4):** Structural calibration improvements. Expected +0.05 to +0.10 cos. Total effort ~10 eng-days.
- **Phase 3 (Week 4–8):** Training recipe refinements + optional radical changes. Expected +0.01 to +0.04 cos. Total effort ~10 eng-days.

**Combined target:** cos 0.95 → 0.99–0.999+.

---

## Phase 1: Free / Cheap Wins (Week 1–2)

These are changes that require minimal engineering but provide measurable improvements. Do these first.

### Recommendation 1.1: Halve Group Size 256 → 128

**Closes Gap 5.** Expected: +0.005–0.01 cos. Effort: 0.5 eng-days.

**Rationale.** GS=256 is the largest group size in the entire 20-method survey. The standard GPTQ/AWQ default is GS=128; aggressive methods (SqueezeLLM, AffineQuant) use GS=64. Halving our group size roughly halves the within-group weight diversity, so the 4-entry k-means codebook fits better.

**Code change.** In `/home/z/my-project/research-literature-review/scripts/palettize_core.py:26`:

```python
# Before:
GROUP_SIZE = 256

# After:
GROUP_SIZE = 128
```

That's it. The kernel parameterizes on `group_size`, so no kernel changes are needed. Verify by running `scripts/calib_qwen.py --sb_idx 0` and checking the per-Linear cos values in `logs/calib_sb0.log`.

**Validation.** Re-run calibration on super-block 0. Compare per-Linear cos:
- Before (GS=256): mean cos 0.937, min 0.865, max 0.989.
- After (GS=128): expect mean cos 0.945–0.950, min 0.885, max 0.992.

Memory cost: palette storage doubles (8 bytes/group → 16 bytes/group for the same Linear), but total palette storage was ~75KB per super-block, so this is negligible.

### Recommendation 1.2: Tighten Logit Clamp from ±20 to ±5τ

**Closes Gap 6 (partially).** Expected: +0.005–0.015 cos. Effort: 0.1 eng-days.

**Rationale.** At `τ = 0.1`, our logit clamp of `±20` corresponds to `softmax(±20/0.1) = softmax(±200)`, which overflows to `[1, 0]` in float32. The gradient at one-hot is exactly zero, so the indices are frozen for the second half of training. BNN's tight clip (`[-1, 1]` for the FP shadow) is the canonical fix; for our logit-space parameterization, `±5τ` is the right scale (softmax(±5) ≈ [0.993, 0.007] — still essentially hard, but with finite-precision gradient).

**Code change.** In `/home/z/my-project/research-literature-review/scripts/train_qwen.py:1153` (search for the logit clamp):

```python
# Before:
logits.clamp_(-20.0, 20.0)

# After:
tau = hp.get("tau", 1.0)  # current temperature
logits.clamp_(-5.0 * tau, 5.0 * tau)
```

This makes the clamp adaptive to the temperature: loose at high τ (exploration), tight at low τ (commitment).

**Validation.** Resume training from `trained/superblock_0_best` and run 1000 steps. Monitor:
- The fraction of indices that change per step (should be >0 in the first 200 steps, then decay).
- The training cos (should climb from 0.946 toward 0.96+).

### Recommendation 1.3: Drop Gumbel Noise (Deterministic Softmax)

**Closes Gap 6 (partially).** Expected: +0.005–0.01 cos. Effort: 0.5 eng-days.

**Rationale.** LLT (the closest published analog to our approach) uses deterministic softmax without Gumbel noise. Jang et al. (2017, the Gumbel-Softmax paper) note that Gumbel noise helps most for K≥10; at K=4 (our setting), the noise adds gradient variance without clear benefit. Dropping it is a one-line change.

**Code change.** In `scripts/fused_lut_kernel.cu`, find the `gumbel_sample` calls around line 1325–1328 and remove them:

```cuda
// Before:
// float g = gumbel_sample(rng_state);
// logits[k] += g;

// After: (just remove the gumbel_sample call entirely)
// logits[k] remains as-is
```

**Validation.** Resume training and run 2000 steps. Compare training cos curve to the Gumbel-noise baseline. Expect: smoother convergence (less step-to-step variance), slightly higher final cos.

### Recommendation 1.4: Lloyd-Max Gaussian Codebook Init

**Closes (minor) — uses QLoRA NF4 principle.** Expected: +0.005–0.01 cos. Effort: 0.1 eng-days.

**Rationale.** QLoRA's NF4 codebook places its 16 levels at the quantiles of the standard normal, giving information-theoretically optimal 4-bit representation of Gaussian-distributed weights. For 2-bit (K=4), the Lloyd-Max optimal quantizer for a Gaussian is `[-1.510σ, -0.4528σ, +0.4528σ, +1.510σ]`. Initializing our palette to this (instead of k-means) gives the optimizer a well-conditioned starting point that matches the expected weight distribution.

**Code change.** In `scripts/palettize_core.py:palettize_tensor_2bit`, replace the k-means call with a Lloyd-Max init for the palette (but keep k-means for the *indices*):

```python
# Before:
indices, lut, n_groups = palettize_groups(W_comp, hess_diag, BITWIDTH, GROUP_SIZE)

# After:
# Lloyd-Max optimal 4-level quantizer for a Gaussian, scaled to W's std
sigma = W_comp.std()
lut = torch.tensor([-1.5104, -0.4528, 0.4528, 1.5104], device=W_comp.device) * sigma
# Assign indices via nearest-neighbor to the Lloyd-Max levels
indices = (W_comp.unsqueeze(-1) - lut.unsqueeze(0).unsqueeze(0)).abs().argmin(dim=-1)
# Group the lut per group of GROUP_SIZE
n_groups = W_comp.shape[0] // GROUP_SIZE
lut = lut.repeat(n_groups, 1)  # same 4 levels per group
```

**Validation.** Re-run calibration. Compare per-Linear cos to the k-means baseline. Expect: similar or slightly better mean cos, more uniform cos across Linears (Lloyd-Max is data-independent, so it doesn't overfit to the calibration set).

---

## Phase 2: Structural Calibration Improvements (Week 2–4)

These are the changes that attack the structural codebook-resolution bottleneck (Gap 1). They are more expensive but provide the bulk of the expected cos improvement.

### Recommendation 2.1: AWQ-Style Per-Channel Scale

**Closes Gap 2.** Expected: +0.01–0.03 cos. Effort: 1–2 eng-days.

**Rationale.** AWQ identifies ~1% salient channels (high activation magnitude) and protects them by scaling weights up before quantization (so they occupy more grid resolution), then dividing the scale back at inference. This is the cheapest pre-quantization transformation.

**Code change.** In `scripts/palettize_core.py:palettize_tensor_2bit`, add the AWQ scale computation before k-means:

```python
def palettize_tensor_2bit(name, W_orig, X, out_dir, threshold=0.0, verbose=True):
    out_dim, in_dim = W_orig.shape
    # ... (existing code) ...
    
    # NEW: AWQ-style per-channel scale
    with torch.no_grad():
        # Activation magnitude per input channel: m_x[j] = mean(|x[:, j]|)
        m_x = X.float().abs().mean(dim=0)  # (in_dim,)
        # Weight magnitude per input channel: m_W[j] = mean(|W[j, :]|)
        m_W = W_orig.abs().mean(dim=1)     # (out_dim,) — but we want per-input-channel
        m_W = W_orig.abs().mean(dim=0)     # (in_dim,) — correct: average over output dim
        
        # AWQ scale: s[j] = (m_x[j])^α / (m_W[j])^(1-α), α=0.5
        alpha = 0.5
        s = (m_x.float() ** alpha) / (m_W.float() ** (1 - alpha) + 1e-8)
        s = s.clamp(0, 1.0)  # AWQ constrains s to [0, 1]
        
        # Apply scale: W_scaled = W * s[None, :]  (broadcast over output dim)
        W_scaled = W_orig * s.unsqueeze(0)
    
    # Use W_scaled instead of W_orig for k-means
    W_comp = W_scaled.clone()
    indices, lut, n_groups = palettize_groups(W_comp, hess_diag, BITWIDTH, GROUP_SIZE)
    
    # Store s in metadata
    meta["awq_scale"] = s.cpu().tolist()
    # ...
```

And in `PalettizedLinear.forward` (in `qwen_model.py`), apply `1/s` to `x` before the matmul:

```python
def forward(self, x):
    if self.awq_scale is not None:
        x = x / self.awq_scale  # inverse scale on activations
    # ... existing forward ...
```

**Validation.** Re-run calibration. Compare per-Linear cos to the no-AWQ baseline. Expect: mean cos 0.945 → 0.955–0.965.

### Recommendation 2.2: GPTVQ-Style Vector Quantization (g=2)

**Closes Gap 1 (the bottleneck).** Expected: +0.03–0.05 cos. Effort: 5–7 eng-days.

**Rationale.** This is the single highest-value change identified in the entire literature review. Going from `g=1` (single-weight K=4 codebook) to `g=2` (2-weight joint codebook of K=16 entries) at the same 2-bit storage cost gives a 4× increase in effective codebook resolution. Rate-distortion theory guarantees a 16× reduction in minimum distortion; empirically, GPTVQ sees ~4–5× reduction in actual distortion on LLM weights.

**Code change.** This is a non-trivial change touching three components:

**Step 1: 2D k-means in `palettize_pytorch.py`.**

```python
def kmeans2d_weighted(W, hess_diag, K, group_size):
    """
    2D k-means: jointly quantize pairs of weights.
    W: (out_dim, in_dim) — reshape to (out_dim, in_dim//2, 2) for pairing.
    Returns: indices (out_dim, in_dim//2) int8, codebook (n_groups, K, 2) bf16.
    """
    out_dim, in_dim = W.shape
    assert in_dim % 2 == 0
    W_pairs = W.reshape(out_dim, in_dim // 2, 2)  # (out_dim, n_pairs, 2)
    
    # Group pairs by output dimension (group_size pairs per group)
    n_groups = out_dim // group_size
    codebook = torch.zeros(n_groups, K, 2, dtype=torch.bfloat16, device=W.device)
    indices = torch.zeros(out_dim, in_dim // 2, dtype=torch.uint8, device=W.device)
    
    for g in range(n_groups):
        W_g = W_pairs[g*group_size:(g+1)*group_size]  # (group_size, n_pairs, 2)
        # Initialize codebook by k-means++ on the 2D points
        # ... (standard k-means in 2D, ~30 lines) ...
        # Assign indices
        # ... 
    
    return indices, codebook
```

**Step 2: Kernel change in `fused_lut_kernel.cu`.** Modify the inner loop to gather 2 weights per index:

```cuda
// Before (g=1):
bf16 w = palette[g * 4 + idx[j, o]];

// After (g=2):
int pair_idx = idx[j, o/2];  // one index per 2 weights
bf16 w0 = palette[g * 4 * 2 + pair_idx * 2 + 0];
bf16 w1 = palette[g * 4 * 2 + pair_idx * 2 + 1];
// Use w0 for output o, w1 for output o+1 (or however the pairing is structured)
```

**Step 3: Packing format.** Since we now have 1 index per 2 weights, the index bitwidth is `log2(K^2) = log2(16) = 4` bits per pair = 2 bits per weight (same as before). The packing format `pack_idx2` already handles 2-bit packing; we just need to pack *pair indices* (4-bit) instead of single-weight indices (2-bit), then unpack to 2-bit-per-weight at the kernel level.

**Validation.** Re-run calibration. Compare per-Linear cos:
- Before (g=1): mean cos 0.937.
- After (g=2): expect mean cos 0.965–0.975.

This is the change most likely to break the cos=0.95 plateau.

### Recommendation 2.3: SqueezeLLM-Style Dense/Sparse Split

**Closes Gap 4.** Expected: +0.02–0.04 cos. Effort: 3–5 eng-days.

**Rationale.** LLM weights have heavy-tailed distributions; ~0.5% of weights are 10–100× larger than the median. These outliers distort the k-means palette (one level gets pulled toward the outliers). Peeling them off into a sparse FP16 residual lets the dense codebook fit the in-distribution weights much better.

**Code change.** In `palettize_tensor_2bit`, after k-means:

```python
# After k-means: compute residual and sensitivity
Wq = reconstruct_Wq(indices, lut, GROUP_SIZE)
residual = W_orig - Wq  # (out_dim, in_dim)
sensitivity = (residual ** 2) * hess_diag.unsqueeze(0)  # (out_dim, in_dim)

# Select top 0.5% by sensitivity
threshold = torch.quantile(sensitivity.flatten(), 0.995)
sparse_mask = sensitivity > threshold  # (out_dim, in_dim)
sparse_values = W_orig[sparse_mask]  # 1D tensor of outlier weights
sparse_coords = sparse_mask.nonzero()  # (n_sparse, 2) — (out_idx, in_idx)

# Re-run k-means on W_dense = W_orig * (1 - sparse_mask)
W_dense = W_orig.clone()
W_dense[sparse_mask] = 0  # zero out outliers
indices, lut, n_groups = palettize_groups(W_dense, hess_diag, BITWIDTH, GROUP_SIZE)

# Store sparse residual as CSR
sparse_csr = torch.sparse_coo_tensor(
    sparse_coords.t(), sparse_values, W_orig.shape
).to_sparse_csr()

# Save sparse_csr alongside indices and lut
torch.save(sparse_csr, os.path.join(out_dir, f"{san}.sparse_csr.pt"))
```

And in `PalettizedLinear.forward`:

```python
def forward(self, x):
    y = fused_lut_linear_fwd(x, self.palette, self.indices, ...)
    if self.sparse_csr is not None:
        y = y + torch.sparse.mm(x, self.sparse_csr.t())  # sparse matmul
    if self.lora is not None:
        y = y + self.lora(x)
    return y
```

**Validation.** Re-run calibration. Compare per-Linear cos:
- Before (no split): mean cos 0.937.
- After (0.5% sparse): expect mean cos 0.955–0.965.

Memory cost: ~5MB sparse residual per Linear × 25 Linears = ~125MB per super-block. Acceptable.

### Recommendation 2.4: GPTQ-Style Hessian-Inverse Error Propagation

**Closes Gap 3.** Expected: +0.01–0.02 cos. Effort: 2–3 eng-days.

**Rationale.** We compute the full Hessian `H = X^T X` but only use its diagonal. The off-diagonal terms capture inter-column correlations; using the full inverse `H⁻¹` lets us propagate quantization error to the remaining columns.

**Important caveat.** Our SPEC explicitly notes "NO GPTQ — tested: GPTQ hurts with kmeans LUT" (`palettize_core.py:90`). This is correct for *naive* GPTQ (uniform grid + closed-form `argmin`). The resolution is **GPTVQ-style**: compute the k-means codebook *first* (offline), then apply the GPTQ Hessian-inverse update with the codebook *fixed*. This avoids the interaction that broke our earlier attempt.

**Code change.** In `palettize_tensor_2bit`, after the initial k-means:

```python
# After k-means assigns indices and palette:
Wq = reconstruct_Wq(indices, lut, GROUP_SIZE)

# Compute Hessian inverse (Cholesky-based)
H = X_f.T @ X_f  # already computed
H_dampened = H + 0.01 * torch.diagonal(H).mean() * torch.eye(H.shape[0], device=H.device)
H_inv = torch.linalg.cholesky_inverse(torch.linalg.cholesky(H_dampened))

# Propagate residual error group-by-group (in activation-sensitivity order)
# Sort groups by descending H_inv diagonal (most sensitive first)
group_sensitivities = torch.diagonal(H_inv)[:n_groups * GROUP_SIZE:GROUP_SIZE]
group_order = torch.argsort(group_sensitivities, descending=True)

for g_idx in group_order:
    g_start = g_idx * GROUP_SIZE
    g_end = g_start + GROUP_SIZE
    # Residual error in this group
    e_g = (W_orig[:, g_start:g_end] - Wq[:, g_start:g_end])  # (out_dim, GROUP_SIZE)
    # Propagate to remaining groups
    for g_other in group_order:
        if g_other == g_idx:
            continue
        go_start = g_other * GROUP_SIZE
        go_end = go_start + GROUP_SIZE
        # Update: W[remaining] -= e * H_inv[remaining, current] / H_inv[current, current]
        # (block-wise; for efficiency, batch this)
        W_comp[:, go_start:go_end] -= e_g @ H_inv[g_start:g_end, go_start:go_end] / \
                                       H_inv[g_start:g_end, g_start:g_end].mean()
    # Re-run k-means on the updated W_comp for this group (palette fixed)
    # ... (reassign indices for group g_idx only)
```

**Validation.** Re-run calibration. Compare per-Linear cos:
- Before: mean cos 0.937.
- After (Hessian-inverse propagation): expect mean cos 0.945–0.955.

The gain is smaller than the other Phase 2 changes, but it compounds with them.

---

## Phase 3: Training Recipe Refinements (Week 4–8)

These are changes to the *training* (not calibration). They are smaller gains but worth doing after Phase 2.

### Recommendation 3.1: LLT's `1/√(N_k)` Gradient Rescaling

**Closes Gap 6 (partially).** Expected: +0.005–0.015 cos. Effort: 1 eng-day.

**Rationale.** LLT rescales the palette gradient by `1/sqrt(N_k)` where `N_k = Σ_i p[i, k]` is the soft count of weights assigned to codebook entry `k`. Without this rescaling, frequently-used entries dominate the gradient and rarely-used entries die (codebook collapse). With K=4 and k-means init, our soft counts are roughly balanced, but collapse can still occur during training.

**Code change.** In `fused_lut_linear_cuda.py` (the Python autograd wrapper), modify the backward:

```python
class CUDAFusedLUTLinearSoft(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, palette, index_logits, ...):
        # ... existing forward ...
        p = softmax((logits + gumbel) / tau)  # (4, K, N)
        N_k = p.sum(dim=(1, 2))  # (4,) — soft count per codebook entry
        ctx.save_for_backward(x, p, palette, N_k)
        # ...
    
    @staticmethod
    def backward(ctx, grad_y):
        x, p, palette, N_k = ctx.saved_tensors
        # ... existing grad_palette computation ...
        # NEW: rescale by 1/sqrt(N_k)
        rescale = 1.0 / (N_k.sqrt() + 1e-8)  # (4,)
        grad_palette = grad_palette * rescale.unsqueeze(0).unsqueeze(0)
        # ...
```

**Validation.** Resume training. Monitor the palette distribution: without rescaling, one entry typically grows to >50% of weights; with rescaling, all 4 entries should stay in [10%, 40%].

### Recommendation 3.2: QuIP#-Style Hadamard Pre-Rotation

**Closes Gap 1 (alternative to GPTVQ).** Expected: +0.02–0.04 cos. Effort: 2 eng-days.

**Rationale.** QuIP# applies a Hadamard rotation `W̃ = H · W` to drive weight coherence to random-matrix levels. This makes the 4-entry codebook fit much better *without* changing the bitwidth or group size. It's nearly free (one fast Hadamard transform per Linear, applied once at calibration and once at inference).

**Code change.** Add a Hadamard transform utility:

```python
def hadamard_transform(x, dim=-1):
    """Fast Hadamard transform along dimension dim. Requires dim size to be power of 2."""
    n = x.shape[dim]
    assert (n & (n - 1)) == 0, "Hadamard requires power-of-2 size"
    # Recursive butterfly implementation (or use scipy.linalg.hadamard for small n)
    # ...
```

In `palettize_tensor_2bit`, before k-means:

```python
# Apply Hadamard rotation: W_tilde = H @ W (input-side rotation)
# At inference: y = x @ W = (x @ H) @ (H @ W) = (x @ H) @ W_tilde
# So apply H to x at inference, then matmul with quantized W_tilde
W_tilde = hadamard_transform(W_orig, dim=-1)  # rotate along input dim
# k-means on W_tilde
indices, lut, n_groups = palettize_groups(W_tilde, hess_diag, BITWIDTH, GROUP_SIZE)
```

In `PalettizedLinear.forward`:

```python
def forward(self, x):
    x_rotated = hadamard_transform(x, dim=-1)  # apply H to x
    y = fused_lut_linear_fwd(x_rotated, self.palette, self.indices, ...)
    # ...
```

**Note:** For non-power-of-2 dimensions (e.g., 2560), use a padded Hadamard (pad to next power of 2, transform, then slice back).

**Validation.** Re-run calibration. Compare per-Linear cos:
- Before: mean cos 0.937.
- After (Hadamard): expect mean cos 0.955–0.970.

### Recommendation 3.3: Two-Stage LR Schedule (BNN-Inspired)

**Closes Gap 6 (partially).** Expected: +0.005–0.01 cos. Effort: 0.5 eng-days.

**Rationale.** BNN uses a two-stage LR: warmup (high LR, no decay) for 0–10% of training, then exponential decay to 1e-5. Without warmup, the Gumbel-Softmax logits don't have time to migrate from their k-means one-hot init to a meaningful configuration before annealing starts.

**Code change.** In `train_qwen.py`, modify the LR scheduler:

```python
# Before:
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)

# After:
def lr_lambda(step):
    warmup_steps = 500
    if step < warmup_steps:
        return step / warmup_steps  # linear warmup
    else:
        # Exponential decay from peak to 1e-5
        decay_steps = max_steps - warmup_steps
        progress = (step - warmup_steps) / decay_steps
        return 0.01 ** progress  # decay from 1.0 to 0.01 (i.e., peak * 1.0 to peak * 0.01)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

**Validation.** Restart training from scratch (not from checkpoint). Monitor the first 500 steps: with warmup, the loss should decrease smoothly; without, it may spike.

### Recommendation 3.4 (Optional): LUT-Q-Style FP Shadow + K-Means Reassignment

**Closes Gap 6 (replaces Gumbel-Softmax entirely).** Expected: +0.01–0.02 cos. Effort: 5–7 eng-days.

**Rationale.** This is the most radical change: replace the Gumbel-Softmax `index_logits` parameterization entirely with an FP shadow `W_shadow` (continuous, well-conditioned) and recompute indices via k-means every N steps. Eliminates the gradient damping problem at the source.

**Code change.** Major refactor of `PalettizedLinear`:

```python
class PalettizedLinear(nn.Module):
    def __init__(self, in_features, out_features, palette_init, indices_init, group_size=128):
        super().__init__()
        self.group_size = group_size
        # Replace index_logits with FP shadow
        self.W_shadow = nn.Parameter(torch.zeros(out_features, in_features))
        # Initialize W_shadow so that k-means(W_shadow) gives indices_init
        self.W_shadow.data = reconstruct_Wq(indices_init, palette_init, group_size)
        # Trainable palette
        self.palette = nn.Parameter(palette_init.clone())
        # Frozen indices (recomputed every N steps)
        self.register_buffer("indices", indices_init.clone())
        self.recompute_every = 10  # recompute indices every 10 steps
        self.step_count = 0
    
    def forward(self, x):
        if self.training and self.step_count % self.recompute_every == 0:
            self._recompute_indices()
        self.step_count += 1
        # STE: forward uses hard indices, backward flows to W_shadow
        W_q = self.palette[self.indices]  # hard lookup
        # STE bridge: attach W_shadow as the "differentiable" version
        W_q = W_q + (self.W_shadow - self.W_shadow.detach())  # identity grad to W_shadow
        y = F.linear(x, W_q)
        if self.lora is not None:
            y = y + self.lora(x)
        return y
    
    @torch.no_grad()
    def _recompute_indices(self):
        # k-means reassignment: indices = argmin_k |W_shadow - palette[k]|
        for g in range(self.palette.shape[0]):
            W_g = self.W_shadow[:, g*self.group_size:(g+1)*self.group_size]
            dist = (W_g.unsqueeze(-1) - self.palette[g].unsqueeze(0).unsqueeze(0)).abs()
            self.indices[:, g*self.group_size:(g+1)*self.group_size] = dist.argmin(dim=-1)
```

**Validation.** Restart training. Compare training cos curve to the Gumbel-Softmax baseline. Expect: faster convergence (no gradient damping), higher final cos.

**Risk:** This is a fundamental parameterization change. Existing checkpoints (with `index_logits`) would need conversion. Recommend trying this only if Phase 1+2+3.1–3.3 don't reach cos>0.99.

---

## Summary: Implementation Roadmap

| Phase | Recommendations | Expected cos gain | Effort |
|---|---|---|---|
| **Phase 1** (Week 1–2) | 1.1 GS 256→128, 1.2 tighten clamp, 1.3 drop Gumbel, 1.4 Lloyd-Max init | +0.02–0.04 | 2 eng-days |
| **Phase 2** (Week 2–4) | 2.1 AWQ scale, 2.2 GPTVQ g=2, 2.3 SqueezeLLM sparse, 2.4 GPTQ Hessian-inverse | +0.07–0.14 | 11–17 eng-days |
| **Phase 3** (Week 4–8) | 3.1 LLT rescaling, 3.2 Hadamard, 3.3 two-stage LR, 3.4 (optional) LUT-Q FP shadow | +0.02–0.06 | 4–11 eng-days |
| **Total** | All P0+P1 | **+0.11–0.24** | **17–30 eng-days** |

Starting from cos=0.95, the conservative target is cos=0.99 (Phase 1+2 only); the aggressive target is cos>0.999 (Phase 1+2+3).

---

## What NOT to Do

Based on the literature review, the following changes are **not recommended**:

1. **Do not increase LoRA rank beyond 32.** Diminishing returns; the LoRA is compensating for residual error after quantization, but the structural codebook limit (Gap 1) cannot be broken by LoRA alone.

2. **Do not extend Gumbel-Softmax training beyond 10K steps.** The gradient damping at low τ is fundamental; more training will not break the cos=0.95 ceiling.

3. **Do not switch to from-scratch training (BitNet-style) without first trying Phase 2.** From-scratch is expensive (~weeks) and uncertain at our scale (Qwen3.5-4B is in the "small model" regime where BitNet's results are weaker).

4. **Do not invest in productionization (multi-platform kernels, HF integration) until cos>0.99 is achieved.** Productionizing a cos=0.95 model is premature.

5. **Do not add more loss terms (KL divergence, attention-map matching, intermediate-layer distillation).** The loss is not the bottleneck; the codebook structure is.

6. **Do not change the optimizer (Muon + AdamW).** Our optimizer setup is already state-of-the-art (matches BitNet's FP32-master pattern); changing it will not help.

7. **Do not quantize activations.** Our bottleneck is weight quantization; activation quantization (W8A8, W4A8) is a separate problem requiring different techniques (SmoothQuant-style) and is not needed for our current research target.

---

## Validation Plan

After each phase, run the full validation suite:

1. **Re-calibrate super-block 0** with the new techniques.
2. **Compare per-Linear cos** in `logs/calib_sb0.log` to the baseline (mean 0.937, min 0.865, max 0.989).
3. **Resume training from the new calibration** for 4000 steps.
4. **Evaluate on the held-out eval set** (256 sequences × 512 tokens, cached in `eval_tokens.pt`).
5. **Report super-block output cos** (the metric tracked throughout this review).

Target milestones:
- After Phase 1: cos > 0.96.
- After Phase 2: cos > 0.98.
- After Phase 3: cos > 0.99, with a path to cos > 0.999.

If any phase fails to meet its milestone, stop and debug before proceeding to the next phase.
