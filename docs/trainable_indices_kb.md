# Knowledge Base: Trainable Indices for 2-bit LUT Palettization

## Problem Statement

Our Qwen3.5-4B 2-bit palettized model plateaus at cos~0.93 during distillation training. The model has sufficient expressivity (112M correction layer + 2M LoRA + trainable palettes) to reach 0.999, but cannot because:

- **Indices are frozen** — each weight is locked to one of 4 palette entries per group (GS=256)
- The index assignment was determined by GPTQ + kmeans1d at calibration time
- As training shifts palette values, the optimal index assignment changes
- But frozen indices can't adapt — weights stuck in wrong clusters can never be reassigned

**Solution**: Make indices trainable via differentiable discrete optimization, so the model can reassign weights to better palette entries during training.

---

## Literature Review

### 1. LUT-Q (Cardinaux et al., NeurIPS 2018 Workshop) — arXiv:1811.05355
**"Iteratively Training Look-Up Tables for Network Quantization"**

- Learns a dictionary (palette) AND assigns each weight to one dictionary value
- Uses **straight-through estimator (STE)** for the assignment: forward = hard argmax, backward = soft gradient
- Iteratively alternates between:
  1. Fix dictionary, update assignments (via gradient + STE)
  2. Fix assignments, update dictionary (via gradient descent on palette values)
- Shows this is a general framework — pruning, power-of-2, uniform quant are all special cases
- **Key insight for us**: the iterative alternation between palette update and index update is the core idea. Our current code only does step 2 (fix indices, update palette). We need to add step 1 (fix palette, update indices).

### 2. LLT — Learnable Lookup Table (Wang et al., CVPR 2022)
**"Learnable Lookup Table for Neural Network Quantization"**

- Directly addresses our exact problem: differentiable LUT for weight quantization
- Formulates quantization as: `W_q = LUT[indices]` where both LUT and indices are learnable
- **Key technique**: uses a **soft assignment matrix** `P` of shape `(n_weights, n_palette_entries)`:
  - `P[i, k]` = probability that weight `i` belongs to palette entry `k`
  - Forward: `W_q[i] = Σ_k P[i, k] * LUT[k]` (soft combination)
  - Backward: gradients flow through P to update assignment probabilities
  - Annealing: gradually sharpen P toward one-hot (temperature schedule)
- **Critical difference from Gumbel-Softmax**: LLT uses deterministic soft assignment, not stochastic sampling
- Applied to 4-bit quantization on ResNet, EDSR — shows 1-2% accuracy recovery vs frozen LUT
- Code: https://github.com/SYSU-SAIL/LLT

### 3. GSQ — Gumbel-Softmax Quantization (Dadgarnia et al., 2026) — arXiv:2604.18556
**"GSQ: Highly-Accurate Low-Precision Scalar Quantization for LLMs via Gumbel-Softmax Sampling"**

- Most directly relevant to our LLM use case (Qwen, Llama at 2-3 bits)
- Uses **Gumbel-Softmax relaxation** to learn discrete grid assignments
- Two-stage pipeline:
  1. GPTQ initialization (what we already do)
  2. Gumbel-Softmax refinement — train logits over discrete values with temperature annealing
- Applied to Llama-3.1-8B/70B, Kimi-K2.5, **Qwen3-8B** at 2-3 bits
- **Key result**: at 2.13 bpp, GSQ achieves 68.55 avg accuracy on Llama-8B (vs 37.53 for GPTQ)
- Code: https://github.com/IST-DASLab/GSQ
- **Critical detail**: GSQ is PTQ (post-training), not QAT (quantization-aware training). But the Gumbel-Softmax relaxation technique is directly applicable to our QAT setting.

### 4. FLUTE (Guo et al., EMNLP 2024) — arXiv:2407.10960
**"Fast Matrix Multiplications for Lookup Table-Quantized LLMs"**

- NOT about trainable indices — it's about fast inference kernels for LUT-quantized models
- But introduces **NFL (Learned Normal Float)**: the LUT entries themselves are learned via STE
  - Initializes LUT with NF (NormalFloat) values
  - Then trains the LUT entries using straight-through estimation
  - Shows the LUT entries (not indices) can be learned during fine-tuning
- **Key insight**: even without trainable indices, learning the LUT entries helps. But the gap to full precision remains because indices are frozen.
- Code: https://github.com/HanGuo97/flute

### 5. LSQ — Learned Step Size Quantization (Esser et al., CVPR 2019) — arXiv:1902.08153
- Not LUT-based, but relevant for the gradient estimation technique
- Learns the quantization step size via STE
- **Key insight**: the STE gradient for the quantizer is `sign(grad) * step_size` — the gradient bypasses the rounding operation
- 1485 citations — the canonical reference for STE in quantization training

### 6. Pixel Embedding (Tokunaga et al., 2024) — arXiv:2407.16174
- Differentiable LUT for input quantization (not weight quantization)
- Replaces float-valued input pixels with vectors of quantized values via a trainable LUT
- Shows the LUT can be trained end-to-end with backprop
- Less relevant to our weight-quantization use case, but confirms the differentiable LUT approach works

---

## Three Approaches for Our Architecture

### Approach A: Straight-Through Estimator (STE) — Simplest

**How it works:**
1. Store logits `L` of shape `(in_features, out_features, 4)` — one logit per palette entry per weight
2. Forward: `indices = argmax(L, dim=-1)` (hard, non-differentiable)
3. Forward: `W = palette[indices]` (gather — same as current)
4. Backward: gradient flows through `argmax` as if it were identity (STE)
5. The logits `L` are updated by the optimizer

**Pros:**
- Simplest to implement — just add a logit parameter and a custom autograd Function
- No temperature schedule needed
- Works with our existing CUDA kernel (indices are still discrete int8 at forward time)

**Cons:**
- STE gradients are biased — the gradient w.r.t. logits doesn't account for the discrete nature
- Can oscillate (same problem as Nagel et al. ICML 2022)
- May need gradient clipping or noise injection

**Implementation sketch:**
```python
class TrainableIndicesSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, palette):
        # logits: (K, N, 4) — one logit per palette entry per weight
        # palette: (G, 4) — the LUT entries
        indices = logits.argmax(dim=-1)  # (K, N) — hard assignment
        # Gather W from palette using indices
        # W[j, o] = palette[o // GS, indices[j, o]]
        group = torch.arange(N, device=logits.device) // GS
        flat_idx = group.unsqueeze(0) * 4 + indices  # (K, N)
        W = palette.reshape(-1)[flat_idx]  # (K, N) bf16
        ctx.save_for_backward(logits, indices)
        return W

    @staticmethod
    def backward(ctx, grad_W):
        logits, indices = ctx.saved_tensors
        # STE: gradient passes through as-is
        grad_logits = torch.zeros_like(logits)
        # Scatter grad_W into the argmax position
        grad_logits.scatter_(-1, indices.unsqueeze(-1), grad_W.unsqueeze(-1))
        return grad_logits, None
```

**Memory overhead:** logits `(K, N, 4)` × 4 bytes (fp32) = 4× the indices tensor.
For our largest Linear (9216×2560): 4 × 9216 × 2560 × 4 = 377 MB extra per Linear.
With 43 Linears: **~16 GB extra** — too much for 24GB GPU.

**Mitigation:** Store logits at fp16 (2 bytes) → 8 GB extra. Still tight but feasible if we freeze some layers.

### Approach B: Gumbel-Softmax Relaxation — Most Principled

**How it works:**
1. Store logits `L` of shape `(in_features, out_features, 4)`
2. Forward: `P = softmax((L + Gumbel_noise) / temperature, dim=-1)` — soft assignment
3. Forward: `W = Σ_k P[k] * palette[k]` — soft combination of all 4 palette entries
4. Backward: gradients flow through softmax naturally (no STE)
5. Anneal temperature: start at τ=1.0, decay to τ→0 (becomes hard argmax)

**Pros:**
- Unbiased gradient estimator (with proper Gumbel noise)
- Temperature schedule gives smooth transition from exploration to exploitation
- Well-studied in literature (Jang et al. 2017, Maddison et al. 2017)
- GSQ shows it works for LLMs at 2-3 bits

**Cons:**
- Forward pass uses soft combination (not discrete indices) — **breaks our CUDA kernel**
- Need a separate training-mode kernel that handles (K, N, 4) probability tensors
- Temperature schedule adds hyperparameters
- Higher memory: logits (K, N, 4) + probabilities (K, N, 4) during forward

**Implementation approach:**
- During training: use soft forward (matmul with soft-combined W)
- During eval/inference: use hard argmax + CUDA kernel (fast)
- Switch between the two via `model.train()` / `model.eval()`

**Memory:** Same as STE — logits at fp16 = ~8 GB extra.

### Approach C: Iterative Re-quantization — Practical Compromise

**How it works:**
1. Train normally with frozen indices + trainable palette (current approach)
2. Every N steps (e.g., 500), re-run kmeans1d on the current reconstructed weights
3. Update the frozen indices to the new kmeans assignment
4. Resume training with the new indices

**Pros:**
- No extra memory (indices stay int8, no logits)
- No custom autograd Function needed
- Works with our existing CUDA kernel
- Simple to implement (just add a periodic re-kmeans step)

**Cons:**
- Not truly "trainable" — indices are updated in discrete jumps, not continuously
- May cause training instability (sudden index changes)
- kmeans1d re-computation takes time (~1s per Linear × 43 Linears = ~43s per cycle)
- The palette values may need to re-adjust after each index update

**Implementation:**
```python
def re_quantize_indices(model, sb_idx):
    """Re-run kmeans1d on current reconstructed weights, update indices."""
    from palettize_core import kmeans1d_weighted
    for name, mod in model.named_modules():
        if not isinstance(mod, PalettizedLinear):
            continue
        # Reconstruct current weights
        W = mod.reconstruct_weight()  # (in_dim, out_dim) bf16
        # Re-run kmeans per group
        for g in range(mod.n_groups):
            s = g * mod.group_size
            e = s + mod.group_size
            w_group = W[:, s:e]  # (in_dim, GS)
            # kmeans1d on flattened group
            indices_new, lut_new = kmeans1d_weighted(
                w_group.flatten().float(),
                torch.ones_like(w_group.flatten().float()),
                2,  # 2-bit = 4 entries
                mod.group_size,
            )
            mod.indices[:, s:e] = indices_new.reshape(w_group.shape)
            mod.palette[g] = lut_new
        # Update int8 buffer for CUDA kernel
        mod.indices_int8 = mod.indices.to(torch.int8)
```

---

## Recommended Approach for Our Architecture

### Phase 1: Iterative Re-quantization (Approach C) — Quick Win
- **Effort**: Low (1 function + periodic call in training loop)
- **Expected gain**: +0.02-0.05 cos (break through 0.93 plateau)
- **Risk**: Low — indices are updated in controlled jumps, not continuous
- **Timeline**: Implement in 1 hour, test in 1 training run

### Phase 2: STE Trainable Indices (Approach A) — Medium Effort
- **Effort**: Medium (custom autograd Function + fp16 logits + gradient handling)
- **Expected gain**: +0.03-0.08 cos (reach 0.96-0.98)
- **Risk**: Medium — STE bias may cause oscillation, need careful LR tuning
- **Memory**: ~8 GB extra for fp16 logits (feasible if we freeze some layers)
- **Timeline**: Implement in 4 hours, test in 1-2 training runs
- **Key**: Use the STE only for the original layers (0-3), NOT the correction layer or LoRA (those are dense and already trainable)

### Phase 3: Gumbel-Softmax (Approach B) — High Effort
- **Effort**: High (separate training-mode kernel, temperature schedule, soft forward)
- **Expected gain**: +0.05-0.10 cos (reach 0.98-0.999)
- **Risk**: High — complex implementation, temperature tuning, may break CUDA kernel
- **Memory**: ~8 GB extra for logits + probabilities
- **Timeline**: Implement in 1-2 days, test in 3-5 training runs
- **Only worth it if Phase 1+2 don't reach 0.96+**

---

## Detailed Implementation Plan: Phase 2 (STE Trainable Indices)

### Data Structures

```python
class PalettizedLinear(nn.Module):
    def __init__(self, ...):
        # Existing:
        self.palette = nn.Parameter(...)  # (G, 4) bf16 — trainable
        self.register_buffer("indices", ...)  # (K, N) int8 — currently frozen

        # NEW: trainable logits for index assignment
        # Initialize from current hard assignment (one-hot → logits)
        self.index_logits = nn.Parameter(
            torch.zeros(K, N, 4, dtype=torch.float16, device=device)
        )
        # Initialize: logit=0 for current index, logit=-10 for others
        with torch.no_grad():
            for k in range(K):
                for n in range(N):
                    self.index_logits.data[k, n, self.indices[k, n]] = 10.0
```

### Forward Pass (Training Mode)

```python
def forward(self, x):
    if self.training:
        # STE: hard argmax in forward, soft gradient in backward
        indices = STEIndices.apply(self.index_logits)  # (K, N) int8
        # Use existing CUDA kernel with the STE-derived indices
        y = fused_lut_linear(x, self.palette, indices, self.bias, self.group_size)
    else:
        # Eval mode: use frozen indices (fast, no STE overhead)
        y = fused_lut_linear(x, self.palette, self.indices_int8, self.bias, self.group_size)
    return y
```

### STE Autograd Function

```python
class STEIndices(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits):
        # logits: (K, N, 4) fp16
        # Return hard argmax as int8
        indices = logits.argmax(dim=-1).to(torch.int8)  # (K, N)
        ctx.save_for_backward(logits, indices.long())
        return indices

    @staticmethod
    def backward(ctx, grad_indices):
        # grad_indices: gradient w.r.t. the indices (from the gather+matmul backward)
        # STE: scatter gradient to the argmax position in logits
        logits, indices = ctx.saved_tensors
        grad_logits = torch.zeros_like(logits)
        # The gradient for the selected index = grad_indices
        # The gradient for non-selected indices = 0
        grad_logits.scatter_(-1, indices.unsqueeze(-1), grad_indices.unsqueeze(-1).to(logits.dtype))
        return grad_logits
```

### Challenge: Integrating STE with the CUDA Kernel

The CUDA kernel expects `indices` as an int8 buffer (not a Parameter). The STE function returns a **detached** int8 tensor. The CUDA kernel's autograd handles the gradient w.r.t. palette, but the gradient w.r.t. indices needs to flow back to `index_logits`.

**Solution**: Wrap the entire forward in a custom autograd Function that:
1. Takes `index_logits` + `palette` as inputs
2. Computes `indices = STE(index_logits)` internally
3. Calls the CUDA kernel for the matmul
4. In backward, combines the CUDA kernel's `grad_palette` with the STE's `grad_logits`

This requires modifying the CUDA kernel's backward to also return `grad_indices` (currently it returns None for indices). Then we scatter `grad_indices` into `grad_logits` via STE.

### Memory Budget

| Component | Size (bf16/fp16) | Count | Total |
|---|---|---|---|
| index_logits (K, N, 4) fp16 | 9216×2560×4×2 = 189 MB | 43 Linears | 8.1 GB |
| palette (G, 4) bf16 | <1 KB | 43 | ~0 |
| indices (K, N) int8 | 9216×2560×1 = 23.6 MB | 43 | 1.0 GB |
| **Total extra** | | | **~8.1 GB** |

Current GPU usage: ~18 GB. With 8 GB extra: ~26 GB → **exceeds 24 GB L4**.

**Mitigation:**
1. Only make indices trainable for the largest Linears (gate_proj, up_proj, down_proj) — skip small ones (k_proj, v_proj, in_proj_a/b)
2. Use gradient checkpointing on the index_logits
3. Freeze layers 0-1 (only make layers 2-3 + correction trainable)
4. Use 8-bit logits (quantize logits to int8 with scale) — halves memory to ~4 GB

---

## References

1. **LUT-Q** — Cardinaux et al., "Iteratively Training Look-Up Tables for Network Quantization", NeurIPS 2018 Workshop. arXiv:1811.05355
2. **LLT** — Wang et al., "Learnable Lookup Table for Neural Network Quantization", CVPR 2022. Code: https://github.com/SYSU-SAIL/LLT
3. **GSQ** — Dadgarnia et al., "GSQ: Highly-Accurate Low-Precision Scalar Quantization for LLMs via Gumbel-Softmax Sampling", 2026. arXiv:2604.18556. Code: https://github.com/IST-DASLab/GSQ
4. **FLUTE** — Guo et al., "Fast Matrix Multiplications for Lookup Table-Quantized LLMs", EMNLP 2024. arXiv:2407.10960. Code: https://github.com/HanGuo97/flute
5. **LSQ** — Esser et al., "Learned Step Size Quantization", CVPR 2019. arXiv:1902.08153
6. **Gumbel-Softmax** — Jang et al., "Categorical Reparameterization with Gumbel-Softmax", ICLR 2017. arXiv:1611.01144
7. **STE** — Bengio et al., "Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation", 2013. arXiv:1308.3432
8. **Pixel Embedding** — Tokunaga et al., "Pixel Embedding: Fully Quantized CNN with Differentiable Lookup Table", 2024. arXiv:2407.16174
9. **Overcoming Oscillations in QAT** — Nagel et al., ICML 2022. arXiv:2203.11086
