# 03 — Straight-Through Estimator (STE) Analysis

**Scope:** Mathematical proof of STE correctness (and incorrectness), alternative formulations, and concrete recommendations for fixing the vanishing-gradient problem.

---

## 1. The STE used in this repo

The Straight-Through Estimator in `fused_lut_linear_cuda.py` (around the `CUDAFusedLUTLinearSoft.forward` method) is:

```python
# STE trick: forward = W_hard, backward = through W_soft
W = W_hard - W_soft.detach() + W_soft
# Recompute y with the STE weight (original y_soft used W_soft)
y = torch.matmul(x, W)
```

where:
- `W_hard = palette[g, argmax(logits)]` — the hard one-hot reconstruction (no gradient).
- `W_soft = Σ_k P[k] * palette[g, k]` — the Gumbel-Softmax blend (differentiable).
- `W_soft.detach()` is the same tensor as `W_soft` but with `requires_grad=False`.

**Goal:** Use `W_hard` in the forward pass (exact one-hot, preserves cos) while routing the gradient through `W_soft` (so the index_logits can be trained).

---

## 2. Mathematical proof of forward correctness

**Claim:** In the forward pass, `W = W_hard - W_soft.detach() + W_soft` evaluates to `W_hard`.

**Proof:**

In PyTorch's forward evaluation, `W_soft.detach()` returns a tensor whose *value* is identical to `W_soft` (the `.detach()` method only affects the autograd graph, not the tensor's data). Therefore:

```
W = W_hard - W_soft.detach() + W_soft
  = W_hard - W_soft + W_soft      (since W_soft.detach() ≡ W_soft in value)
  = W_hard + (W_soft - W_soft)
  = W_hard + 0
  = W_hard                                        ∎
```

**Corollary:** The output `y = x @ W` in the forward equals `x @ W_hard` — the exact hard reconstruction. The Gumbel-Softmax noise does NOT appear in the forward output. **This is the desired behavior:** the forward uses the crisp one-hot indices, so the cos similarity is preserved from the calibration step (cos ≈ 0.937 per the calibration log).

---

## 3. Mathematical proof of backward behavior

**Claim:** In the backward pass, the gradient flows ONLY through `W_soft`, not through `W_hard` or `W_soft.detach()`.

**Proof:**

Let `L` be the loss. In the backward, PyTorch computes `∂L/∂θ` for each parameter `θ` by chain rule through the autograd graph.

The graph for `W = W_hard - W_soft.detach() + W_soft` is:
- `W_hard` is constructed under `torch.no_grad()` (line `with torch.no_grad():` in the Python code), so it has NO gradient — `∂W_hard/∂θ = 0` for all `θ`.
- `W_soft.detach()` has `requires_grad=False`, so PyTorch treats it as a constant — `∂W_soft.detach()/∂θ = 0`.
- `W_soft` has gradients flowing to `logits` (via the softmax) and `palette` (directly).

Therefore:
```
∂W/∂θ = ∂W_hard/∂θ - ∂W_soft.detach()/∂θ + ∂W_soft/∂θ
      = 0 - 0 + ∂W_soft/∂θ
      = ∂W_soft/∂θ                                    ∎
```

And by chain rule:
```
∂L/∂θ = ∂L/∂W * ∂W/∂θ = ∂L/∂W * ∂W_soft/∂θ
```

where `∂L/∂W = grad_W` (the gradient of loss w.r.t. the W tensor, computed as `grad_W = x.T @ grad_y`).

**Corollary:** The gradient to `logits` flows through `W_soft` as if the forward had used `W_soft`. This is the "straight-through" property: forward uses `W_hard`, backward pretends it used `W_soft`.

---

## 4. Proof that the gradient formula is correct

**Claim:** The gradient formula `grad_logits[j, o, k] = grad_W[j, o] * P[j, o, k] * (c[k] - W_soft[j, o])` is mathematically correct.

**Proof:**

Recall:
- `W_soft[j, o] = Σ_k P[j, o, k] * c[k]` where `c[k] = palette[g, k]`.
- `P[j, o, k] = softmax(noisy_logit[j, o, k])` where `noisy_logit = (logit + gumbel) / tau`.

We want `∂L/∂logit[j, o, k]`. By chain rule:
```
∂L/∂logit[j, o, k] = ∂L/∂W_soft[j, o] * ∂W_soft[j, o]/∂logit[j, o, k]
                    = grad_W[j, o] * ∂W_soft[j, o]/∂logit[j, o, k]
```

Now compute `∂W_soft[j, o]/∂logit[j, o, k]`:
```
W_soft[j, o] = Σ_m P[j, o, m] * c[m]
∂W_soft[j, o]/∂logit[j, o, k] = Σ_m c[m] * ∂P[j, o, m]/∂logit[j, o, k]
```

The softmax derivative is:
```
∂P[m]/∂logit[k] = P[m] * (δ[m,k] - P[k])
```

where `δ[m,k]` is the Kronecker delta. Substituting:
```
∂W_soft[j, o]/∂logit[j, o, k] = Σ_m c[m] * P[m] * (δ[m,k] - P[k])
                               = c[k] * P[k] - P[k] * Σ_m c[m] * P[m]
                               = P[k] * (c[k] - W_soft[j, o])
```

Therefore:
```
∂L/∂logit[j, o, k] = grad_W[j, o] * P[j, o, k] * (c[k] - W_soft[j, o])    ∎
```

This matches the formula implemented in the CUDA kernel (`fused_lut_linear_soft_bwd_grad_logits_kernel`, lines 1364–1398 of `fused_lut_kernel.cu`) and the Python backward (line `grad_logits = (grad_W_f.unsqueeze(-1) * P_kno_f * (pal_pos - W_val.unsqueeze(-1)))...` in `fused_lut_linear_cuda.py`).

**The gradient formula is correct.** The issue is not the formula — it's the *magnitude* of the gradient at low temperature, which we analyze next.

---

## 5. Proof that the gradient vanishes at low temperature

**Claim:** As τ → 0, the Gumbel-Softmax gradient `grad_W * P[k] * (c[k] - W_soft)` approaches zero for ALL k.

**Proof:**

At low τ, the softmax becomes one-hot. Without loss of generality, let `k* = argmax_k noisy_logit[k]` be the argmax index. Then:
```
P[k*] = 1 - ε_1
P[k] = ε_k  for k ≠ k*
```
where `ε_k` are small positive numbers (exponentially small in `(noisy_logit[k*] - noisy_logit[k]) / τ`).

The reconstructed weight is:
```
W_soft = Σ_k P[k] * c[k] = c[k*] * (1 - ε_1) + Σ_{k≠k*} ε_k * c[k] ≈ c[k*]
```

(to first order in ε, assuming the c[k] values are bounded).

Now the gradient:
```
grad_logits[k] = grad_W * P[k] * (c[k] - W_soft)
```

For `k = k*`:
```
grad_logits[k*] = grad_W * (1 - ε_1) * (c[k*] - W_soft)
                ≈ grad_W * 1 * (c[k*] - c[k*])
                = grad_W * 0
                = 0
```

For `k ≠ k*`:
```
grad_logits[k] = grad_W * ε_k * (c[k] - c[k*])
```

This is small (proportional to `ε_k`), but non-zero. However, in fp16/fp32 storage, when `ε_k` underflows to zero (which happens when `noisy_logit[k*] - noisy_logit[k] > 18` in fp16, or `> 60` in fp32), `P[k]` is stored as exactly zero, and the gradient is computed as exactly zero.

**At τ = 0.1 with logits = ±10:**
- `noisy_logit = (logit + gumbel) / τ`. With gumbel in [-3, +16] and logit in [-10, 10]:
- Best case: `(10 + 16) / 0.1 = 260`.
- Worst competing case: `(-10 + 0) / 0.1 = -100`.
- Difference: 360.
- `exp(-360) ≈ 0` in fp16 (and even in fp32, `exp(-360) ≈ 1e-156` which underflows to 0).

So `P[k]` is stored as exactly `[0, 0, 1, 0]` (or similar one-hot), and the gradient is computed as exactly zero for all k.

**Therefore, at low τ, the Gumbel-Softmax gradient vanishes, and the index_logits do not train.** This is consistent with the developers' own empirical observation: *"all 25 index_logits grads are 0.0."*

This is a mathematical property of Gumbel-Softmax at low temperature, not a bug in the implementation. ∎

---

## 6. The STE assumption: is it valid?

The STE assumes `∂L/∂W_soft ≈ ∂L/∂W_hard`. This is valid ONLY when `W_soft ≈ W_hard`, i.e., when the temperature is low enough that the softmax is (approximately) one-hot but high enough that `P[k]` for non-argmax entries is non-zero in storage.

**The "sweet spot" for STE validity:**

- At τ = 2.0: softmax is smooth, `W_soft ≠ W_hard` significantly. STE assumption is INVALID — gradient direction is wrong.
- At τ = 0.5: softmax is moderately peaked. `W_soft ≈ W_hard` but with non-trivial soft blending. STE assumption is approximately valid. Gradients flow.
- At τ = 0.1: softmax is one-hot. `W_soft = W_hard` exactly. STE assumption is valid in direction but gradient magnitude is zero.
- At τ = 0.01: same as τ = 0.1 but more extreme.

**The fundamental trade-off:** Low τ makes the forward correct (W_hard = exact one-hot) but kills the gradient. High τ makes the gradient flow but the forward is wrong (W_soft ≠ W_hard, so the output is not what we want).

The repo anneals τ from 2.0 → 0.1 over 4000 steps. This means:
- Steps 0-2000 (τ > 1.0): STE assumption is invalid. Gradient direction is wrong. Indices drift.
- Steps 2000-4000 (τ in [0.5, 1.0]): STE is approximately valid. Gradient flows. Indices train.
- Steps 4000-8300 (τ = 0.1): Gradient vanishes. Indices frozen.

So the effective training window for indices is only ~2000 steps (out of 8300), and even then, the gradient direction is approximate.

---

## 7. Alternative STE formulations

### 7.1 Vanilla STE (Bengio et al. 2013)

The original STE simply uses `W_hard` in forward and `∂L/∂W_hard ≈ ∂L/∂W_soft` in backward, WITHOUT the `W_hard - W_soft.detach() + W_soft` trick:

```python
# Forward: use W_hard
y = x @ W_hard
# Backward: pretend we used W_soft
# (PyTorch's autograd derives this from the forward graph)
```

In practice, this is implemented by overriding the backward of a custom `torch.autograd.Function`:

```python
class HardArgmaxSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, palette, group_size):
        # Hard argmax
        idx = logits.argmax(dim=0)
        W_hard = palette[g, idx]
        ctx.save_for_backward(logits, palette, idx)
        return W_hard
    @staticmethod
    def backward(ctx, grad_W):
        logits, palette, idx = ctx.saved_tensors
        # Compute gradient as if forward was soft
        P = softmax(logits / tau)
        # grad_logits = grad_W * P * (c - W_soft)  -- same as current
        return grad_logits, grad_palette, None
```

**This is functionally equivalent to the repo's STE** — both use `W_hard` in forward and route gradient through `W_soft` in backward. The repo's formulation (`W_hard - W_soft.detach() + W_soft`) is just a clever way to express this using standard PyTorch ops without a custom Function.

### 7.2 Gumbel-Softmax with persistent temperature (Jang et al. 2017)

The original Gumbel-Softmax paper recommends **NOT annealing τ to 0**. Instead, keep τ at a fixed moderate value (e.g., τ = 0.5) where the gradient flows and the softmax is approximately one-hot.

**Implementation:**
```python
# Fixed tau = 0.5 throughout training
W_soft = gumbel_softmax(logits, tau=0.5, hard=False)
# No STE — just use W_soft directly
y = x @ W_soft
```

**Trade-off:** The forward is NOT exactly W_hard — it's a soft blend. The cos similarity will be slightly lower than the hard path, but the gradient flow is maintained throughout training. This is the approach used by AQLM during the early phases of training (they switch to hard argmax only at the very end).

### 7.3 Gumbel-Softmax with hard=True (PyTorch built-in)

PyTorch's `torch.nn.functional.gumbel_softmax(logits, tau=0.5, hard=True)` returns a one-hot tensor in forward but routes gradient through the soft softmax in backward. This is exactly the STE behavior, implemented as a built-in:

```python
import torch.nn.functional as F
# One-hot in forward, soft gradient in backward
P_hard = F.gumbel_softmax(logits, tau=0.5, hard=True)
W = (P_hard * palette).sum(dim=0)  # = palette[argmax] in forward
y = x @ W
```

**Advantage over the repo's STE:** PyTorch's implementation handles the gradient routing correctly without the `W_hard - W_soft.detach() + W_soft` arithmetic (which can introduce small numerical errors when `W_soft.detach()` and `W_soft` have slightly different values due to recompute).

### 7.4 Straight-Through Gumbel-Softmax with temperature floor

A common practical recipe (from the AQLM and QTIP papers) is to anneal τ but never below a floor (e.g., τ_min = 0.5):

```python
tau = max(0.5, tau_init * (1.0 - step / anneal_steps))
```

This ensures the gradient always flows. The trade-off is that the final W_hard (after training) may differ from the final W_soft (since τ is not 0), so a final "argmax" step is needed:

```python
# After training: extract hard indices
final_indices = logits.argmax(dim=0)
```

### 7.5 Reinforce-style gradient (REINFORCE with baseline)

Instead of the Gumbel-Softmax relaxation, use the REINFORCE algorithm to estimate the gradient through the hard argmax:

```python
# Sample hard index
idx = Categorical(logits=logits).sample()
W_hard = palette[g, idx]
y = x @ W_hard
loss = compute_loss(y, y_teacher)
# REINFORCE gradient
loss_surrogate = -log_prob(logits, idx) * (loss - baseline).detach()
```

**Advantage:** No vanishing gradient at low τ. **Disadvantage:** High variance — requires many samples and a good baseline. Not commonly used for LLM quantization.

### 7.6 AQLM-style direct STE through additive codebook

AQLM avoids the Gumbel-Softmax entirely. Instead, it uses a direct STE on the additive codebook:

```python
# Forward: hard additive codes (via beam search)
codes = beam_search_best_codes(x, codebooks)
W_hard = sum(codebooks[k][codes[k]] for k in range(K))
y = x @ W_hard
# Backward: STE — pretend the beam search was differentiable
# grad_codes = grad_W * codebook_entries (computed directly, no softmax)
```

This avoids the vanishing-gradient problem entirely because there is no softmax — the gradient is computed directly through the codebook entries. **This is why AQLM can train at any "temperature" (it doesn't use one).**

### 7.7 Concrete distribution (Maddison et al. 2017)

The Concrete distribution (a.k.a. RelaxedOneHotCategorical) is a continuous relaxation of the categorical distribution. It's mathematically equivalent to Gumbel-Softmax but uses a different parameterization that can be more numerically stable:

```python
# Concrete relaxation
u = Uniform(0, 1).sample(shape)
concrete_logit = (logits + log(u) - log(1-u)) / tau
P = softmax(concrete_logit)
```

**Advantage:** The `log(u) - log(1-u)` is the logistic transform, which has a finite range (unlike `log(u)` in Gumbel which can be `-inf`). This avoids extreme values in the noisy logits.

---

## 8. Recommended STE fix for the qwen-palettize repo

Based on the analysis, the recommended fix is a combination of:

1. **Floor τ at 0.5** (not 0.1) — keeps the gradient flowing throughout training.
2. **Switch to PyTorch's built-in `gumbel_softmax(hard=True)`** — handles the STE correctly.
3. **Use fp32 for P, grad_logits** — avoid the fp16 underflow issues.
4. **Add a final argmax step** — after training, extract hard indices for inference.
5. **Consider switching to AQLM-style direct STE** — eliminates the vanishing-gradient problem entirely, but requires rearchitecting the codebook.

Concrete code patch (pseudocode):

```python
# BEFORE (current, with vanishing gradient):
W = W_hard - W_soft.detach() + W_soft
y = torch.matmul(x, W)

# AFTER (fixed, with persistent gradient):
import torch.nn.functional as F
# Fixed tau = 0.5, hard=True for one-hot forward, soft gradient backward
P_hard = F.gumbel_softmax(
    logits.permute(1, 2, 0),  # (K, N, 4) -> (N, K, 4)? check dim
    tau=0.5,
    hard=True,
).permute(2, 0, 1)  # back to (4, K, N)
W = (P_hard * palette[g]).sum(dim=0)  # = palette[argmax] in forward, soft grad in backward
y = torch.matmul(x, W)
```

**Pseudocode for AQLM-style direct STE** (longer-term fix):

```python
# Forward: beam search for best codes (no softmax)
codes = beam_search(x, codebooks, beam_width=4)  # (K, n_blocks)
W_hard = sum(codebooks[k][codes[k]] for k in range(K))
y = x @ W_hard

# Backward: STE — grad flows directly through codebook entries
# (Implement as a custom torch.autograd.Function)
class AdditiveVQSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, codebooks, codes):
        W = sum(codebooks[k][codes[k]] for k in range(K))
        ctx.save_for_backward(x, codebooks, codes)
        return x @ W
    @staticmethod
    def backward(ctx, grad_y):
        x, codebooks, codes = ctx.saved_tensors
        grad_W = x.T @ grad_y
        # Gradient to codebook entries (direct, no softmax)
        grad_codebooks = [scatter_add(grad_W, codes[k], dim=...) for k in range(K)]
        # Gradient to x
        grad_x = grad_y @ W.T
        return grad_x, grad_codebooks, None
```

---

## 9. Summary

| Property | Current STE | Recommended fix |
|----------|-------------|-----------------|
| Forward | `W_hard` (exact) ✓ | `W_hard` (exact) ✓ |
| Backward | Through `W_soft` ✓ | Through `W_soft` ✓ |
| Gradient at τ=0.1 | **Zero** ✗ | Non-zero (τ floored at 0.5) ✓ |
| Gradient at τ=2.0 | Wrong direction ✗ | Correct direction ✓ |
| Numerical stability | fp16 P, fp16 grad_logits ✗ | fp32 P, fp32 grad_logits ✓ |
| PyTorch integration | Manual arithmetic | Built-in `gumbel_softmax(hard=True)` |
| Long-term | Gumbel-Softmax | AQLM-style direct STE (no softmax) |

**Mathematical conclusion:** The STE formula is mathematically correct (sections 2-4). The gradient computation is correct (section 4). The vanishing-gradient at low τ is a mathematical property of Gumbel-Softmax, not a bug (section 5). The fix is to floor τ at a value where the gradient flows (0.5 or higher) and to use fp32 storage for P and grad_logits to avoid underflow.

**Practical conclusion:** Even with the STE fix, the scalar 2-bit LUT scheme has a fundamental cos ceiling of ~0.94. The STE fix enables index training, but the indices can only choose among 4 palette entries per group — they cannot break the information-theoretic limit. Breaking cos 0.99 requires switching to VQ codebooks (see 04_literature_comparison.md).
