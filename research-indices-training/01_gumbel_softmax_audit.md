# 01 — Gumbel-Softmax Implementation Audit

**Scope:** Line-by-line audit of our `CUDAFusedLUTLinearSoft` Gumbel-Softmax relaxation against the canonical formulation of Jang, Gu & Poole (2017) ([arXiv:1611.01144](https://arxiv.org/abs/1611.01144)) and the closely-related "no-Gumbel" variant of Wang et al. (CVPR 2022, "Learnable Lookup Table for Neural Network Quantization"). The goal is to determine *exactly* where our implementation diverges from the textbook recipe and whether each divergence contributes to the observed pathology: `grad_logits` ≈ 2e-4 at τ=2.0 with logits=±1, and the cos plateau at 0.95.

---

## 1. Reference formulation (Jang et al. 2017)

The Jang et al. paper defines the Concrete / Gumbel-Softmax distribution as follows. For a K-way categorical variable with class probabilities `π = (π_1, ..., π_K)`, draw K i.i.d. Gumbel(0,1) samples `g_k = -log(-log(u_k))` where `u_k ~ Uniform(0,1)`, then form the soft sample

```
y_k = exp( (log π_k + g_k) / τ ) / Σ_{k'} exp( (log π_{k'} + g_{k'}) / τ )
```

Three properties are essential:

1. **Marginal correctness.** As `τ → 0`, `argmax(y) → Categorical(π)`. At `τ → ∞`, `y → Uniform(1/K)`.
2. **Differentiability.** `y` is a smooth function of `log π` for any finite `τ`, so backprop works.
3. **Variance.** The Gumbel noise `g_k` injects gradient variance proportional to `π_k(1-π_k)/τ²`. This variance is the price of marginal-correctness; it is *not* free.

The paper's Section 3.2 explicitly proposes a **Straight-Through (ST) variant**: in the forward pass, use the hard one-hot `argmax` sample; in the backward pass, use the soft `y`. The ST bridge is `y_hard = one_hot(argmax(y)); y_st = y_hard - y.detach() + y`. This is exactly the form our code uses (audit in §4 below).

The paper recommends annealing `τ` from `~1` down to `~0.1` to `0.5`, and notes that *lower* `τ` sharpens the distribution but increases gradient variance and risks saturation (one logit dominates → all others receive zero gradient).

---

## 2. Our implementation: file map

Our Gumbel-Softmax path is split across three files:

| Concern | File | Lines |
|---|---|---|
| Python `autograd.Function` (STE, save_for_backward, dispatch) | `scripts/fused_lut_linear_cuda.py` | 519–684 |
| C++ host wrapper (allocates outputs, calls launcher) | `scripts/fused_lut_linear_cuda.py` | 217–360 |
| CUDA kernels (Gumbel sampler, softmax, `compute_P_W`, `bwd_grad_logits`) | `scripts/fused_lut_kernel.cu` | 1249–1481 |
| Per-module logits init (±10) and soft-reset to ±1 | `scripts/qwen_model.py` | 120–130 |
| Resume-time ±1 re-init | `scripts/train_qwen.py` | 830–841 |
| τ schedule (linear 2.0 → 0.1 over 4000 steps) | `scripts/train_qwen.py` | 1034–1040 |
| Optimizer + LR for `index_logits` (FP32 master AdamW, lr=1e-2) | `scripts/train_qwen.py` | 580–597 |

The audit below cites line numbers from these files (as committed at `b82a6be`).

---

## 3. The Gumbel sampler — our LCG vs the spec

### 3.1 Where it lives

`fused_lut_kernel.cu` lines **1273–1285**:

```cpp
__device__ __forceinline__ float gumbel_sample(uint32_t seed, uint32_t idx) {
    uint32_t x = seed ^ (idx * 0x9E3779B9u);
    x ^= x >> 13;
    x = x * 1103515245u + 12345u;
    x ^= x >> 17;
    x = x * 1103515245u + 12345u;
    float u = (float)(x & 0xFFFFFFu) * (1.0f / 16777216.0f);   // [0, 1)
    u = fmaxf(u, 1e-7f);                            // avoid log(0)
    return -logf(-logf(u));                          // Gumbel(0, 1)
}
```

The sampler is a hand-rolled 32-bit Linear Congruential Generator (LCG) with two xorshifts, mixed with the canonical Fibonacci multiplier `0x9E3779B9` (the "golden ratio" constant from Knuth). It produces a `uint32_t`, takes the low 24 bits as a mantissa, scales to `[0,1)`, clamps to `[1e-7, 1)`, and applies the inverse-transform formula `-log(-log(u))` for Gumbel(0,1).

### 3.2 Audit findings

**Finding 1 (correct):** the inverse-transform `-log(-log(u))` is the textbook Gumbel(0,1) generator. ✓

**Finding 2 (correct but worth noting):** the `fmaxf(u, 1e-7f)` floor prevents `log(0)` in `logf(u)`. This is necessary but introduces a tiny bias: ~1 sample in 16M is clamped to `1e-7`, producing a Gumbel draw of `-log(-log(1e-7)) ≈ 2.6` instead of the unbounded tail. The bias is negligible for K=4 softmax (the clamped draw would still lose the softmax race in practice), but it does mean our `g_k` distribution is *slightly* truncated on the right tail.

**Finding 3 (concerning — correlation structure):** the LCG state `x` is *deterministic per* `(seed, idx)`. Inside one forward pass, the kernel calls `gumbel_sample(step_seed, idx * 4 + 0..3)` to draw the 4 noises for a single `(j, o)` position (`fused_lut_kernel.cu` lines 1325–1328). The 4 calls share the same `seed` and differ only in `idx` by `+1`. Because the LCG is iterated *twice* (lines 1277–1280), the 4 outputs are decorrelated by the xorshift+multiply chain — empirically fine. **But across positions**, two threads with adjacent `idx` values produce adjacent LCG states, which after only 2 iterations of mixing may still be correlated in the low bits. A proper fix is to use `curand_normal` (which uses Box-Muller and a high-quality Philox4_32_10 counter-based generator) — see `curand_normal.h` in the CUDA toolkit. The current LCG is "good enough" for K=4 softmax but is **not cryptographically or statistically uniform**; this matters because Jang et al.'s marginal-correctness guarantee assumes *i.i.d.* Gumbel draws.

**Finding 4 (concerning — seed determinism across resamples):** the seed is set globally by `_next_soft_step_seed()` (`fused_lut_linear_cuda.py` lines 510–516), incremented once per *forward call*. This means **all 25 `PalettizedLinear` modules in the super-block share the same step_seed** (each just increments the global counter). With 25 modules × ~10M positions each, the LCG produces a very long correlated stream. Again, empirically this is fine for the softmax saturations we see, but it means the Gumbel noise is not statistically independent across layers — a subtle violation of the Jang et al. assumption.

**Finding 5 (correct, important):** the kernel divides `(logits + gumbel) / tau` and applies softmax (`fused_lut_kernel.cu` lines 1324–1337). This matches the Jang et al. formula `exp((log π + g)/τ) / Σ ...` exactly, where `logits` plays the role of `log π`. The numerically-stable `softmax - max` subtraction is used (line 1331). ✓

---

## 4. The STE bridge — line-by-line

`fused_lut_linear_cuda.py` lines 580–596:

```python
# ── STE: Straight-Through Gumbel-Softmax ──────────────────────────
# Forward uses HARD weight: W_hard = palette[argmax(logits)]
# Backward flows through SOFT weight: W_soft (non-zero gradients)
# W = W_hard - W_soft.detach() + W_soft
#   forward value = W_hard (exact one-hot → cos preserved)
#   backward grad  = through W_soft (indices actually train)
with torch.no_grad():
    argmax_idx = logits.argmax(dim=0)  # (K, N) — hard index assignment
    group_idx = torch.arange(N, device=palette.device) // group_size
    group_per_col = group_idx.unsqueeze(0).expand(K, N)
    W_hard = palette[group_per_col.long(), argmax_idx.long()].to(W_soft.dtype)
W = W_hard - W_soft.detach() + W_soft
y = torch.matmul(x, W)
del y_soft
```

### 4.1 Audit findings

**Finding 6 (correct):** the STE bridge `W = W_hard - W_soft.detach() + W_soft` is the textbook Jang et al. ST-Gumbel-Softmax (their Eq. 8). Forward: `W == W_hard` exactly (the `W_soft.detach()` cancels with `-W_soft.detach()` in the forward because `W_soft - W_soft.detach() == 0` in forward mode). Backward: `dW/dlogits == dW_soft/dlogits` (because `W_hard` is detached). ✓

**Finding 7 (correct, important):** this is **why** the user observes "STE preserves cos=0.9473 at ALL tau values". The forward pass is the hard one-hot lookup; τ affects only the backward gradient. As long as `argmax(logits)` doesn't change during a single forward, the forward output is τ-independent. ✓

**Finding 8 (subtle bug — W_hard is recomputed in Python, not in the kernel):** the CUDA `fused_lut_linear_soft_fwd` kernel *already* produces `W_soft` (line 576: `y_soft, P, W_soft = mod.fused_lut_linear_soft_fwd(...)`), then Python runs `argmax` + gather to build `W_hard`, then recomputes `y = matmul(x, W)` (line 595). The `y_soft` from the kernel is discarded (line 596: `del y_soft`). **This is a wasted matmul** — `y_soft` was computed by cuBLAS at significant cost. The kernel could instead directly produce `W = W_hard - W_soft.detach() + W_soft` in CUDA, halving the matmul count. The current implementation does `matmul(x, W_soft)` *inside* the kernel call (via `torch.matmul(x, W)` at `fused_lut_linear_cuda.py:258`) and *then* `matmul(x, W)` again at line 595. **Twice the cuBLAS cost for the forward.**

**Finding 9 (correctness, low concern):** `argmax_idx = logits.argmax(dim=0)` (line 587) uses the *clean* logits without Gumbel noise. This is the deterministic-argmax ST variant (vs the stochastic-argmax variant that uses `argmax(noisy_logits)`). Both are valid; the deterministic variant has lower variance but loses the marginal-correctness guarantee. Given our K=4 / 2-bit setting, deterministic argmax is the right choice — it ensures the forward is reproducible and that small Gumbel noise doesn't cause spurious index flips.

---

## 5. Logits initialization: ±10 vs ±1

`qwen_model.py` lines 120–128:

```python
if use_soft_indices and not pre_transposed:
    K_dim, N_dim = indices.shape
    logits = torch.full((4, K_dim, N_dim), -10.0, dtype=torch.float16, device=device)
    for k in range(4):
        mask = (indices == k)
        logits[k][mask] = 10.0
    self.index_logits = nn.Parameter(logits)
```

`train_qwen.py` lines 836–841 (resume-time override):

```python
if mod.index_logits is not None:
    with torch.no_grad():
        mod.index_logits.data.fill_(-1.0)
        for k in range(4):
            mask = (indices == k)
            mod.index_logits.data[k][mask] = 1.0
        mod.index_logits.data = mod.index_logits.data.to(torch.float16)
```

### 5.1 Audit findings

**Finding 10 (the smoking gun):** at construction (`qwen_model.py:124`), logits are initialized to ±10. This corresponds to a softmax probability of `exp(20) / (exp(20) + 3·exp(0)) ≈ 4.85×10^8 / (4.85×10^8 + 3) ≈ 1.0` — i.e., **the softmax is numerically one-hot at any reasonable τ**. At τ=2.0 the logit gap of 20 becomes `20/2=10` in the exponent, still `exp(10)≈22026` vs `exp(0)=1`, so `P_winner ≈ 1 - 1.4e-4`. The gradient `∂L/∂logit_k = ∂L/∂W · P_k · (c_k - W)` (derivation in §3 of `02_ste_correctness.md`) is proportional to `P_k` for the non-winning classes, so `grad_logits ≈ 2e-4 × 1.4e-4 ≈ 3e-8` — well below the AdamW `eps=1e-8` floor. **The optimizer literally cannot move the logits.**

**Finding 11 (the patch):** `train_qwen.py:836-841` re-initializes logits to ±1 at resume time. With logit gap of 2 and τ=2.0, the exponent is `2/2=1`, so `P_winner = exp(1)/(exp(1)+3·exp(0)) ≈ 2.718/5.718 ≈ 0.475`, and `P_loser ≈ 0.175`. This is *not* one-hot — it's barely-more-than-uniform. Gradients flow freely, which is why `grad_logits` becomes "non-zero for 21/25 Linears, max_grad=2e-4" as the user observed. **However**, the forward now uses `W_soft ≈ 0.475·c_winner + 3·0.175·c_losers` — which is *not* the hard one-hot, so cos drops. The STE bridge fixes this in the forward (Finding 6), but the *backward* now sees a `W_soft` that is far from `W_hard`, so the "gradient direction" is misleading.

**Finding 12 (recommended):** a middle ground is `±3` (gap of 6). At τ=2.0, exponent is 3, so `P_winner = exp(3)/(exp(3)+3) ≈ 20/23 ≈ 0.87`, `P_loser ≈ 0.043`. The forward `W_soft` is now close to `W_hard` (within 13%), so the STE gradient direction is meaningful, while `P_loser` is large enough (0.043) to give `grad_logits ≈ 2e-4 × 0.043 ≈ 9e-6` — still small but ~300× larger than the ±10 case. We explore this in `07_recommendations.md`.

---

## 6. The backward kernel — `grad_logits` formula

`fused_lut_kernel.cu` lines 1364–1398 (the `fused_lut_linear_soft_bwd_grad_logits_kernel`):

```cpp
__global__ void fused_lut_linear_soft_bwd_grad_logits_kernel(
    const float* grad_W, const __half* P, const __nv_bfloat16* palette,
    __half* grad_logits, int K, int N, int group_size
) {
    // ... load P[k, j, o] and palette[g, k] for k=0..3 ...
    float W_val = c0 * p0 + c1 * p1 + c2 * p2 + c3 * p3;
    grad_logits[0 * plane_size + idx] = __float2half(dW * p0 * (c0 - W_val));
    grad_logits[1 * plane_size + idx] = __float2half(dW * p1 * (c1 - W_val));
    grad_logits[2 * plane_size + idx] = __float2half(dW * p2 * (c2 - W_val));
    grad_logits[3 * plane_size + idx] = __float2half(dW * p3 * (c3 - W_val));
}
```

### 6.1 Audit findings

**Finding 13 (correct):** the formula `∂L/∂logit_k = (∂L/∂W) · P_k · (c_k - W)` is the correct gradient of `L = (1/2)·(W - target)²` where `W = Σ_k P_k · c_k` and `P = softmax(logits/τ)`. The full chain rule is:

```
∂L/∂logit_k = ∂L/∂W · ∂W/∂logit_k
            = ∂L/∂W · Σ_{k'} (∂W/∂P_{k'}) · (∂P_{k'}/∂logit_k)
            = ∂L/∂W · Σ_{k'} c_{k'} · P_{k'} (δ_{k,k'} - P_k)
            = ∂L/∂W · [c_k · P_k - P_k · Σ_{k'} c_{k'} · P_{k'}]
            = ∂L/∂W · P_k · (c_k - W)
```

This is the standard softmax-Jacobian-vector product. ✓ Note that this formula is **independent of τ** because the kernel receives `P` *post-softmax*, and `P` already absorbed the `1/τ` scaling during the forward. The τ-dependence enters only through the magnitude of `P_k` (low τ → one-hot → small `P_loser` → tiny gradient).

**Finding 14 (potential bug — fp16 storage of grad_logits):** `grad_logits` is allocated as fp16 (`torch.empty({4, K, N}, P.options())` at `fused_lut_linear_cuda.py:286` where `P.options()` is fp16). The product `dW * P_k * (c_k - W_val)` is computed in fp32 (good), but the final `__float2half` truncates to fp16. For the typical magnitudes `|dW| ~ 1e-3`, `|P_k| ~ 0.1`, `|c_k - W| ~ 0.05`, the product is `~5e-6` — well within fp16 normal range (`6.1e-5` minimum normal). **But** for `|dW| ~ 1e-5` (early training, small gradient), the product can fall into fp16 subnormal range (`5.96e-8` minimum subnormal), where it loses ~3 bits of precision. For an `AdamW` step that depends on `grad/√(v + eps)`, this precision loss is acceptable but accumulates over many steps.

**Finding 15 (concerning — the kernel is unused):** the audit of `backward()` in `fused_lut_linear_cuda.py:609-684` shows that the **PyTorch fallback path** is used (lines 651-676), not the CUDA kernel. The comment at lines 628-640 explains:

> The pure-CUDA fused kernel (IX.b) was 3.8× SLOWER than PyTorch vectorized ops because of strided global memory access to P (4, K, N). PyTorch's vectorized ops use coalesced memory access patterns and are much faster.

So the `fused_lut_linear_soft_bwd_grad_logits_kernel` we just audited is **dead code** in the current training path. The actual gradient computation is:

```python
# fused_lut_linear_cuda.py lines 664-676
grad_W_f = grad_W.float()
P_kno_f = P.permute(1, 2, 0).float()
g_idx = torch.arange(N, device=x.device) // GS
pal_pos = palette[g_idx.long()].unsqueeze(0).expand(K, N, 4).float()
W_val = (P_kno_f * pal_pos).sum(dim=-1)
grad_logits = (
    grad_W_f.unsqueeze(-1) * P_kno_f * (pal_pos - W_val.unsqueeze(-1))
).to(torch.float16).permute(2, 0, 1).contiguous()
```

This is mathematically identical to the kernel (`grad_W · P · (c - W)`), just done in PyTorch with fp32 intermediates. The audit in §6.1 above therefore applies to the *active* code path. ✓

---

## 7. The τ schedule

`train_qwen.py` lines 1034–1040:

```python
if use_soft_indices:
    tau = max(tau_final, tau_init * (1.0 - global_step / tau_anneal_steps))
    for name, mod in student.named_modules():
        if hasattr(mod, 'tau'):
            mod.tau = tau
```

With `tau_init=2.0`, `tau_final=0.1`, `tau_anneal_steps=4000` (line 1241-1245), this is **linear** annealing over 4000 steps.

### 7.1 Audit findings

**Finding 16 (deviation from Jang et al.):** Jang et al. recommend an *exponential* schedule `τ(t) = τ_0 · exp(-λt)` (or geometric, equivalently), not linear. Linear annealing spends too long at low τ (where gradients vanish) and not enough time at high τ (where exploration happens). An exponential schedule `τ(t) = 2.0 · exp(-t · log(20)/4000)` would reach `τ=0.4` at step 2000 (instead of `τ=1.05` with linear), giving more time at the high-gradient regime.

**Finding 17 (deviation from LLT):** Wang et al. (LLT, CVPR 2022) anneal `τ: 1 → 1e-3` over 30–50 epochs, also exponentially. Our `2.0 → 0.1` over 4000 steps is far more aggressive — the final `τ=0.1` leaves `P_loser ≈ exp(-10) ≈ 4.5e-5` for ±1 logits, so gradients are essentially zero by step 4000. After step 4000, `τ=0.1` is held fixed, and the indices are *effectively frozen* by the softmax saturation. We analyze this in detail in `04_tau_schedule.md`.

---

## 8. Where we diverge from Jang et al. (summary)

| Aspect | Jang et al. (2017) | Our implementation | Verdict |
|---|---|---|---|
| Softmax formulation | `exp((log π + g)/τ) / Σ ...` | identical (line 1325) | ✓ |
| Gumbel noise source | `curand_uniform` or `curand_normal` | custom LCG (line 1274) | ⚠ weaker statistical quality |
| STE bridge | `y_hard - y_soft.detach() + y_soft` | identical (line 593) | ✓ |
| τ schedule | exponential `1 → 0.1` | linear `2.0 → 0.1` (line 1036) | ⚠ less time in high-grad regime |
| Logit init scale | small random or zero-mean | ±10 (then ±1 at resume) | ⚠ ±10 saturates immediately |
| Argmax tie-breaking | `argmax(noisy_logits)` (stochastic) | `argmax(logits)` (deterministic, line 587) | ✓ reasonable for K=4 |
| Gradient variance | inherent to GS, controlled by τ | identical | ✓ |
| K (category count) | typically 10–100 (MNIST/CIFAR) | K=4 (2-bit) | ⚠ K=4 → P_loser has small headroom |

---

## 9. Where we diverge from LLT (CVPR 2022)

LLT drops the Gumbel noise entirely (uses plain softmax of negative L2 distances), adds the `1/√(N_i)` per-codebook gradient rescaling, and anneals `τ: 1 → 1e-3` exponentially over 30+ epochs.

| Aspect | LLT | Our implementation | Verdict |
|---|---|---|---|
| Gumbel noise | none (deterministic softmax) | yes (LCG Gumbel) | ⚠ we add variance without clear benefit at K=4 |
| Softmax input | `-||w - c_k||² / τ` (distance-based) | `(logits + g)/τ` (free logits) | different parameterization |
| Gradient rescaling | `1/√(N_i)` per codebook entry | none | ⚠ missing — see `03_gradient_flow_analysis.md` |
| τ schedule | exponential `1 → 1e-3` over 30-50 epochs | linear `2 → 0.1` over 4000 steps | ⚠ too aggressive |
| Codebook | learned by backprop | learned by backprop (palette is trainable) | ✓ same |
| STE bridge | yes | yes | ✓ same |

The single most consequential divergence is the **absence of the `1/√(N_i)` rescaling**. Without it, frequently-used codebook entries (the dominant `argmax` for many positions in a group) accumulate disproportionately large `v` (second moment) in AdamW, while rarely-used entries get small `v`, so `lr_eff = lr / √(v + eps)` *underweights* the dominant direction. This is the *opposite* of what we want — we want the dominant direction to *keep* training (it represents the bulk of the cost), but AdamW's `1/√v` slows it down. The `1/√(N_i)` rescaling corrects for this by *boosting* the gradient on underused entries, encouraging exploration. We derive the exact formula in `03_gradient_flow_analysis.md` §4.

---

## 10. Conclusions of the audit

Our Gumbel-Softmax implementation is **structurally correct** — the STE bridge, the softmax formula, the gradient computation, and the kernel-vs-PyTorch path all match the canonical recipe. The pathology (gradients ~2e-4, cos plateaus at 0.95) is *not* due to a bug in the Gumbel-Softmax math.

The pathology is due to **three independent hyperparameter / structural choices**, all of which depress `grad_logits`:

1. **Logits init at ±10** (`qwen_model.py:124`) saturates the softmax to one-hot at construction time. Even after the ±1 reset (`train_qwen.py:837-841`), the *gap* of 2 is small enough that `P_loser ≈ 0.175` at τ=2.0 — gradients are non-zero but small.

2. **Linear τ schedule 2.0 → 0.1** (`train_qwen.py:1036`) spends too long at low τ (where `P_loser` vanishes) and not enough at high τ (where exploration happens). An exponential schedule reaching `τ=0.5` at step 4000 would keep `P_loser > 0.01` throughout.

3. **Missing `1/√(N_i)` gradient rescaling** (no code — simply absent) means AdamW's adaptive learning rate *underweights* the dominant codebook entries, the opposite of what LLT prescribes.

Additionally, two minor inefficiencies:

4. The CUDA `bwd_grad_logits` kernel is **dead code** (the PyTorch fallback is used); the kernel could be revived if P were stored in a coalesced `(K, N, 4)` layout instead of `(4, K, N)`.

5. The forward does **two matmuls** (`y_soft` from kernel + `y` from STE recompute) when one would suffice; the kernel could compute `W = W_hard - W_soft.detach() + W_soft` directly.

These findings flow into the concrete code patches in `07_recommendations.md`. The mathematical proof of STE correctness and the derivation of the `1/√(N_i)` rescaling are in `02_ste_correctness.md` and `03_gradient_flow_analysis.md` respectively.

---

## 11. References

1. Jang, E., Gu, S., Poole, B. *Categorical Reparameterization with Gumbel-Softmax.* ICLR 2017. [arXiv:1611.01144](https://arxiv.org/abs/1611.01144)
2. Wang, L., Dong, X., Wang, Y., Liu, L., An, W., Guo, Y. *Learnable Lookup Table for Neural Network Quantization.* CVPR 2022. [OpenAccess](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Learnable_Lookup_Table_for_Neural_Network_Quantization_CVPR_2022_paper.html)
3. Maddison, C. J., Mnih, A., Teh, Y. W. *The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables.* ICLR 2017. [arXiv:1611.00712](https://arxiv.org/abs/1611.00712)
4. Bengio, Y., Léonard, N., Courville, A. *Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation.* arXiv 2013. [arXiv:1308.3432](https://arxiv.org/abs/1308.3432)
5. Cardinaux, F., Uhlich, S., et al. *Iteratively Training Look-Up Tables for Network Quantization (LUT-Q).* NeurIPS workshop 2018. [arXiv:1811.05355](https://arxiv.org/abs/1811.05355)

*File citations refer to commit `b82a6be` of the qwen-palettize repository, specifically `scripts/fused_lut_linear_cuda.py`, `scripts/fused_lut_kernel.cu`, `scripts/qwen_model.py`, and `scripts/train_qwen.py`.*
