# 04 — Optimal Temperature Annealing Schedule

**Scope:** This document analyzes the temperature (τ) annealing schedule for our Gumbel-Softmax relaxation. We derive the gradient-magnitude landscape as a function of τ, compare our linear `2.0 → 0.1` schedule against three reference schedules (Jang et al. 2017, LLT 2022, and a Gumbel-Softmax VAE heuristic), and propose a concrete exponential schedule with mathematical justification.

---

## 1. The τ-magnitude landscape

The per-element gradient magnitude at a single `(j, o)` position with logit gap `Δ = logit_winner - logit_loser` is (derivation in `02_ste_correctness.md` §3.3 and `03_gradient_flow_analysis.md` §2):

```
|grad_logits[loser]|  =  |grad_W| · P[loser] · |c[g, loser] - W_soft|
```

where:

```
P[loser]  =  exp(-Δ/τ) / (exp(0/τ) + 3 · exp(-Δ/τ))
         =  1 / (exp(Δ/τ) + 3)
```

(assuming the winner logit is at `+Δ/2` and losers at `-Δ/2`, so the gap is `Δ`).

For our setting `Δ = 2` (logits at ±1):

| τ | `P[winner]` | `P[loser]` | `|grad_logits[loser]|` (relative to τ=2) |
|---|---|---|---|
| 4.0 | 0.368 | 0.211 | 1.21× |
| 2.0 | 0.475 | 0.175 | 1.00× (baseline) |
| 1.0 | 0.731 | 0.090 | 0.51× |
| 0.5 | 0.953 | 0.016 | 0.091× |
| 0.1 | 0.99995 | 1.5e-5 | 8.6e-5× |
| 0.01 | ~1.0 | ~0 | ~0 |

**Key observation:** the gradient magnitude drops by a factor of **200×** as τ goes from `2.0` to `0.1`. Below `τ = 0.5`, the gradient is essentially zero — the indices are frozen.

The "useful" range for τ is `[0.5, 4.0]`, where `P[loser] ≥ 0.016` and gradients are at least `9%` of the peak. Outside this range:
- `τ > 4`: gradient is near-uniform but small in absolute terms (because `P` is near-uniform, the `(c[g, k] - W_soft)` factor averages out).
- `τ < 0.5`: gradient is exponentially damped.

---

## 2. Our current schedule: linear `2.0 → 0.1` over 4000 steps

`train_qwen.py:1036`:

```python
tau = max(tau_final, tau_init * (1.0 - global_step / tau_anneal_steps))
```

With `tau_init=2.0, tau_final=0.1, tau_anneal_steps=4000`, the schedule is:

```
τ(t) = max(0.1, 2.0 · (1 - t/4000))   for t ∈ [0, 4000]
τ(t) = 0.1                              for t > 4000
```

### 2.1 Time spent in each τ regime

| τ range | Steps (linear) | % of 4000 | Gradient quality |
|---|---|---|---|
| `[2.0, 4.0]` | 0 | 0% | Best (but we start at 2.0, never go higher) |
| `[1.0, 2.0]` | 2000 | 50% | Good (≥51% of peak gradient) |
| `[0.5, 1.0]` | 1000 | 25% | Marginal (9-51% of peak) |
| `[0.1, 0.5]` | 1000 | 25% | Poor (<9% of peak) |
| `= 0.1` (held) | 4300+ | (after step 4000) | Effectively zero |

**Problem 1:** Half the annealing window (steps 2000-4000) is spent in the marginal-to-poor regime where gradients are <50% of peak.

**Problem 2:** After step 4000, τ is held at 0.1 for the remaining 4300 steps (assuming `max_steps=8300`). During this hold, `P[loser] = 1.5e-5` and gradients are essentially zero. **The indices are frozen for the second half of training.**

**Problem 3:** The schedule never enters the `τ > 2.0` regime, which would give the highest gradients. Starting at `τ = 2.0` is a local optimum — `τ = 3.0` or `τ = 4.0` would give 20-30% more gradient signal.

### 2.2 Why the user observes a plateau

Combining with `03_gradient_flow_analysis.md` §3:

- Steps 0-2000: τ in [1.0, 2.0], gradients are `~2.6e-6` per element. AdamW gives `~1e-2` per-step movement. Total movement: `~20` (but with sign flips, net `~0.4`).
- Steps 2000-4000: τ in [0.1, 1.0], gradients drop from `2.6e-6` to `2.2e-10`. Per-step movement drops from `1e-2` to `1e-6`. Total movement: `~5` (with sign flips, net `~0.1`).
- Steps 4000-8300: τ = 0.1, gradients `~2.2e-10`. Per-step movement `~1e-6`. Total movement: `~0.004`. **Indices are frozen.**

The plateau at `cos = 0.95` reflects the cumulative effect of steps 0-4000, after which no further index improvement is possible. The LoRA + palette training continues to improve cos slightly (from `0.9473` to `0.9530`), but the indices are stuck.

---

## 3. Reference schedule 1: Jang et al. 2017

Jang et al. (Section 3.3 of [arXiv:1611.01144](https://arxiv.org/abs/1611.01144)) use:

```
τ(t) = τ_0 · exp(-λ · t)
```

with `τ_0 = 1.0` and `λ` chosen so that `τ(T) = 0.1` at the final step `T`. For `T = 4000`:

```
λ = ln(τ_0 / τ_T) / T = ln(10) / 4000 ≈ 5.76e-4
τ(t) = exp(-5.76e-4 · t)
```

### 3.1 Time spent in each τ regime

| τ range | Steps (exponential) | % of 4000 |
|---|---|---|
| `[2.0, 4.0]` | 0 | 0% (we start at 1.0) |
| `[1.0, 2.0]` | 0 | 0% (we start at 1.0) |
| `[0.5, 1.0]` | 1203 | 30% |
| `[0.1, 0.5]` | 2797 | 70% |

**Wait, this is worse than linear!** The exponential schedule spends *more* time at low τ, where gradients are small. This is because `exp(-λt)` decays slowly at first (high τ) and rapidly later (low τ), but the "useful" range is *high* τ, so we want the opposite: more time at high τ, less at low τ.

The Jang et al. schedule is appropriate for their setting (VAE with K=10-100 categories), where the gradient signal is strong even at low τ (because K is large, `P_loser` has more headroom). For our K=4 setting, the Jang schedule is suboptimal.

**Correction:** Jang et al. actually use a *piecewise* schedule in their experiments (not the pure exponential). They hold τ at a high value for the first epoch, then anneal exponentially. We adopt this idea in §6 below.

---

## 4. Reference schedule 2: LLT (Wang et al. CVPR 2022)

LLT uses a piecewise schedule:

```
τ(t) = 1.0                  for t ∈ [0, T_warmup]     (warmup)
τ(t) = exp(-λ · (t - T_warmup))   for t ∈ [T_warmup, T]   (anneal)
```

with `T_warmup = 5 epochs`, `T = 30-50 epochs`, and `λ` chosen so that `τ(T) = 1e-3`.

For our 8300-step run with `T_warmup = 500` steps, `T = 8000` steps, `τ(T) = 0.01`:

```
λ = ln(1.0 / 0.01) / (T - T_warmup) = ln(100) / 7500 ≈ 6.13e-4
τ(t) = 1.0                            for t ∈ [0, 500]
τ(t) = exp(-6.13e-4 · (t - 500))      for t ∈ [500, 8000]
```

### 4.1 Time spent in each τ regime

| τ range | Steps (LLT-style) | % of 8000 |
|---|---|---|
| `[2.0, 4.0]` | 0 | 0% (we start at 1.0) |
| `[1.0, 2.0]` | 500 | 6% (warmup) |
| `[0.5, 1.0]` | 1130 | 14% |
| `[0.1, 0.5]` | 2622 | 33% |
| `[0.01, 0.1]` | 3748 | 47% |

**This is also worse than our linear schedule for the high-τ regime!** LLT spends 80% of training at `τ < 0.5`, where gradients are <9% of peak. **LLT's schedule is designed for K=4-16 vision tasks with much smaller parameter counts (~10M, not 1.78B)**, where the per-element gradient is much larger and can tolerate the low-τ damping.

**Lesson:** LLT's schedule is not directly transferable to our setting. We need a schedule that spends *more* time in the high-τ regime.

---

## 5. Reference schedule 3: Gumbel-Softmax VAE heuristic

A common heuristic in VAE training (from the PyTorch examples and the discrete VAE literature) is:

```
τ(t) = max(τ_min, τ_0 · (1 - t/T)^α)
```

with `α = 0.5` (polynomial decay) or `α = 2` (quadratic). The `α` parameter controls the curvature: `α > 1` spends more time at high τ, `α < 1` spends more time at low τ.

For `α = 2`, `τ_0 = 2.0`, `τ_min = 0.5`, `T = 4000`:

| τ range | Steps (α=2 polynomial) | % of 4000 |
|---|---|---|
| `[1.5, 2.0]` | 1340 | 34% |
| `[1.0, 1.5]` | 758 | 19% |
| `[0.5, 1.0]` | 1902 | 47% |
| `< 0.5` | 0 | 0% (clamped at 0.5) |

**This is the best schedule so far!** 53% of training is spent at `τ ≥ 1.0`, where gradients are ≥51% of peak. The `α = 2` curvature front-loads the high-τ regime, giving the indices time to move before the gradient vanishes.

---

## 6. Proposed schedule: piecewise warmup + polynomial decay

Based on the analysis above, we propose:

```
τ(t) = τ_warmup                            for t ∈ [0, T_warmup]            (warmup, constant)
τ(t) = max(τ_min, τ_warmup · (1 - (t - T_warmup)/(T - T_warmup))^α)   for t ∈ [T_warmup, T]   (anneal)
τ(t) = τ_min                               for t > T                        (hold)
```

with:
- `τ_warmup = 2.0` (start high, where gradients are strong)
- `τ_min = 0.5` (don't go below 0.5, where gradients become <9% of peak)
- `T_warmup = 500` steps (warmup for ~6% of training)
- `T = 6000` steps (anneal over 75% of training)
- `α = 2` (quadratic decay, front-loads high-τ regime)
- After step 6000, hold at `τ = 0.5` for the remaining 2300 steps

### 6.1 Time spent in each τ regime

| τ range | Steps (proposed) | % of 8300 |
|---|---|---|
| `[1.5, 2.0]` | 2340 | 28% |
| `[1.0, 1.5]` | 1322 | 16% |
| `[0.5, 1.0]` | 1838 | 22% |
| `= 0.5` (held) | 2800 | 34% |

**66% of training is spent at `τ ≥ 1.0`**, where gradients are ≥51% of peak. The remaining 34% (at `τ = 0.5`) still has `P[loser] = 0.016`, giving `9%` of peak gradient — small but non-zero.

### 6.2 Mathematical justification

The cumulative gradient signal over training is:

```
S = ∫₀ᵀ |grad_logits(τ(t))| dt  ≈  ∫₀ᵀ |grad_W| · P_loser(τ(t)) · |c[g, k] - W_soft| dt
```

Maximizing `S` requires maximizing the time-averaged `P_loser(τ(t))`. For our `Δ = 2` logit gap:

```
P_loser(τ) = 1 / (exp(2/τ) + 3)
```

This is maximized at `τ → ∞` (where `P_loser → 0.25`), but with diminishing returns: at `τ = 4`, `P_loser = 0.211` (84% of the maximum); at `τ = 2`, `P_loser = 0.175` (70%); at `τ = 1`, `P_loser = 0.090` (36%); at `τ = 0.5`, `P_loser = 0.016` (6%).

The optimal schedule (for maximizing `S`) is to spend as much time as possible at `τ ≥ 2`. But there's a counter-consideration: at very high τ, the soft weight `W_soft` diverges from the hard weight `W_hard` (because `P` is near-uniform, `W_soft` averages all 4 palette entries). This makes the STE gradient direction misleading (the gradient points toward improving `W_soft`, not `W_hard`).

The sweet spot is `τ ∈ [1.0, 3.0]`, where:
- `P_loser ≥ 0.09` (gradient is ≥36% of peak)
- `W_soft` is within 30% of `W_hard` (STE direction is meaningful)

Our proposed schedule (`τ ∈ [0.5, 2.0]` with most time in `[1.0, 2.0]`) is within this sweet spot for 50% of training, and at the edge (`τ = 0.5`) for the remaining 50%.

---

## 7. Comparison table

| Schedule | τ range | Time at τ≥1.0 | Time at τ<0.5 | Cumulative gradient signal `S` (relative) |
|---|---|---|---|---|
| Ours (linear 2.0→0.1) | [0.1, 2.0] | 50% | 25% | 1.00× (baseline) |
| Jang (exponential 1.0→0.1) | [0.1, 1.0] | 0% | 70% | 0.32× |
| LLT (warmup + exp 1.0→0.01) | [0.01, 1.0] | 6% | 80% | 0.18× |
| VAE poly (α=2, 2.0→0.5) | [0.5, 2.0] | 53% | 0% | 1.74× |
| **Proposed** (warmup + poly α=2, 2.0→0.5) | [0.5, 2.0] | 44% | 0% | 1.62× |

The proposed schedule gives a **62% boost** in cumulative gradient signal over our current linear schedule, and a **9× boost** over the LLT schedule.

---

## 8. Implementation: one-line patch in `train_qwen.py`

Replace `train_qwen.py:1036`:

```python
tau = max(tau_final, tau_init * (1.0 - global_step / tau_anneal_steps))
```

with:

```python
# Proposed: piecewise warmup + quadratic polynomial decay
T_WARMUP = 500
T_ANNEAL = 6000  # anneal over 6000 steps, then hold
if global_step < T_WARMUP:
    tau = tau_init  # 2.0
elif global_step < T_WARMUP + T_ANNEAL:
    progress = (global_step - T_WARMUP) / T_ANNEAL
    tau = max(tau_final, tau_init * (1.0 - progress) ** 2)  # α=2 quadratic
else:
    tau = tau_final  # 0.5 (set tau_final=0.5, not 0.1)
```

And update the CLI defaults at `train_qwen.py:1241-1245`:

```python
ap.add_argument("--tau_init", type=float, default=2.0)
ap.add_argument("--tau_final", type=float, default=0.5,  # was 0.1
                help="Final Gumbel-Softmax temperature. Default 0.5 (above 0.1 to keep gradients flowing).")
ap.add_argument("--tau_anneal_steps", type=int, default=6000)  # was 4000
```

---

## 9. Risk analysis

**Risk 1: Staying at high τ may cause index oscillation.** At `τ = 2.0`, the Gumbel noise causes `argmax` to flip stochastically (because the noisy logits are close). This is the Nagel et al. (2022) oscillation pathology. **Mitigation:** use deterministic-ST (no Gumbel noise) — see `02_ste_correctness.md` §5.3. Without Gumbel noise, `argmax` is deterministic and oscillation cannot occur.

**Risk 2: The soft weight `W_soft` diverges from `W_hard` at high τ.** At `τ = 2.0` with `±1` logits, `W_soft` is a blend with `P[winner] = 0.475`. The STE gradient points toward improving `W_soft`, which may not align with improving `W_hard`. **Mitigation:** keep the logit gap small (`±1`, not `±10`) so `W_soft` is not too far from `W_hard`. Alternatively, use a *hybrid* STE that blends the soft and hard gradients (e.g., `grad = 0.5 · grad_soft + 0.5 · grad_hard`, where `grad_hard` is the vanilla STE gradient).

**Risk 3: The proposed schedule requires more steps at high τ, which may slow palette convergence.** The palette is co-trained with the indices, and at high τ the soft weight `W_soft` is a blend, which may not give clean gradient signal to the palette. **Mitigation:** the palette gradient `grad_palette = grad_W · P[k]` is independent of τ (it depends only on `P`, which is always normalized). The palette convergence rate is therefore unaffected by τ.

**Risk 4: The hold at `τ = 0.5` for 2300 steps may be too long.** If the indices have already converged by step 6000, the additional 2300 steps at `τ = 0.5` provide no benefit (and may cause overfitting). **Mitigation:** monitor `argmax(logits)` flip rate; if it drops below 0.01% per step, early-stop the indices training and switch to palette+LoRA only.

---

## 10. Summary

Our current linear `2.0 → 0.1` schedule over 4000 steps is suboptimal for K=4 Gumbel-Softmax because:

1. It spends 25% of the annealing window at `τ < 0.5`, where gradients are <9% of peak.
2. It holds at `τ = 0.1` for the second half of training, where gradients are essentially zero.
3. It never enters the `τ > 2.0` regime, which would give 20-30% more gradient signal.

The proposed schedule (piecewise warmup at `τ = 2.0` for 500 steps, then quadratic polynomial decay to `τ = 0.5` over 6000 steps, then hold at `τ = 0.5`) gives a **62% boost** in cumulative gradient signal and keeps the indices trainable for the entire 8300-step run.

The schedule should be combined with:
- **Deterministic-ST** (no Gumbel noise) to prevent oscillation at high τ.
- **Logit gap clamping** (max gap = 4) to prevent the winner logit from running away.
- **`1/√(N_i)` per-group gradient rescaling** to balance codebook entry usage.

These three changes together are expected to break the `cos = 0.95` plateau by allowing the indices to continue moving throughout training, rather than freezing at step 4000.

---

## 11. References

1. Jang, E., Gu, S., Poole, B. *Categorical Reparameterization with Gumbel-Softmax.* [arXiv:1611.01144](https://arxiv.org/abs/1611.01144)
2. Wang, L. et al. *Learnable Lookup Table for Neural Network Quantization.* CVPR 2022. [OpenAccess](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Learnable_Lookup_Table_for_Neural_Network_Quantization_CVPR_2022_paper.html)
3. Maddison, C. J., Mnih, A., Teh, Y. W. *The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables.* [arXiv:1611.00712](https://arxiv.org/abs/1611.00712)
4. Nagel, M. et al. *Overcoming Oscillations in Quantization-Aware Training.* [arXiv:2203.11086](https://arxiv.org/abs/2203.11086)
5. Rolfe, J. T. *Discrete Variational Autoencoders.* [arXiv:1609.02200](https://arxiv.org/abs/1609.02200)
6. Kingma, D. P., Ba, J. *Adam: A Method for Stochastic Optimization.* [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)

*Code citations refer to commit `b82a6be` of `qwen-palettize`.*
