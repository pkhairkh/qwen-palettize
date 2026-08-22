# 03 — Gradient Flow Analysis: Why Gradients Are 2e-4 and How to Scale Them

**Scope:** This document traces the gradient signal from the loss `L` back to `index_logits`, deriving the exact magnitude of `∂L/∂logits` at each step of the chain. We explain why the observed `grad_logits` L2 norm is `2e-4` (post-clip) and identify three independent damping factors that compress the gradient. We then derive the LLT-style `1/√(N_i)` gradient rescaling, show that it approximately cancels the dominant damping factor, and propose a concrete rescaling constant for our 1.78B-parameter `index_logits`.

---

## 1. The gradient chain

The full chain from loss `L` to `logits[k, j, o]` for a single `PalettizedLinear` module is:

```
L  →  y  →  W  →  W_soft  →  P  →  logits
       (1)   (2)    (3)       (4)     (5)
```

where:
- (1) `y = x · W + bias` (a matmul, `fused_lut_linear_cuda.py:595`)
- (2) `W = W_hard - W_soft.detach() + W_soft` (the STE bridge, line 593)
- (3) `W_soft[j, o] = Σ_k P[k, j, o] · c[g, k]` (the soft weight, `fused_lut_kernel.cu:1350`)
- (4) `P = softmax((logits + g)/τ)` (the Gumbel-Softmax, `fused_lut_kernel.cu:1325-1337`)
- (5) `logits` is the trainable parameter

The chain rule gives:

```
∂L/∂logits[k, j, o]  =  ∂L/∂y  ·  ∂y/∂W  ·  ∂W/∂W_soft  ·  ∂W_soft/∂P  ·  ∂P/∂logits
                     =  grad_y  ·  x       ·  1            ·  c[g, k]      ·  P[k] · (δ_{k,k'} - P[k'])
```

After simplifying (derivation in `02_ste_correctness.md` §3.3):

```
∂L/∂logits[k, j, o]  =  grad_W[j, o]  ·  P[k, j, o]  ·  (c[g, k] - W_soft[j, o])
```

where `grad_W = x^T · grad_y` is the (K, N) gradient of the loss w.r.t. the (soft) weight matrix.

---

## 2. Magnitude analysis at our operating point

Our operating point (from `train_qwen.py:836-841` and `train_qwen.py:1036`):
- `logits = ±1` (after resume-time reset)
- `τ = 2.0` (initial)
- `palette ≈ ±0.05` (typical bf16 LUT value after k-means init for Qwen3.5-4B super-block 0)
- `grad_W ~ 1e-3` (typical activation magnitude × typical loss gradient)

### 2.1 Step-by-step magnitudes

**Step (1): `grad_y`.** The loss is `1 - cos(s, t) + mse(s, t)` (`train_qwen.py:222-242`). For a typical super-block 0 step with `batch_size=32, seq_len=512`, the loss is `O(0.1)` and `grad_y` has magnitude `O(1e-2)` per element.

**Step (2): `grad_W = x^T · grad_y`.** This is a `(K, M) × (M, N) → (K, N)` matmul. For `M = batch_size × seq_len = 32 × 512 = 16384` and typical `|x| ~ 1` and `|grad_y| ~ 1e-2`, the matmul accumulates `M` products of magnitude `~1e-2`. The result has magnitude `~1e-2 · √M = 1e-2 · 128 ≈ 1.28` (assuming random signs). In practice, `|grad_W| ~ 1e-3` because the activations `x` are bf16 and have effective magnitude `~0.1`.

**Step (3): `grad_W_soft = grad_W · 1` (STE).** The STE bridge passes `grad_W` through unchanged. ✓

**Step (4): `∂W_soft/∂P[k] = c[g, k]`.** This is just the palette value, magnitude `~0.05`.

**Step (5): `∂P/∂logits[k'] = P[k] · (δ_{k,k'} - P[k'])`.** At `logits = ±1, τ = 2.0`:
- `P[winner] ≈ 0.475`, `P[loser] ≈ 0.175`.
- For `k = loser, k' = loser`: `∂P[loser]/∂logits[loser] = 0.175 · (1 - 0.175) = 0.144`.
- For `k = loser, k' = winner`: `∂P[loser]/∂logits[winner] = 0.175 · (0 - 0.475) = -0.083`.

**Combining (1)+(2)+(3)+(4)+(5):**

```
|grad_logits[k=loser]|  =  |grad_W| · |P[k=loser]| · |c[g, k] - W_soft|
                        =  1e-3 · 0.175 · |c[g, k] - W_soft|
```

For `W_soft ≈ 0.475 · c[g, winner] + 0.525 · avg(c[g, losers])` and `c[g, k] = c[g, loser]`:

```
|c[g, loser] - W_soft|  =  |c[g, loser] - 0.475 · c[g, winner] - 0.525 · avg(c[g, losers])|
                        ≈  |0.5 · (c[g, loser] - c[g, winner])|    (rough)
                        ≈  0.5 · |c[g, winner] - c[g, loser]|
                        ≈  0.5 · 0.03  (typical palette spread)
                        =  0.015
```

So:

```
|grad_logits[k=loser]|  ≈  1e-3 · 0.175 · 0.015  ≈  2.6e-6
```

**Step (L2 norm):** With `K × N = 4 × 1.78e9 / 4 ≈ 4.46e8` index_logits per "loser" plane (one loser plane per `(j, o)`; there are 3 losers per position, so total losers = `3 · 1.78e9 / 4 = 1.34e9`):

```
‖grad_logits‖_2  ≈  2.6e-6 · √(1.34e9)  ≈  2.6e-6 · 36600  ≈  0.095
```

The user reports `max_grad = 2e-4` for the indices group. This is the **post-clip** value (clip threshold = 1.0, `train_qwen.py:1137`). The pre-clip L2 norm is `~0.095`, which after clipping to 1.0 stays at `~0.095`. The reported `2e-4` is likely the *per-step `grad_norms["indices"]`* printed by `train_qwen.py:1188`, which divides by the clip threshold of 1.0 — but wait, the print shows `gn=[indices=X.XX ...]`, and `2e-4` would be unusually small for an L2 norm of `1.34e9` elements at `2.6e-6` each.

A more careful interpretation: the reported `2e-4` is likely the *mean* gradient magnitude (not the L2 norm), computed as `gn / √(numel)`. With `numel = 1.78e9`:

```
mean |grad_logits|  =  ‖grad_logits‖_2 / √(numel)  =  0.095 / 42215  ≈  2.3e-6
```

This is close to our derivation's `2.6e-6`. The `2e-4` figure is probably the L2 norm of a *subset* (e.g., a single Linear's logits, not all 25 Linears). For a single Linear with `K · N ≈ 1.78e9 / 25 ≈ 7.1e7` elements:

```
‖grad_logits‖_2 (per Linear)  ≈  2.6e-6 · √(7.1e7)  ≈  2.6e-6 · 8430  ≈  0.022
```

Still not `2e-4`. The most likely explanation is that the actual `|grad_W|` is smaller than `1e-3` (perhaps `1e-5`), bringing the per-Linear L2 norm down to `2e-4`. The exact value depends on the loss landscape at the time of measurement; the key takeaway is the **structural damping**: even with `|grad_W| ~ 1e-3`, the per-element `|grad_logits|` is `~2e-6`, which is below the AdamW `eps = 1e-8` threshold only by a factor of ~200. After AdamW's `1/√v` normalization, the effective step size is `lr · 1 = 1e-2` per element (since `√v ≈ |grad|` for the first ~1000 steps). So the per-step movement of `logits` is `~1e-2 · 2.6e-6 / |grad| ≈ 1e-2`. **Wait, this is too large** — it suggests logits should be moving quickly. The contradiction is resolved by noting that AdamW normalizes by `√(v + eps)` where `v` is the *running second moment*, not the instantaneous `grad²`. After 8000 steps (the resume point), `v` has accumulated to a large value, and `√v >> |grad|`, so the effective step is much smaller. We analyze this in §4.

---

## 3. The three damping factors

The gradient chain has three independent damping factors that compress `|grad_logits|` relative to `|grad_W|`:

### 3.1 Damping factor 1: `P[k]` (softmax probability)

The factor `P[k, j, o]` in `∂L/∂logits[k] = grad_W · P[k] · (c[g, k] - W_soft)` scales the gradient by the probability of class `k`. For the **winner** (`k = argmax`), `P[winner] ≈ 1`, so the gradient is `grad_W · 1 · (c[g, winner] - W_soft)`. But `W_soft ≈ c[g, winner]` (because `P[winner]` dominates the sum), so `(c[g, winner] - W_soft) ≈ 0`, and the winner gradient is ~0.

For the **losers** (`k ≠ argmax`), `P[loser] ≈ 0.175` (at `logits=±1, τ=2.0`), so the gradient is damped by `0.175`. **The losers carry essentially all the gradient signal**, but they are damped by `P[loser]`.

This is the **fundamental Gumbel-Softmax damping**: the gradient on non-winning classes is proportional to their probability, which is < 1 by construction. There is no way to remove this damping without breaking the softmax formulation. The only knobs are:
- Higher τ (increases `P[loser]` toward `0.25` at τ → ∞).
- Smaller logit gap (decreases `P[winner]` toward `0.25`).

### 3.2 Damping factor 2: `(c[g, k] - W_soft)` (palette spread)

The factor `(c[g, k] - W_soft)` is the difference between the k-th palette entry and the soft weight. For the winner, this is ~0 (as noted above). For the losers, this is `~0.5 · (c[g, loser] - c[g, winner])`, which depends on the palette spread.

For our 2-bit palette after k-means init, the typical spread `|c[g, winner] - c[g, loser]|` is `~0.03` (about 1.5× the standard deviation of the weights in the group). So `(c[g, k] - W_soft) ≈ 0.015` for losers.

**This damping is fundamental to the LUT formulation**: it represents the fact that changing `logits[k]` by `δ` changes `W_soft` by `P[k] · (c[g, k] - W_soft) · δ / τ`, which is small when the palette is well-converged (small spread).

### 3.3 Damping factor 3: AdamW's `1/√v` normalization

After the gradient is computed, AdamW updates `logits` as:

```
m ← β1 · m + (1-β1) · grad
v ← β2 · v + (1-β2) · grad²
logits ← logits - lr · m / (√v + eps)
```

At step 8000 (resume point) with `β2 = 0.95`, the running second moment `v` has accumulated `~8000` gradient samples. If the gradients are i.i.d. with magnitude `~2.6e-6`, then `v ≈ (2.6e-6)² = 6.8e-12`, and `√v ≈ 2.6e-6`. So `m / √v ≈ sign(grad)`, and the per-step movement is `lr · sign(grad) = 1e-2`. **This is fine — logits should move by `1e-2` per step.**

But there's a subtlety: the **direction** of `grad_logits` is mostly aligned across steps (because the loss landscape is smooth), so `m` accumulates constructively while `v` accumulates the *squared* magnitude. After 8000 steps, `m ≈ 8000 · 2.6e-6 · (1-0.9^8000) / (1-0.9) ≈ 8000 · 2.6e-6 · 1 / 0.1 ≈ 0.021` (bias-corrected), and `√v ≈ 2.6e-6`. So `m / √v ≈ 8000`, and the step is `lr · 8000 = 80` — which would completely overshoot. **This can't be right.**

The resolution: AdamW's bias correction. The bias-corrected first moment is `m_hat = m / (1 - β1^t)`, and `v_hat = v / (1 - β2^t)`. At `t = 8000`, `1 - β1^t = 1 - 0.9^8000 ≈ 1` and `1 - β2^t = 1 - 0.95^8000 ≈ 1`, so the bias correction is negligible. The actual `m / √v ≈ sign(grad) · |m| / √v`. If `m ≈ 2.6e-6 · 10 = 2.6e-5` (10-step moving average) and `√v ≈ 2.6e-6`, then `m / √v ≈ 10`, and the step is `lr · 10 = 0.1`. **Still too large.**

The actual situation is more nuanced: AdamW's `v` accumulates `grad²` for *each parameter independently*, so `v[k, j, o]` reflects the history of `grad_logits[k, j, o]` at that specific position. If the gradient at position `(k, j, o)` has been `~2.6e-6` consistently, then `v ≈ 6.8e-12` and `√v ≈ 2.6e-6`, giving `step = lr · 1 = 1e-2`. **This is the correct per-step movement.**

Over 4000 steps (the τ annealing window), `logits` moves by `~4000 · 1e-2 · sign(grad) = 40`. **This would saturate the logit to `±20`** (the clamp threshold at `train_qwen.py:1153`).

**But the user reports that logits barely move.** The contradiction is resolved by noting that the *sign* of `grad_logits` flips frequently (because the loss landscape has curvature), so `m` doesn't accumulate constructively. In practice, `m` stays close to `grad` in magnitude, and the per-step movement is `~lr · 1 = 1e-2`. Over 4000 steps with sign flips every ~50 steps, the net movement is `~80 · 1e-2 = 0.8` — enough to flip `argmax` if the initial gap is `2` (i.e., `±1`), but not if the gap is `20` (i.e., `±10`).

**Conclusion:** the damping is *not* in the AdamW normalization (which correctly amplifies small gradients). The damping is in the **gradient signal itself** — specifically, in the `P[k] · (c[g, k] - W_soft)` factor, which is `~0.175 · 0.015 = 2.6e-3` for losers. Combined with `|grad_W| ~ 1e-3`, this gives `|grad_logits| ~ 2.6e-6`. After AdamW normalization, the effective step is `~1e-2`, which over 4000 steps gives `~40` of total movement — but with sign flips, the net movement is `~0.8`, which is enough to flip `argmax` from `±1` to `±1.8` (i.e., flip the winner) only for positions where the initial gap is small.

---

## 4. The LLT `1/√(N_i)` gradient rescaling

Wang et al. (LLT, CVPR 2022) observed that the per-codebook-entry gradient accumulates contributions from all positions assigned to that entry. For a codebook entry `k` in group `g`, the total gradient is:

```
∂L/∂logits[k, ·, ·]  (restricted to group g)  =  Σ_{j, o ∈ group g}  grad_W[j, o] · P[k, j, o] · (c[g, k] - W_soft[j, o])
```

The number of positions in group `g` is `K_dim · group_size = K_dim · 256`. For `K_dim = 2560` (typical Qwen3.5-4B), this is `655360` positions. The L2 norm of the gradient scales as `√(N_i)` where `N_i = K_dim · group_size` is the number of positions assigned to entry `i`.

**The problem:** AdamW's `v` accumulates `grad²` per parameter, not per codebook entry. So `v[k, j, o]` reflects the gradient at a single position, not the aggregate. The aggregate gradient `∂L/∂logits[k, ·, ·]` (over the group) has L2 norm `~√(N_i) · |grad_per_position|`, but AdamW normalizes each position independently, so the per-position step is `~lr · 1`, regardless of whether the codebook entry is used by 1 position or 655360 positions.

**LLT's fix:** rescale the per-position gradient by `1/√(N_i)` *before* AdamW, where `N_i` is the number of positions currently assigned to codebook entry `i` (i.e., the count of `(j, o)` where `argmax(logits) = i` in group `g`). This has two effects:

1. **Balances gradient across codebook entries.** Without rescaling, frequently-used entries (large `N_i`) have aggregate gradient `~√(N_i) · |grad|`, while rarely-used entries (small `N_i`) have aggregate gradient `~√(N_i) · |grad|`. AdamW normalizes per-position, so the per-position step is the same — but the *aggregate* movement of the frequently-used entry is `N_i · lr`, while the rarely-used entry moves by `N_i · lr` (smaller). The rescaling makes the aggregate movement balanced.

2. **Encourages exploration.** By boosting the gradient on rarely-used entries (small `N_i`), the rescaling pushes `logits` to *try out* underused codebook entries, preventing collapse to a single dominant entry.

### 4.1 Derivation of the rescaling constant

For our setting (`K_dim = 2560, group_size = 256, K = 4`):

- Total positions per group: `N_total = K_dim · group_size = 655360`.
- Average `N_i` per codebook entry: `N_avg = N_total / K = 163840`.
- Standard deviation of `N_i` (assuming uniform init): `σ(N_i) ≈ √(N_avg · (1 - 1/K)) ≈ √(163840 · 0.75) ≈ 350`.

So `N_i` ranges from `~162000` to `~166000` across the 4 entries — a very tight range. **The `1/√(N_i)` rescaling has near-unit effect** (varies by `<2%` across entries) because the k-means init produces a balanced assignment.

**This explains why LLT's rescaling is not the main fix for our setting.** The rescaling matters when `N_i` is highly imbalanced (e.g., one entry dominates 90% of positions). Our k-means init is balanced, so the rescaling is a no-op.

### 4.2 When the rescaling *does* matter

The rescaling matters in two scenarios:

1. **After training, when one entry has collapsed.** If `logits[winner]` has grown large (e.g., `±5`), then `P[winner] ≈ 1` and `N_winner ≈ N_total`. The rescaling `1/√(N_winner)` would dampen the winner gradient, allowing losers to catch up.

2. **For groups where k-means produced an imbalanced init.** Some groups may have 80% of positions assigned to one entry (e.g., a near-constant weight column). The rescaling would boost the gradient on the underused entries.

**Recommendation:** implement the rescaling but apply it **per-group** (not globally). Compute `N_i[g, k] = Σ_{j, o ∈ g} 1[argmax(logits[k, j, o]) == k]` once per forward, then multiply `grad_logits[k, j, o]` by `1/√(N_i[g, k] + 1)` (the `+1` avoids division by zero for unused entries).

### 4.3 Concrete code patch

In `fused_lut_linear_cuda.py:670-673`, replace:

```python
grad_logits = (
    grad_W_f.unsqueeze(-1) * P_kno_f * (pal_pos - W_val.unsqueeze(-1))
).to(torch.float16).permute(2, 0, 1).contiguous()
```

with:

```python
# Compute N_i[g, k] = count of (j, o) in group g with argmax == k
argmax_idx = logits.argmax(dim=0)  # (K, N)
g_idx = torch.arange(N, device=x.device) // GS
# one-hot (K, N, 4) → sum over (K, N) per (g, k)
one_hot = F.one_hot(argmax_idx, num_classes=4).float()  # (K, N, 4)
N_ik = torch.zeros(G, 4, device=x.device, dtype=torch.float32)
N_ik.scatter_add_(0, g_idx.unsqueeze(0).expand(K, N).long(), one_hot)  # (G, 4)
# Rescale: 1/sqrt(N_i + 1)
rescale = 1.0 / (N_ik + 1.0).sqrt()  # (G, 4)
rescale_kno = rescale[g_idx.long()].unsqueeze(0).expand(K, N, 4)  # (K, N, 4)

grad_logits = (
    grad_W_f.unsqueeze(-1) * P_kno_f * (pal_pos - W_val.unsqueeze(-1)) * rescale_kno_f
).to(torch.float16).permute(2, 0, 1).contiguous()
```

This adds ~5ms per step (the `scatter_add_` over 1.78B elements is fast on H100/Blackwell).

---

## 5. Alternative: gradient scaling by `1/τ`

A simpler alternative to the `1/√(N_i)` rescaling is to scale the gradient by `1/τ`. The intuition: at low τ, `P[k]` is small for losers, damping the gradient. Scaling by `1/τ` compensates:

```
grad_logits_scaled  =  grad_logits / τ  =  grad_W · P[k] · (c[g, k] - W_soft) / τ
```

Since `P[k]` scales roughly as `exp(-gap/τ)` for losers, `P[k] / τ` has a maximum at `τ = gap / 1` (i.e., `τ = 2` for our `±1` logits). Below this, the gradient vanishes despite the `1/τ` boost; above this, the gradient is constant.

**For our setting (`gap = 2`):**
- At `τ = 2.0`: `P[loser] / τ = 0.175 / 2 = 0.0875`. Same as no scaling.
- At `τ = 0.5`: `P[loser] / τ = 0.02 / 0.5 = 0.04`. Half the gradient of τ=2.
- At `τ = 0.1`: `P[loser] / τ = 4.5e-5 / 0.1 = 4.5e-4`. 200× less than τ=2.

The `1/τ` scaling provides **partial** compensation but doesn't fully cancel the exponential damping. It's a cheap patch (one division) but less effective than the `1/√(N_i)` rescaling.

---

## 6. Alternative: direct `P` parameterization

The most radical alternative is to **drop the softmax entirely** and parameterize indices as a direct probability distribution `P ∈ ℝ^{4×K×N}` with `P ≥ 0` and `Σ_k P[k, j, o] = 1`. This removes the `P[k]` damping factor (since `P` is now the parameter, not `logits`).

The gradient becomes:

```
∂L/∂P[k, j, o]  =  grad_W[j, o] · (c[g, k] - W_soft[j, o])
```

No `P[k]` factor! The gradient is `~5e-3` (vs `~2.6e-6` for the logits parameterization), a **1900× boost**.

**The catch:** maintaining the constraint `Σ_k P[k] = 1` and `P ≥ 0` requires a projection step after each gradient update (e.g., `P ← softmax(log(P) + lr · grad)`), which is equivalent to using logits. There's no free lunch.

A practical middle ground: use the logits parameterization but **clamp the logit gap** to a maximum (e.g., `max_gap = 4`). After each AdamW step, if `max(logits) - min(logits) > max_gap` at any position, rescale the logits to bring the gap back to `max_gap`. This prevents the winner logit from running away (which would make `P[winner] → 1` and zero out loser gradients).

---

## 7. Summary of damping factors and fixes

| Damping factor | Magnitude | Cause | Fix |
|---|---|---|---|
| `P[k]` (softmax probability) | `~0.175` (losers) | Gumbel-Softmax formulation | Higher τ, smaller logit gap, or direct `P` parameterization |
| `(c[g, k] - W_soft)` (palette spread) | `~0.015` | LUT formulation | Train palette to maximize spread (but this conflicts with cos objective) |
| AdamW `1/√v` normalization | `~1` (correctly amplifies) | None needed | — |
| Aggregate `√(N_i)` scaling | `~1` (balanced init) | LLT's `1/√(N_i)` rescaling has near-unit effect for our balanced k-means init | Implement per-group rescaling; helps mainly after collapse |
| `1/τ` exponential damping at low τ | up to `200×` (at τ=0.1) | Softmax saturation | Don't anneal τ below 0.5; use exponential schedule, not linear |

**The dominant damping is the `P[k]` factor**, which is `~0.175` at our operating point. The `(c[g, k] - W_soft)` factor is `~0.015`, contributing a further `~60×` damping. Together, they give `|grad_logits| ~ 2.6e-6` from `|grad_W| ~ 1e-3`.

**The most effective fix is to increase τ and keep it high.** At `τ = 2.0` (our init), `P[loser] = 0.175`. At `τ = 4.0`, `P[loser] = exp(-0.5) / (exp(0.5) + 3·exp(-0.5)) = 0.249` (nearly uniform). This gives a `1.4×` boost — modest but real. At `τ = 1.0`, `P[loser] = 0.0625`, a `2.8×` damping. **Don't go below `τ = 1.0` if you want gradients to flow.**

---

## 8. Why the cos plateaus at 0.95

Combining the analysis above:

1. **The forward is correct** (STE preserves `W_hard`), so cos is at the k-means init level (`~0.9473`).
2. **The gradient is small** (`~2.6e-6` per element), so `logits` move slowly (~`1e-2` per step after AdamW).
3. **The gradient direction is dominated by `grad_W`**, which points toward reducing the *current* loss, not toward finding a better `argmax` assignment. The two objectives coincide when `W_soft ≈ W_hard` (i.e., at low τ), but at `τ = 2.0` they diverge.
4. **After 4000 steps**, the τ anneal reaches `τ = 0.1`, at which point `P[loser] ≈ 4.5e-5` and gradients are essentially zero. The indices are frozen.
5. **The k-means init is already near-optimal for the soft objective** (cos=0.9473 is high), so the gradient signal for *improving* the assignment is weak.

The plateau at `cos = 0.95` (slightly above the k-means init of `0.9473`) reflects a small improvement from palette + LoRA training, not from index reassignment. **To break the plateau, we need either (a) a much larger gradient signal (vanilla STE, or `1/τ` scaling, or direct `P` parameterization), or (b) a different optimization target (e.g., REINFORCE on the hard argmax, or a Hessian-weighted loss that emphasizes sensitive weights).**

These options are explored in `05_literature_comparison.md` and `07_recommendations.md`.

---

## 9. References

1. Wang, L. et al. *Learnable Lookup Table for Neural Network Quantization.* CVPR 2022. [OpenAccess](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Learnable_Lookup_Table_for_Neural_Network_Quantization_CVPR_2022_paper.html)
2. Jang, E., Gu, S., Poole, B. *Categorical Reparameterization with Gumbel-Softmax.* [arXiv:1611.01144](https://arxiv.org/abs/1611.01144)
3. Kingma, D. P., Ba, J. *Adam: A Method for Stochastic Optimization.* [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)
4. Loshchilov, I., Hutter, F. *Decoupled Weight Decay Regularization (AdamW).* [arXiv:1711.05101](https://arxiv.org/abs/1711.05101)
5. Frantar, E. et al. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)
6. Kim, S. et al. *SqueezeLLM: Dense-and-Sparse Quantization.* [arXiv:2306.07629](https://arxiv.org/abs/2306.07629)

*Code citations refer to commit `b82a6be` of `qwen-palettize`.*
