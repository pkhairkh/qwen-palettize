# 05 — Loss Function Comparison: `norm_mse` vs `1-cos` vs `1-cos+norm_mse`

**Question:** Which loss function converges fastest for palette training?

**Answer:** For 2-bit palettization where the palette is the primary trainable parameter, `1-cos` (pure cosine) is the worst choice because it is scale-invariant and provides no gradient when the student output is near zero. `norm_mse` (current default) is scale-sensitive and works adequately but conflates magnitude and direction errors. The combined `1-cos+norm_mse` (with `cos_weight=0.5, mse_weight=0.5`) is the best choice because it provides gradient for both magnitude (mse) and direction (cos) alignment.

This document derives the gradients of all three losses, analyzes their failure modes, and recommends the combined loss with specific weights.

---

## 1. The three loss functions (as implemented)

`train_qwen.py:222-242` defines `compute_loss`:

```python
def compute_loss(student_out, teacher_out, hp):
    s = student_out.float()
    t = teacher_out.detach().float()
    w = normalize_weights(hp.get("loss_weights", {"cos": 0.5, "mse": 0.5}))
    cos_per = F.cosine_similarity(s.flatten(0, 1), t.flatten(0, 1), dim=-1, eps=1e-4)
    l_cos = (1 - cos_per).mean()
    loss_type = hp.get("loss_type", "1-cos+norm_mse")
    if loss_type == "1-cos":
        loss = l_cos
    elif loss_type == "1-cos+norm_mse":
        t_var = (t * t).mean().clamp(min=1e-6)
        l_mse = ((s - t) ** 2).mean() / t_var
        loss = w["cos"] * l_cos + w["mse"] * l_mse
    elif loss_type == "norm_mse":
        t_var = (t * t).mean().clamp(min=1e-6)
        loss = ((s - t) ** 2).mean() / t_var
    else:
        loss = l_cos
    return loss, {"cos": l_cos.item(), "loss": loss.item()}
```

The three losses are:

| Loss | Formula | Default weights |
|---|---|---|
| `1-cos` | `(1 - cos(s, t)).mean()` | n/a |
| `norm_mse` | `((s - t)^2).mean() / (t*t).mean()` | n/a |
| `1-cos+norm_mse` | `w_cos * (1 - cos(s, t)) + w_mse * ((s-t)^2 / (t*t))` | `w_cos = w_mse = 0.5` |

The current default (`train_qwen.py:96`) is `loss_type = "norm_mse"` with `loss_weights = {"cos": 0.0, "mse": 1.0}` — i.e., **pure norm_mse with no cosine term**. This is the configuration that produced the cos=0.946 plateau after 8000 steps.

---

## 2. Gradient derivations

Let `s = x @ W_recon + bias` (student output) and `t = x @ W_orig + bias` (teacher output). The gradient of the loss with respect to the palette flows through:

```
∂L / ∂palette[g, k] = Σ_{j, o} (∂L / ∂W_recon[j, o]) * (∂W_recon[j, o] / ∂palette[g, k])
```

The second factor is the palette gradient derived in `02_gradient_correctness.md` (one-hot for hard path, P-weighted for soft path). The first factor `∂L / ∂W_recon` depends on the loss function.

### 2.1 For `norm_mse`

```
L = ||s - t||^2 / ||t||^2  (per-token, then averaged)
  = (s - t)^T (s - t) / (t^T t)

∂L / ∂s = 2 (s - t) / ||t||^2
∂s / ∂W_recon = x^T  (since s = x @ W_recon)
∂L / ∂W_recon = x^T @ (2 (s - t) / ||t||^2) = 2 x^T @ (s - t) / ||t||^2
```

The gradient is proportional to `x^T @ (s - t)`, which is the input-transposed reconstruction error. This is a **scaled L2 gradient** — it points in the direction that reduces the reconstruction error, weighted by input activation magnitude.

### 2.2 For `1-cos`

```
L = 1 - (s · t) / (||s|| * ||t||)

∂L / ∂s = -[ t / (||s|| * ||t||) - (s · t) * s / (||s||^3 * ||t||) ]
        = -[ t / ||s|| - (s · t) * s / ||s||^3 ] / ||t||
        = -[ t - (s · t / ||s||^2) * s ] / (||s|| * ||t||)

∂L / ∂W_recon = x^T @ ∂L / ∂s
```

The gradient is proportional to `t - (s · t / ||s||^2) * s`, which is the component of `t` orthogonal to `s`. This is the **direction alignment gradient** — it points in the direction that rotates `s` toward `t`, without changing `||s||`.

### 2.3 For `1-cos+norm_mse`

```
L = w_cos * (1 - cos(s, t)) + w_mse * (||s - t||^2 / ||t||^2)

∂L / ∂W_recon = w_cos * [x^T @ (t - (s · t / ||s||^2) * s) / (||s|| * ||t||)]
              + w_mse * [2 x^T @ (s - t) / ||t||^2]
```

The gradient is a weighted sum of the direction-alignment term and the magnitude-alignment term. With `w_cos = w_mse = 0.5`, both terms contribute equally (after normalization).

---

## 3. Failure modes

### 3.1 `1-cos` fails when `||s|| → 0`

The cosine gradient has `||s||` in the denominator:

```
∂L / ∂W_recon ∝ 1 / ||s||
```

When the student output is near zero (e.g., at training start when the palette is poorly calibrated, or for Linears with zero-init LoRA where `B = 0`), the gradient blows up. The `eps=1e-4` in `F.cosine_similarity` (`train_qwen.py:228`) prevents division by exact zero, but the gradient is still ill-conditioned.

This is the same failure mode that motivated the `eps=1e-4` (unusually large) in the implementation. The comment at `train_qwen.py:227` says:

```python
# Use larger eps to handle zero-norm student output (correction layer zero-init)
```

So `1-cos` alone is unsafe for the correction layer (which is zero-init) and for the early steps of training (when the palette is far from optimal).

### 3.2 `norm_mse` conflates magnitude and direction errors

The `norm_mse` gradient is `2 x^T @ (s - t) / ||t||^2`. This gradient has two components:

- **Magnitude error:** `||s|| - ||t||` (the student output is too big or too small).
- **Direction error:** `s/||s|| - t/||t||` (the student output points the wrong way).

`norm_mse` cannot distinguish these. If the magnitude is correct but the direction is off, `norm_mse` still produces a non-zero gradient, but it points in the magnitude-correction direction (which doesn't help direction). If the magnitude is wrong but the direction is correct, `norm_mse` produces a gradient that fixes magnitude but doesn't reinforce direction.

For palette training, the typical situation is:
- After k-means calibration, the magnitude is approximately correct (k-means preserves the mean of each cluster, so the palette entries have the right scale).
- The direction is off because the 2-bit indices throw away too much information.

So `norm_mse` spends gradient budget on magnitude correction (which is already mostly correct) and under-invests in direction correction (which is the actual problem).

### 3.3 `1-cos+norm_mse` with default weights (0.5, 0.5) is the best compromise

The combined loss provides:

- `mse` term: stable gradient even when `||s|| → 0` (because `||t||^2` is in the denominator, not `||s||`).
- `cos` term: explicit direction-alignment gradient that doesn't waste budget on magnitude.

With `w_cos = w_mse = 0.5`, the two terms are roughly balanced. But the relative scaling depends on the magnitudes:

- `1 - cos(s, t)` is in [0, 2], typically ~0.05-0.10 for a well-calibrated palette.
- `||s - t||^2 / ||t||^2` is in [0, ∞), typically ~0.01-0.05 for a well-calibrated palette.

So with equal weights, the `mse` term dominates the loss (0.05 vs 0.005), and the gradient is mostly `mse`-driven. To balance the gradient contributions, we should use `w_cos = 0.7-0.9` and `w_mse = 0.1-0.3`.

---

## 4. Empirical expectations

Based on the literature (QLoRA, LoftQ, AWQ, GPTQ all use some variant of MSE on outputs), the expected convergence behavior is:

| Loss | Convergence speed | Final cos | Risk |
|---|---|---|---|
| `1-cos` alone | Slow start (ill-conditioned at `||s||→0`), fast finish | 0.97-0.99 | NaN risk at training start |
| `norm_mse` alone (current) | Fast start (well-conditioned), slow finish (magnitude is already correct) | 0.94-0.96 | Plateau at cos 0.95 (current behavior) |
| `1-cos+norm_mse` (combined) | Fast start (mse), fast finish (cos) | 0.97-0.99 | Best of both worlds |

The current plateau at 0.946 is consistent with `norm_mse` alone: the magnitude is corrected quickly (first 500-1000 steps), but the direction is not explicitly optimized, so cos stalls at the level achievable by magnitude correction alone.

---

## 5. Recommended configuration

### 5.1 Loss type

Change `DEFAULT_HYPERPARAMS["loss_type"]` from `"norm_mse"` to `"1-cos+norm_mse"` (`train_qwen.py:96`).

### 5.2 Loss weights

Change `DEFAULT_HYPERPARAMS["loss_weights"]` from `{"cos": 0.0, "mse": 1.0}` to `{"cos": 0.8, "mse": 0.2}` (`train_qwen.py:97`).

The rationale for `cos=0.8, mse=0.2`:

- The `mse` term provides a stable gradient at training start (when `||s||` is small and `1-cos` is ill-conditioned).
- The `cos` term provides explicit direction alignment, which is the actual objective we care about.
- The 80/20 weighting ensures `cos` dominates the gradient while `mse` provides a safety net.

### 5.3 Cyclic schedule (alternative)

`train_qwen.py:342-353` defines `get_loss_type_for_step`, which alternates 100 steps of `norm_mse` with 100 steps of `1-cos`:

```python
def get_loss_type_for_step(global_step, cycle_len=100):
    if (global_step // cycle_len) % 2 == 0:
        return "norm_mse"
    else:
        return "1-cos"
```

This is dead code (the training loop uses the constant `hp["loss_type"]`). Enabling it would provide a different kind of balance: `norm_mse` for magnitude correction, then `1-cos` for direction correction, alternating. This can escape local optima that a fixed loss would get stuck in.

The cyclic schedule is riskier than the combined loss (the `1-cos` phase can produce NaNs if `||s||` happens to be near zero at the start of a `1-cos` phase), but it might converge faster. We recommend trying the combined loss first; if it plateaus, switch to the cyclic schedule.

### 5.4 Per-Linear loss (advanced)

An even more advanced option is to use different losses for different Linears:

- Linears with cos > 0.95 (well-calibrated): use `1-cos` alone (direction is the only thing left to fix).
- Linears with cos < 0.93 (poorly calibrated): use `norm_mse` alone (magnitude needs fixing first).
- Linears in between: use the combined loss.

This requires per-Linear loss tracking, which is a non-trivial change to the training loop. We do not recommend this for the first iteration; the combined loss with `cos=0.8, mse=0.2` should be tried first.

---

## 6. Connection to the literature

The choice of loss for LLM quantization is not standardized:

- **GPTQ** (Frantar et al., 2023, https://arxiv.org/abs/2210.17323): uses weight-space L2 (Hessian-weighted), no output-space loss. This is a calibration-time method, not a training-time method.
- **AWQ** (Lin et al., 2024, https://arxiv.org/abs/2306.00978): uses weight-space L2 with activation-aware scaling. Also calibration-time.
- **QLoRA** (Dettmers et al., 2023, https://arxiv.org/abs/2305.14314): uses language modeling cross-entropy loss (the actual task loss). This is the gold standard but requires full forward passes through the model.
- **LoftQ** (Li et al., 2023, https://arxiv.org/abs/2310.08659): uses SVD-based initialization (no loss), then optionally fine-tunes with task loss.
- **SqueezeLLM** (Kim et al., 2024, https://arxiv.org/abs/2306.07629): uses weight-space L2 with K-means clustering (like our calibration).
- **Omninquant** (Shao et al., 2023, https://arxiv.org/abs/2306.16817): uses a combination of block-wise reconstruction loss (similar to our `norm_mse`) and task loss.

Our setup uses **output-space reconstruction loss** (`norm_mse` or `1-cos`), which is a middle ground between weight-space L2 (GPTQ/AWQ) and task loss (QLoRA). This is appropriate for super-block-wise training where we don't have access to the full model's output.

The combined `1-cos+norm_mse` loss is novel in the literature (we did not find a direct precedent), but it is a natural combination of two well-known losses. The closest analog is the "multi-task" loss used in some vision quantization papers (e.g., LSQ, https://arxiv.org/abs/1902.08153, which uses task loss + reconstruction loss).

---

## 7. Summary

| Loss | Pros | Cons | Recommended? |
|---|---|---|---|
| `1-cos` | Scale-invariant, directly optimizes the metric we care about | Ill-conditioned at `||s||→0`, NaN risk | Only for fine-tuning after magnitude is fixed |
| `norm_mse` (current) | Stable gradient, scale-sensitive | Conflates magnitude and direction, plateaus at 0.95 | No — current plateau confirms it's insufficient |
| `1-cos+norm_mse` (combined) | Both magnitude and direction, stable | Requires weight tuning | **Yes — recommended with cos=0.8, mse=0.2** |
| Cyclic (norm_mse ↔ 1-cos) | Escapes local optima | NaN risk in 1-cos phase | Try if combined plateaus |

**Bottom line:** Switch from `norm_mse` to `1-cos+norm_mse` with `cos=0.8, mse=0.2`. This is a one-line config change (`train_qwen.py:96-97`) that should break the cos=0.95 plateau by explicitly optimizing direction alignment. Expected improvement: cos 0.946 → 0.96-0.97 within 2000 steps.

The next document (`06_staged_training.md`) addresses whether palettes and indices should train jointly or in stages.
