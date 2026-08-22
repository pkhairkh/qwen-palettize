# 00 — Executive Overview: Trainable Indices Deficiency Diagnosis

**Repository:** [qwen-palettize](https://github.com/pkhairkh/qwen-palettize) @ commit `b82a6be` (post-STE)
**Audited module:** `CUDAFusedLUTLinearSoft` (`scripts/fused_lut_linear_cuda.py:519-684`)
**Pathology:** Training plateaus at `cos = 0.95`; `index_logits` (1.78B params) barely move despite 8300 steps of FP32-master AdamW at `lr=1e-2`.

---

## 1. The problem in one paragraph

We use Gumbel-Softmax to make 2-bit index assignments (K=4 codebook entries per group of 256 weights) trainable. The forward pass uses the hard one-hot `W_hard = palette[argmax(logits)]` via Straight-Through Estimator (STE), which preserves cos=0.9473 (the k-means init quality) regardless of τ. The backward pass uses the soft Gumbel-Softmax gradient `grad_W · P[k] · (c[g,k] - W_soft)`. At our operating point (`logits=±1, τ=2.0`), this gradient is `~2.6e-6` per element — small but non-zero. After AdamW's `1/√v` normalization, the per-step movement is `~1e-2`, which over 4000 steps (the τ anneal window) should give `~40` of total movement. **But sign flips in the gradient reduce the net movement to `~0.4`**, and after step 4000 the τ anneal reaches `τ=0.1`, at which point `P[loser] ≈ 1.5e-5` and gradients are essentially zero. The indices are frozen for the second half of training, and cos plateaus at the k-means init level plus a small improvement from palette+LoRA training.

---

## 2. Root cause: three independent damping factors

The gradient chain `L → y → W → W_soft → P → logits` has three damping factors that compress `|grad_logits|` from `|grad_W| ~ 1e-3` down to `~2.6e-6` (a 380× compression):

1. **`P[k]` (softmax probability):** the gradient on non-argmax logits is proportional to their softmax probability, which is `~0.175` at our operating point. This is the fundamental Gumbel-Softmax damping — there is no way to remove it without breaking the softmax formulation. The only knobs are higher τ (increases `P[loser]` toward `0.25`) and smaller logit gap (decreases `P[winner]` toward `0.25`).

2. **`(c[g,k] - W_soft)` (palette spread):** the gradient is proportional to the difference between the k-th palette entry and the soft weight. For our converged palette, this is `~0.015` for losers. This damping is fundamental to the LUT formulation — it represents the fact that changing `logits[k]` by `δ` changes `W_soft` by `P[k] · (c[g,k] - W_soft) · δ / τ`, which is small when the palette is well-converged.

3. **τ anneal driving `P[loser] → 0`:** our linear `2.0 → 0.1` schedule over 4000 steps spends 25% of the window at `τ < 0.5`, where `P[loser] < 0.016` and gradients are <9% of peak. After step 4000, τ is held at 0.1, where `P[loser] ≈ 1.5e-5` and gradients are essentially zero.

---

## 3. The diagnosis: STE is correct, but insufficient

**The STE implementation is mathematically correct** (proven in `02_ste_correctness.md`):
- Forward: `W = W_hard - W_soft.detach() + W_soft = W_hard` exactly. ✓
- Backward: `∂L/∂logits = grad_W · P[k] · (c[g,k] - W_soft)`, matching the canonical Gumbel-Softmax gradient. ✓
- The formula matches both the CUDA kernel (`fused_lut_kernel.cu:1394-1397`) and the active PyTorch fallback (`fused_lut_linear_cuda.py:670-673`).

**But correctness is not enough.** The STE gradient is biased (it approximates the true zero-gradient of `argmax` with the soft Gumbel-Softmax gradient), and its magnitude is too small to flip `argmax` in 4000 steps for positions with a non-trivial logit gap. The plateau at `cos=0.95` is the expected behavior of Gumbel-ST at K=4 with our choice of τ and logit init — it is not a bug.

---

## 4. The five concrete fixes (detailed in `07_recommendations.md`)

Based on the audit (Wave 1), gradient analysis (Wave 2), literature comparison (Wave 3), and optimizer analysis (Wave 3), we propose five concrete fixes, ordered by expected impact:

### Fix 1: Replace linear τ schedule with polynomial decay (highest impact)

**Current:** `τ(t) = max(0.1, 2.0 · (1 - t/4000))` — linear, spends 25% of window at `τ < 0.5`.
**Proposed:** `τ(t) = 2.0` for 500 steps warmup, then `max(0.5, 2.0 · (1 - (t-500)/5500)²)` — quadratic polynomial, holds at `τ=0.5` for the final 2300 steps.
**Expected impact:** 62% boost in cumulative gradient signal; indices remain trainable throughout the 8300-step run.
**Code change:** `train_qwen.py:1036` (one-line patch).

### Fix 2: Switch from Gumbel-ST to deterministic-ST (LLT pattern)

**Current:** Gumbel noise added in `fused_lut_kernel.cu:1325-1328` via the LCG sampler.
**Proposed:** Remove the Gumbel noise; use plain `softmax(logits/τ)`.
**Expected impact:** Reduces gradient variance (sign flips), allowing `m` to accumulate constructively in AdamW. The marginal-correctness guarantee is lost, but for K=4 this is acceptable.
**Code change:** `fused_lut_kernel.cu:1325-1328` (remove `gumbel_sample` calls).

### Fix 3: Tighten logit clamp from ±20 to ±5

**Current:** `train_qwen.py:1153` clamps `index_logits` to `[-20, 20]` after each step. At ±20, the softmax is numerically one-hot and gradients are zero.
**Proposed:** Clamp to `[-5, 5]`. At ±5 with τ=0.5, `P[loser] = 1/(exp(10)+3) ≈ 4.5e-5` — still small, but the winner logit can't run away to ±20.
**Expected impact:** Prevents logit saturation; keeps `P[loser]` non-zero throughout training.
**Code change:** `train_qwen.py:1153` (change `20.0` to `5.0`).

### Fix 4: Implement LLT's `1/√(N_i)` per-group gradient rescaling

**Current:** No rescaling. AdamW normalizes per-position, so frequently-used codebook entries (large `N_i`) get the same per-position step as rarely-used entries.
**Proposed:** Multiply `grad_logits[k, j, o]` by `1/√(N_i[g, k] + 1)` where `N_i[g, k]` is the count of positions in group `g` currently assigned to entry `k`.
**Expected impact:** Near-unit effect for our balanced k-means init, but becomes important if collapse occurs. Safety net.
**Code change:** `fused_lut_linear_cuda.py:670-673` (~10 lines added).

### Fix 5: Add Hessian-weighted gradient (SqueezeLLM pattern)

**Current:** All positions receive equal gradient weight.
**Proposed:** Multiply `grad_logits[k, j, o]` by `H_diag[j, o] = 2 · Σ_i x[i, j]²` (precomputed from calibration activations).
**Expected impact:** Focuses gradient on sensitive weights; ~2× improvement in convergence speed on the most important positions.
**Code change:** Precompute `H_diag` once (one-time ~1GB), then multiply in backward pass (~1.78B FLOPs/step, negligible).

---

## 5. The radical alternative: switch to LUT-Q pattern

If the five fixes above don't break the plateau, the radical alternative is to **abandon Gumbel-Softmax entirely** and switch to the LUT-Q pattern (Cardinaux et al. 2018):

- Maintain an FP shadow weight matrix `W_shadow` (same shape as `W_hard`).
- Train `W_shadow` with standard STE (`grad_W_shadow = grad_W`).
- Every N steps (e.g., N=10), recompute `indices = argmin_k |W_shadow - palette[g, k]|` via k-means.

This eliminates the Gumbel-Softmax gradient damping entirely — the gradient on `W_shadow` is just `grad_W`, which is `~1e-3` (200× larger than `grad_logits`). The indices are derived, not directly differentiated.

**Cost:** k-means reassignment every 10 steps adds ~7B FLOPs (negligible vs the matmul). Memory: `W_shadow` is an additional `K · N · 4B = 26 MB` per Linear (negligible).

**Risk:** k-means reassignment may cause index oscillation (Nagel et al. 2022). Mitigation: freeze positions that have stabilized.

---

## 6. What we do NOT recommend

- **Switching to Muon optimizer:** the matrix-orthogonalization prior is meaningless for categorical logits (see `06_optimizer_analysis.md` §5).
- **Raising `lr` above `1e-2`:** causes divergence (the user already tried `1e-1` and saw cos → 0.03).
- **Lowering `lr` below `1e-2`:** too slow (the user already tried `1e-3` on L4).
- **Using plain AdamW (no fp32 master):** fp16 underflow with `eps=1e-8` causes NaN (see `06_optimizer_analysis.md` §3).
- **REINFORCE (expected gradient):** variance is too high for our 32-sequence batch size.

---

## 7. Document map

This research folder contains 9 documents (including this one):

| # | File | Pages | Focus |
|---|---|---|---|
| 00 | `00_overview.md` | 2 | Executive summary (this file) |
| 01 | `01_gumbel_softmax_audit.md` | 9.4 | Line-by-line audit of our Gumbel-Softmax impl vs Jang et al. 2017 |
| 02 | `02_ste_correctness.md` | 7.9 | Mathematical proof of STE correctness + 4 alternatives |
| 03 | `03_gradient_flow_analysis.md` | 10.2 | Why grads are 2e-4; three damping factors; `1/√(N_i)` derivation |
| 04 | `04_tau_schedule.md` | 7.5 | Optimal τ annealing; polynomial schedule proposal |
| 05 | `05_literature_comparison.md` | 9.3 | LUT-Q, LLT, GPTQ, SqueezeLLM, AWQ, BitNet, BNN, Nagel QAT |
| 06 | `06_optimizer_analysis.md` | 8.4 | AdamW vs SGD vs Muon; hybrid schedule proposal |
| 07 | `07_recommendations.md` | 3 | Concrete code patches for the 5 fixes |
| 08 | `08_references.md` | 2 | arxiv + GitHub links |

**Total: 59.7 pages** across 9 documents, with 50+ arxiv citations.

---

## 8. Next steps

1. **Apply Fix 1 (τ schedule) and Fix 3 (logit clamp)** — these are one-line changes with clear theoretical justification. Re-run training and observe whether the plateau moves above 0.95.

2. **If Fix 1+3 don't break the plateau, apply Fix 2 (deterministic-ST)** — slightly more involved (CUDA kernel change), but removes the Gumbel noise variance.

3. **If Fix 1+2+3 don't break the plateau, apply Fix 4 (`1/√(N_i)` rescaling) and Fix 5 (Hessian weighting)** — these are additive and address the gradient magnitude from different angles.

4. **If all five fixes don't break the plateau, switch to the LUT-Q pattern** (§5 above) — the radical alternative that eliminates Gumbel-Softmax entirely.

Each fix is independently testable and rollback-able. The recommended order minimizes code changes first, then structural changes.

---

## 9. TL;DR

Our Gumbel-Softmax indices are not training effectively because of three independent damping factors that compress the gradient from `~1e-3` to `~2.6e-6`. The STE implementation is correct but insufficient. The highest-impact fix is replacing the linear `2.0 → 0.1` τ schedule with a polynomial `2.0 → 0.5` schedule that keeps gradients flowing throughout training. Combined with a tighter logit clamp (`±5` instead of `±20`) and the deterministic-ST variant (no Gumbel noise), we expect the indices to continue moving past step 4000 and break the `cos = 0.95` plateau.

The full diagnosis, mathematical derivations, literature comparison, and concrete code patches are in the 8 companion documents.

---

*Generated by the Research Orchestrator Agent. Code citations refer to commit `b82a6be` of `qwen-palettize`.*
