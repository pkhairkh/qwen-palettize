# 06 — Staged Training: Palette-First vs Joint Optimization

**Question:** Should palette and indices train jointly (current approach via Gumbel-Softmax) or in stages (palette first, then indices, or vice versa)?

**Answer:** Joint training via Gumbel-Softmax is theoretically appealing but practically problematic for 2-bit palettization. The Gumbel-Softmax relaxation introduces 4× memory overhead (1.78B index_logits vs 445M int8 indices) and the STE trick creates a structural limitation where only the argmax slot receives gradient at low τ. A staged approach — palette-only training first, then index refinement via LUT-Q-style re-quantization — is simpler, faster, and likely more effective.

This document compares four training schedules and recommends a specific staged approach.

---

## 1. The four candidate schedules

### Schedule A: Joint training via Gumbel-Softmax (current)

- **Phase 1 (steps 0-8000):** Train `palette` (bf16, 2,208 params) + `index_logits` (fp16, 1.78B params) + LoRA + layernorms jointly via AdamW + Muon.
- **Forward:** Soft (Gumbel-Softmax with STE).
- **Backward:** Gradient flows to both palette and index_logits.
- **τ schedule:** 2.0 → 0.1 over 4000 steps, then constant 0.1.
- **End:** Extract hard indices via `argmax(index_logits)`.

This is the current approach (`train_qwen.py:1239` defaults to `use_soft_indices=1`).

### Schedule B: Palette-only training (frozen indices)

- **Phase 1 (steps 0-2000):** Train `palette` only. Indices frozen at k-means calibration values.
- **Forward:** Hard (gather + matmul).
- **Backward:** Gradient flows only to palette.
- **End:** Palette is fine-tuned; indices are still k-means values.

This is the approach that would be used if `use_soft_indices=0`.

### Schedule C: Staged palette-then-indices

- **Phase 1 (steps 0-2000):** Palette-only training (Schedule B).
- **Phase 2 (steps 2000-4000):** Re-quantize indices via k-means on the current `W_recon` (LUT-Q style), then train palette + LoRA with frozen indices for another 2000 steps.
- **Phase 3 (steps 4000-8000):** Optional: another round of re-quantization + palette training.

This is the LUT-Q-inspired hybrid recommended in `04_kmeans_vs_gradient.md`.

### Schedule D: Staged indices-then-palette

- **Phase 1 (steps 0-2000):** Train `index_logits` only (palette frozen at k-means values). Use Gumbel-Softmax.
- **Phase 2 (steps 2000-4000):** Train `palette` only (indices frozen at `argmax(index_logits)` from Phase 1).
- **Phase 3 (steps 4000-8000):** Joint fine-tuning of both.

This is the reverse of Schedule C.

---

## 2. Analysis of each schedule

### 2.1 Schedule A (joint, current): theoretical appeal, practical problems

**Theoretical appeal:** Joint optimization can find a better local optimum than alternating optimization, because the gradient information from both palette and indices is combined at every step.

**Practical problems:**

1. **Memory overhead:** `index_logits` is `(4, K, N) fp16` = 4× the size of hard `int8` indices. For super-block 0, this is 1.78B params = 3.56 GB (`train_sb0.log:34`). On an L4 (24 GB), this is 15% of total memory; on Blackwell (96 GB), it's 4%.

2. **Gradient interference:** The palette gradient is `Σ grad_W * P` (weighted by softmax probabilities), while the index gradient is `∂L/∂logits` (which depends on the Gumbel noise and the palette values). These two gradients can conflict: the palette gradient wants to move the palette entries to better fit the current indices, while the index gradient wants to move the indices to better fit the current palette. This creates a "cat-and-mouse" dynamic that can slow convergence.

3. **STE structural limitation:** As τ → 0, `P → one-hot`, and only the argmax slot receives palette gradient (see `02_gradient_correctness.md` §3.3). This means the non-argmax palette entries are stuck at their k-means values and cannot be fine-tuned. If the k-means values are suboptimal for a non-argmax slot (which they might be, since k-means optimizes the joint palette+indices objective, not the palette-only objective), the joint training cannot fix this.

4. **Index training was empirically a no-op at low τ:** The comment at `fused_lut_linear_cuda.py:643-647` states: "Empirically verified at tau=0.1 with logits=±10: all 25 index_logits grads are 0.0. The L4 'training' of indices was a no-op." The STE fix was added to make `grad_logits` non-zero, but the palette gradient is still governed by `P`, which is one-hot at low τ.

**Empirical result:** cos improved from 0.937 (calibration mean) to 0.946 after 8000 steps (`train_sb0.log:35`). This is a ~1% improvement, consistent with the "1-5% improvement bound" from `04_kmeans_vs_gradient.md` §8.

### 2.2 Schedule B (palette-only): simpler, similar results

**Advantages:**

1. **Memory savings:** No `index_logits` tensor. Saves 3.56 GB on L4.
2. **No gradient interference:** Only the palette receives gradient, so the optimization is clean.
3. **Faster convergence:** Fewer parameters to optimize (2,208 vs 1.78B + 2,208). AdamW converges in ~500-1000 steps for 2,208 params.
4. **No τ schedule:** Hard path is always used; no Gumbel noise; no STE.

**Disadvantages:**

1. **Same local optimum as k-means:** As shown in `04_kmeans_vs_gradient.md` §2.3, gradient descent on palette with frozen indices converges to the same solution as the k-means M-step. So Schedule B can only improve cos by ~1-3% (the off-diagonal Hessian terms that k-means ignores).
2. **No index refinement:** If the k-means indices are suboptimal, Schedule B cannot fix them.

**Expected result:** cos 0.937 → 0.95-0.96 after 1000-2000 steps. Similar to Schedule A but faster and with less memory.

### 2.3 Schedule C (staged palette-then-indices): the LUT-Q hybrid

**Advantages:**

1. **Escapes the k-means local optimum:** The re-quantization step in Phase 2 re-derives indices from the current `W_recon`, which has been updated by Phase 1's gradient descent. If the gradient updates have changed the cluster structure, the re-quantization will find better indices.
2. **Simple forward:** Both phases use the hard path (no Gumbel-Softmax, no STE).
3. **Low memory:** No `index_logits` tensor.
4. **Compatible with existing freeze logic:** `train_qwen.py:246-299` already implements `freeze_settled_palettes`, which can freeze indices that haven't changed between re-quantization steps (the Nagel et al. ICML 2022 fix, https://arxiv.org/abs/2203.11086).

**Disadvantages:**

1. **Re-quantization cost:** Running k-means on 1.78B weights takes ~30-60 seconds (based on the calibration log: `calib_sb0.log:236` says "Palettization done in 308s" for 25 tensors, so ~12s per tensor). Doing this every 2000 steps adds ~1% overhead.
2. **Oscillation risk:** If the re-quantization flips indices back and forth, the palette cannot converge. The `freeze_settled_palettes` function mitigates this.
3. **Two-phase complexity:** The training loop must support phase transitions (re-quantization, freezing, etc.).

**Expected result:** cos 0.937 → 0.96-0.97 after 4000-6000 steps. Better than Schedules A and B because the re-quantization can escape the k-means local optimum.

### 2.4 Schedule D (indices-then-palette): less principled

**Disadvantages:**

1. **Phase 1 is wasteful:** Training `index_logits` with the palette frozen at k-means values is essentially re-running k-means via gradient descent. But k-means is already optimal for the given palette (it's the E-step). So Phase 1 cannot improve cos beyond the k-means solution.
2. **Same STE limitation as Schedule A:** Phase 1 uses Gumbel-Softmax, which has the same memory overhead and structural limitations as Schedule A.

**Expected result:** cos 0.937 → 0.94-0.95 after 4000 steps. Worse than Schedules B and C.

---

## 3. Why the 5 worst Linears stay stuck under Schedule A

The 5 `BIG_LORA_TARGETS` (cos < 0.93 after calibration) receive rank-32 LoRA, but they still don't improve. Three reasons emerge from the staged-training analysis:

### 3.1 The palette is stuck at the k-means local optimum

For these Linears, the k-means palette is particularly suboptimal because the weight distribution is heavy-tailed or bimodal (4 cluster centers cannot represent the distribution well). Gradient descent on the palette (Schedule A's palette gradient) cannot escape this local optimum because the indices are stuck at the k-means assignment (or, in the soft path, the Gumbel-Softmax is converging to the same assignment as τ → 0).

**Fix:** Re-quantize the indices (Schedule C, Phase 2). This allows the indices to update to reflect any palette changes, breaking out of the k-means local optimum.

### 3.2 The LoRA is zero-init (no SVD warm start)

As documented in `01_palette_audit.md` §6.3, the LoRA uses `init="loftq"` but `original_weight=None` (`train_qwen.py:718`), which falls back to `B = 0` initialization. The LoRA must climb out of a zero-init valley from scratch.

**Fix:** Call `capture_original_weights` (`qwen_model.py:779-795`) before `attach_lora_to_layer` and pass the dict into `QwenLoRA`. This gives the LoRA an SVD-based warm start that immediately cancels the leading residual directions.

### 3.3 The residual is high-rank for these Linears

Even with SVD warm start, rank-32 LoRA can only correct the top-32 singular directions of the residual. For the 5 worst Linears, the residual spectrum is flatter (more directions matter), so rank-32 captures <50% of the energy.

**Fix:** Use rank-64 or rank-128 LoRA for these 5 Linears. The memory cost is +12 MB (rank-64) or +24 MB (rank-128) per Linear — manageable. Alternatively, use a smaller GROUP_SIZE (128 or 64) for these Linears, which improves the palette quality at the cost of more palette parameters (still tiny: 8,832 or 17,664 params).

---

## 4. The τ anneal schedule is suboptimal

`train_sb0.log:8` shows `tau: 2.0 → 0.1 over 4000 steps`. This anneal is too aggressive:

- **At τ=2.0 (step 0):** Gumbel noise dominates the logits (which are ±10 from the one-hot initialization at `qwen_model.py:124-127`). The softmax probabilities `P` are close to uniform, so `W_soft ≈ mean(palette)`. The STE correction `W_hard - W_soft.detach() + W_soft` has a large `W_soft.detach()` term, which means the forward value is `W_hard + (W_soft - W_soft) = W_hard` (correct), but the backward gradient flows through `W_soft` which is a poor approximation of `W_hard`. The palette gradient is `Σ grad_W * P` with `P ≈ uniform`, so all 4 slots receive equal gradient — this is a very noisy signal.

- **At τ=0.1 (step 4000+):** `P` is nearly one-hot, so only the argmax slot receives gradient (as discussed in `02_gradient_correctness.md` §3.3). The non-argmax slots are stuck.

**Better τ schedule:** Start at τ=0.5 (not 2.0), anneal to τ=0.3 (not 0.1) over 2000 steps, then hold at τ=0.3. This keeps `P` non-one-hot throughout training, so all 4 slots continue to receive gradient. The cost is that the forward is slightly less "hard" (the STE correction is larger), but the benefit is that the palette can be fine-tuned at all 4 slots, not just the argmax.

Alternatively, **skip the soft path entirely** (use Schedule B or C) and avoid the τ schedule altogether.

---

## 5. Recommended schedule

Based on the above analysis, we recommend **Schedule C (staged palette-then-indices)** with the following specifics:

### 5.1 Phase 1: Palette-only training (steps 0-2000)

- **Path:** Hard (`use_soft_indices=0`).
- **Trainable:** `palette` (fp32, see `03_precision_analysis.md`), LoRA (with SVD warm start), layernorms.
- **Frozen:** Indices (k-means values), embeddings, teacher.
- **Loss:** `1-cos+norm_mse` with `cos=0.8, mse=0.2` (see `05_loss_function.md`).
- **LR:** `palettes=3e-3, lora=1e-3, layernorms=3e-4` (current values).
- **Gradient clip:** Per-group (see `08_recommendations.md`), not global.

### 5.2 Phase 2: Re-quantization + palette training (steps 2000-4000)

- **At step 2000:** Run k-means on the current `W_recon` per group to get new `indices` and `palette`.
- **Path:** Hard.
- **Trainable:** `palette`, LoRA, layernorms.
- **Frozen:** New indices (from re-quantization).
- **Loss:** Same as Phase 1.
- **Freeze logic:** Call `freeze_settled_palettes` (`train_qwen.py:246-299`) every 500 steps to freeze indices that haven't changed.

### 5.3 Phase 3 (optional): Another re-quantization + palette training (steps 4000-6000)

- Repeat Phase 2 if cos is still improving.
- Stop when cos plateaus or when no indices change between re-quantization steps.

### 5.4 Phase 4 (optional): Joint fine-tuning (steps 6000-8000)

- If cos is still below target, switch to the soft path (`use_soft_indices=1`) with τ=0.5 (constant, no anneal) for a final round of joint palette+indices fine-tuning.
- This is the only phase that uses the Gumbel-Softmax path.

### 5.5 Expected results

| Phase | Expected cos | Cumulative steps |
|---|---|---|
| Calibration (k-means) | 0.937 | 0 |
| Phase 1 (palette-only) | 0.95-0.96 | 2000 |
| Phase 2 (re-quant + palette) | 0.96-0.97 | 4000 |
| Phase 3 (re-quant + palette) | 0.97-0.98 | 6000 |
| Phase 4 (joint fine-tune) | 0.97-0.98 | 8000 |

The final cos of 0.97-0.98 is below the 0.999 target (which is achievable only with 4-bit quantization or smaller GROUP_SIZE), but it is a significant improvement over the current 0.946.

---

## 6. Implementation sketch

The staged schedule requires the following changes to `train_qwen.py`:

### 6.1 Add a `--schedule` argument

```python
ap.add_argument("--schedule", type=str, default="staged",
                choices=["joint", "palette_only", "staged"])
```

### 6.2 Implement re-quantization

```python
def re_quantize_indices(model, sb_idx):
    """Re-run k-means on the current W_recon to get new indices + palette."""
    from palettize_pytorch import kmeans1d_weighted
    from qwen_model import PalettizedLinear

    for name, mod in model.named_modules():
        if not isinstance(mod, PalettizedLinear):
            continue
        # Reconstruct W_recon from current palette + indices
        W_recon = mod._reconstruct_weight_for_requant()
        # Run k-means per group
        new_indices, new_palette = run_kmeans_per_group(W_recon, mod.group_size)
        # Update indices and palette
        mod.indices = new_indices.long()
        mod.indices_int8 = new_indices.to(torch.int8).contiguous()
        mod.palette.data.copy_(new_palette.to(mod.palette.dtype))
```

### 6.3 Modify the training loop

```python
if schedule == "staged":
    # Phase 1: palette-only (hard path)
    train_phase(model, opt, steps=2000, use_soft=False, ...)

    # Phase 2: re-quantize + palette-only
    re_quantize_indices(model, sb_idx)
    train_phase(model, opt, steps=2000, use_soft=False, ...)

    # Phase 3: optional re-quantize + palette-only
    if cos < 0.97:
        re_quantize_indices(model, sb_idx)
        train_phase(model, opt, steps=2000, use_soft=False, ...)

    # Phase 4: optional joint fine-tune
    if cos < 0.97:
        convert_to_soft(model)  # create index_logits from argmax(indices)
        train_phase(model, opt, steps=2000, use_soft=True, tau=0.5, ...)
```

The full patch is provided in `08_recommendations.md`.

---

## 7. Summary

| Schedule | Memory | Convergence | Final cos | Complexity |
|---|---|---|---|---|
| A (joint, current) | 3.56 GB extra | Slow (8000+ steps) | 0.946 (plateau) | High (Gumbel, STE, τ schedule) |
| B (palette-only) | Baseline | Fast (1000-2000 steps) | 0.95-0.96 | Low (hard path only) |
| **C (staged, recommended)** | Baseline | Medium (4000-6000 steps) | **0.96-0.97** | Medium (re-quantization step) |
| D (indices-then-palette) | 3.56 GB extra | Slow (4000+ steps) | 0.94-0.95 | High (Gumbel in Phase 1) |

**Bottom line:** Switch from Schedule A (joint Gumbel-Softmax) to Schedule C (staged palette-then-indices with re-quantization). This:

1. **Saves 3.56 GB memory** (no `index_logits` tensor).
2. **Converges faster** (no Gumbel noise, no τ schedule).
3. **Achieves higher cos** (re-quantization escapes the k-means local optimum).
4. **Is simpler** (hard path only, no STE, no Gumbel sampling).

The re-quantization step is the key innovation: it allows the indices to update based on the current palette, breaking out of the k-means local optimum that Schedule A is stuck in. The `freeze_settled_palettes` function (already implemented at `train_qwen.py:246-299`) prevents oscillation.

The next document (`07_literature_comparison.md`) compares our approach to GPTQ, AWQ, SqueezeLLM, LLT/LoftQ, and LUT-Q/FLUTE in detail, with arxiv citations.
