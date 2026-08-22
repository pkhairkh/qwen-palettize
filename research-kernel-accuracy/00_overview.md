# 00 — Executive Summary

**Problem:** Our 2-bit palettized Qwen3.5-4B training plateaus at cos ≈ 0.946 after 8,000 L4 steps + 500 Blackwell steps. The target is cos > 0.999 (GPTQ/AWQ-level quality). The 2,208 palette parameters (the 4 LUT entries per group of 256 weights) are not training optimally, contributing to the plateau.

**Method:** We conducted a 4-wave audit of the `qwen-palettize` codebase, reading every relevant file in `scripts/` and `logs/`, cross-referencing against the literature (12 arxiv papers), and deriving the gradient formulas from first principles.

**Key finding:** The gradient formula is mathematically correct (both hard and soft paths). The plateau is caused by **eight compounding issues**, none of which is a backward-pass bug:

1. **Palette stored as bf16** (`qwen_model.py:82-85`) — 0.78% relative precision per value, comparable to the gap between adjacent palette entries.
2. **Double precision loss on load** — k-means (fp32) → disk (fp16) → load (fp32) → parameter (bf16), losing 16 mantissa bits total.
3. **Actual loss is `norm_mse`, not `1-cos+norm_mse`** as the brief claimed (`train_qwen.py:96`). The cosine term is monitored but not in the loss.
4. **Global gradient clip = 0.3** with 1.78B index_logits in the norm — palette gradient is scaled by ~1/45 of its raw value.
5. **LoRA uses `init="loftq"` but `original_weight=None`** (`train_qwen.py:718`) — falls back to zero-init B instead of SVD warm start.
6. **K-means is already at a local optimum of the L2 objective** — gradient descent on the same objective converges to the same solution, providing only 1-3% improvement from off-diagonal Hessian terms.
7. **Gumbel-Softmax structural limitation** — as τ → 0, only the argmax palette slot receives gradient; non-argmax slots are stuck.
8. **LR justification is stale** — the comment at `train_qwen.py:84-87` refers to "Muon scale 0.63" but palettes use AdamW (no scale factor).

**Recommended fixes (priority order):**

| Priority | Fix | Expected cos gain | Code change |
|---|---|---|---|
| 1 | Enable LoftQ SVD init for LoRA | +1-2% | One-line: pass `original_weight` dict |
| 2 | Switch loss to `1-cos+norm_mse` with `cos=0.8, mse=0.2` | +1-2% | Two-line config change |
| 3 | Promote palette to fp32 | +0.5-1% | One-line dtype change + kernel assertion relaxation |
| 4 | Per-group gradient clipping (not global) | +1-2% | Replace `clip_grad_norm_` with per-group clips |
| 5 | Implement LUT-Q-style re-quantization every 2000 steps | +2-3% | New `re_quantize_indices` function |
| 6 | Try GROUP_SIZE=128 for the 5 worst-cos Linears | +2-4% | Per-tensor GROUP_SIZE override |
| 7 | Revive `freeze_settled_palettes` (already implemented, dead code) | +0.5-1% | Add call in training loop |

**Expected cumulative improvement:** cos 0.946 → 0.96-0.97 with fixes 1-4 (one-day effort). cos 0.96-0.97 → 0.97-0.98 with fixes 5-7 (one-week effort). Reaching cos >0.99 requires either 3-bit quantization or GROUP_SIZE ≤ 64 (calibration-time change, not training-time).

**Fundamental limit:** 2-bit quantization with GROUP_SIZE=256 has an information-theoretic ceiling at cos ~0.95-0.97, confirmed by both our results and the FLUTE paper (Table 3). The 0.999 target is not achievable at 2-bit/GS=256; it requires either higher bitwidth or smaller group size.

**Document map:**

| File | Pages | Content |
|---|---|---|
| `01_palette_audit.md` | 7.0 | Reverse-engineering of the palette subsystem, 15 findings with file:line citations |
| `02_gradient_correctness.md` | 7.2 | Proof that grad_palette is mathematically correct in both paths |
| `03_precision_analysis.md` | 7.0 | bf16 vs fp32 numerical analysis with ULP bounds and subnormal ranges |
| `04_kmeans_vs_gradient.md` | 8.3 | Comparison of k-means, LUT-Q, LLT, GPTQ, AWQ approaches |
| `05_loss_function.md` | 4.0 | norm_mse vs 1-cos vs combined loss, with gradient derivations |
| `06_staged_training.md` | 4.7 | Joint vs staged training schedule comparison |
| `07_literature_comparison.md` | 6.7 | 12-method comparison with 12 arxiv citations |
| `08_recommendations.md` | 3.0 | Concrete code patches for the top 7 fixes |
| `09_references.md` | 2.0 | All arxiv URLs and GitHub repos |

**Total: 49.9 pages** (DoD: ≥30 pages).

**Bottom line:** The cos plateau at 0.95 is not a single bug — it is the multiplicative effect of eight issues, each of which contributes a 0.5-3% cos loss. The fixes are well-understood and mostly one-line changes. The biggest single win is enabling LoftQ SVD initialization for LoRA (already implemented, just not called). The medium-term win is implementing LUT-Q-style re-quantization (which escapes the k-means local optimum that gradient descent alone cannot escape).

The orchestrator should prioritize fixes 1-4 (one-day effort, expected cos 0.96-0.97) before attempting fixes 5-7 (one-week effort, expected cos 0.97-0.98). The 0.999 target is not achievable at 2-bit/GS=256 and should be re-evaluated.
