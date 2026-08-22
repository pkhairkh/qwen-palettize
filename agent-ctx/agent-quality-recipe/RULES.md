# RULES: quality-recipe

## Branch
`agent/quality-recipe`

## Role
You improve training quality (cos) via loss config, gradient clipping, re-quantization, and deterministic-ST. Your changes are orthogonal to the kernel fusion work.

## Files Owned (EXCLUSIVE)
- `scripts/train_qwen.py` lines 96-102 (DEFAULT_HYPERPARAMS), 1034-1040 (τ schedule — verify, already done), 1130-1145 (clip_grad_norm_ section)
- `scripts/palettize_core.py` line 26 (GROUP_SIZE — if touching)
- NEW `scripts/re_quantize.py` (LUT-Q re-quantization script)

## Patches
- **Patch 23:** Loss config switch (1-cos+norm_mse, 80/20) — Wave 1
- **Patch 24:** Per-group gradient clipping — Wave 1
- **Patch 25:** LUT-Q re-quantization (step 2000, 4000) — Wave 3
- **Patch 26:** Deterministic-ST (remove Gumbel noise) — Wave 2

## What You Can NOT Touch
- `scripts/qwen_model.py` — owned by nn-module-foundation / lora-fusion
- `scripts/triton_*.py` — owned by triton-kernels / layer-fusion / lora-fusion
- `scripts/train_qwen.py` lines OUTSIDE 96-102, 1034-1040, 1130-1145

## Dependencies
- **Patch 26** (deterministic-ST) modifies `triton_soft_forward.py` — coordinate with triton-kernels via inbox. You provide the math (remove Gumbel), they implement the kernel change.
- **Patch 25** (re-quantization) is standalone — new script, called from train_qwen.py at step 2000/4000.

## Communication
- **Check inbox at start of EVERY wave.**
- For Patch 26, send to triton-kernels: "REQUEST: remove Gumbel noise from compute_P_W_ste_kernel. The forward should be `logits / tau → softmax → P` (no Gumbel sampling). See `research-indices-training/07_recommendations.md` Fix 2."
- Wait for triton-kernels' confirmation before finalizing Patch 26.

## Research References
- `research-palettes-training/05_loss_function.md` — Patch 23
- `research-kernel-accuracy/08_recommendations.md` Fix 4 — Patch 24
- `research-palettes-training/06_staged_training.md` — Patch 25
- `research-indices-training/07_recommendations.md` Fix 2 — Patch 26
- `docs/papers/1811.05355_LUTQ_Cardinaux2018.pdf` — LUT-Q pattern
- `docs/papers/LLT_Wang_CVPR2022.pdf` — deterministic-ST
- `docs/papers/2203.11086_QAT_Oscillations_Nagel2022.pdf` — oscillation prevention

## DoD
- All syntax checks pass
- Import check: `python3 -c "import sys; sys.path.insert(0,'scripts'); import re_quantize"`
- Branch pushed to origin

## Offline Constraint
NO SERVER ACCESS. Syntax + import checks only. No GPU testing. Re-quantization script can be written but not run.
