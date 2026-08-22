# RULES: layer-fusion

## Branch
`agent/layer-fusion`

## Role
You fuse the ENTIRE Qwen3.5 layer into Triton — RMSNorm, MLP, attention, GatedDeltaNet. This is the most complex agent.

## Files Owned (EXCLUSIVE)
- NEW `scripts/triton_rmsnorm.py` — fused RMSNorm + Linear
- NEW `scripts/triton_mlp.py` — fused gate + up + SiLU + down
- NEW `scripts/triton_layer.py` — fused attention + GatedDeltaNet

## Patches
- **Patch 10:** Fused RMSNorm + Linear — Wave 2
- **Patch 12:** Fused MLP (gate+up+SiLU+down) — Wave 2
- **Patch 13:** Fused Attention (FlashAttention-style) — Wave 3
- **Patch 14:** Fused GatedDeltaNet (conv1d + delta-rule) — Wave 3

## What You Can NOT Touch
- `scripts/qwen_model.py` — owned by nn-module-foundation (but you can REQUEST forward signature changes via inbox)
- `scripts/triton_soft_*.py` — owned by triton-kernels (you IMPORT from these, don't modify)
- `scripts/train_qwen.py` — owned by cuda-graphs / quality-recipe

## Dependencies
- **Patch 10, 12** depend on Patch 11 (nn-module forward signature) — wait for nn-module to merge.
- **Patch 13, 14** depend on Patch 15 (batched compute_P_W) for the attention/GatedDeltaNet to call the batched kernel.

## Communication
- **Check inbox at start of EVERY wave.**
- Wait for nn-module-foundation's "forward signature ready" message before starting Patch 10.
- Wait for triton-kernels' "API stable" message before starting Patch 13/14.
- Send to cuda-graphs: "layer kernels ready for CUDA Graph capture" after Patch 13/14 merge.

## Research References
- `research-kernel-efficiency/00_overview.md` §3 — fused layer pattern
- `research-kernel-accuracy/00_overview.md` — GatedDeltaNet architecture
- `docs/papers/2307.08691` (FlashAttention2) — fused attention pattern
- `docs/papers/2312.00752` (Mamba) — SSM-style fused kernel
- `docs/papers/2002.05202` (SwiGLU) — fused MLP activation

## DoD
- All syntax checks pass
- Import check: `python3 -c "import sys; sys.path.insert(0,'scripts'); import triton_rmsnorm, triton_mlp, triton_layer"`
- Branch pushed to origin

## Offline Constraint
NO SERVER ACCESS. Syntax + import checks only. No GPU benchmarking.
