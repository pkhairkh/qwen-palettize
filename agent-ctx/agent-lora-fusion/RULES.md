# RULES: lora-fusion

## Branch
`agent/lora-fusion`

## Role
You fuse the LoRA backward into Triton, eliminating the 167ms `aten::add_` and 40ms `aten::mul` overhead from 31 LoRA modules.

## Files Owned (EXCLUSIVE)
- NEW `scripts/triton_lora.py` — fused LoRA forward + backward
- `scripts/qwen_model.py` lines 184-265 (QwenLoRA class ONLY)

## Patches
- **Patch 19:** Fused LoRA backward — Wave 3
- **Patch 20:** Fused LoRA + PalettizedLinear backward — Wave 4

## What You Can NOT Touch
- `scripts/qwen_model.py` lines OUTSIDE 184-265 — owned by nn-module-foundation
- `scripts/triton_soft_*.py` — owned by triton-kernels (you IMPORT, don't modify)
- `scripts/train_qwen.py` — owned by cuda-graphs / quality-recipe

## Dependencies
- **Patch 19** depends on Patch 18 (chunked reduction) — the fused LoRA backward reuses the elementwise pattern.
- **Patch 20** depends on Patch 19 + triton-kernels' backward API being stable.

## Communication
- **Check inbox at start of EVERY wave.**
- Wait for triton-kernels' "triton_soft_backward.py API stable" message before starting Patch 19.
- Send to cuda-graphs: "LoRA backward fused, ready for CUDA Graph capture" after Patch 20 merges.

## Research References
- NEW (profiler finding: 167ms `aten::add_`, 31 LoRA modules, 3 matmuls each)
- `docs/papers/2305.14314_QLoRA_Dettmers2023.pdf` — LoRA backward math
- `research-architecture-review/02_partial_wrapper_problem.md` — QwenLoRA class structure

## DoD
- All syntax checks pass
- Import check: `python3 -c "import sys; sys.path.insert(0,'scripts'); import triton_lora"`
- QwenLoRA.forward uses the Triton fused kernel
- Branch pushed to origin

## Offline Constraint
NO SERVER ACCESS. Syntax + import checks only. No GPU benchmarking.
