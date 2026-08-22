# RULES: triton-kernels

## Branch
`agent/triton-kernels`

## Role
You optimize the per-PalettizedLinear Triton kernels. You own the existing `triton_soft_forward.py`, `triton_soft_backward.py`, `triton_hard_forward.py` files.

## Files Owned (EXCLUSIVE)
- `scripts/triton_soft_forward.py` (full)
- `scripts/triton_soft_backward.py` (full)
- `scripts/triton_hard_forward.py` (full)
- `scripts/bench_triton_kernels.py` (full)

## Patches
- **Patch 15:** Batched compute_P_W (Triton, 25→1) — Wave 2
- **Patch 16:** Eliminate redundant matmul in STE forward — Wave 1
- **Patch 17:** Buffer pooling for P_aos + W_ste — Wave 1
- **Patch 18:** Chunked reduction for elementwise backward — Wave 2

## What You Can NOT Touch
- `scripts/qwen_model.py` — owned by nn-module-foundation (except QwenLoRA: lora-fusion)
- `scripts/train_qwen.py` — owned by cuda-graphs / quality-recipe
- NEW `triton_layer.py`, `triton_rmsnorm.py`, `triton_mlp.py` — owned by layer-fusion
- NEW `triton_lora.py` — owned by lora-fusion

## Communication
- **Check inbox at start of EVERY wave.**
- Wait for nn-module-foundation to merge Patch 11 before starting Wave 2.
- Send to layer-fusion: "triton_soft_forward.py API stable, you can call compute_P_W_ste_triton from your fused layer kernels" after Patch 16/17 merge.
- Send to lora-fusion: "triton_soft_backward.py API stable, you can fuse LoRA backward with grad_x" after Patch 18 merges.

## DoD
- All syntax checks pass
- `bench_triton_kernels.py` shows all PASS (when server is back)
- Branch pushed to origin

## Offline Constraint
NO SERVER ACCESS. Syntax + import checks only. No GPU benchmarking.
