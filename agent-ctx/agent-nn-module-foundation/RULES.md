# RULES: nn-module-foundation

## Branch
`agent/nn-module-foundation`

## Role
**Foundation agent.** You prepare the `nn.Module` infrastructure so that `torch.compile` and fused Triton kernels can work. You merge FIRST (Wave 1).

## Files Owned (EXCLUSIVE)
- `scripts/qwen_model.py` — lines 47-48 (is_gated_delta_layer), 65-180 (PalettizedLinear class), 438-600 (PartialModel/PartialWrapper — already done), 779-850 (capture_original_weights, insert_correction_layers)
- `scripts/train_qwen.py` — lines 154-214 (FP32MasterOptimizer), 540-600 (build_optimizers), 632-736 (build_student_super_block)

## Patches
- **Patch 11:** nn.Module forward signature for fused RMSNorm

## What You Can NOT Touch
- `scripts/triton_*.py` — owned by triton-kernels
- `scripts/train_qwen.py` lines 96-102 (quality-recipe), 889-918 (cuda-graphs), 1034-1040 (quality-recipe), 1058-1250 (cuda-graphs)
- Any file in `research-*/` — read-only reference

## Communication
- **Check inbox at start of EVERY wave.** Process all messages before starting work.
- **Send messages BEFORE merging** to warn others of upcoming changes.
- Send "RELEASED: nn.Module forward signature merged — rebase your branches" after Patch 11 merges.
- Send to layer-fusion: "forward signature ready, you can implement fused RMSNorm" after Patch 11.

## DoD
- `python3 -c "import ast; ast.parse(open('scripts/qwen_model.py').read())"` passes
- `python3 -c "import ast; ast.parse(open('scripts/train_qwen.py').read())"` passes
- `PalettizedLinear.forward` accepts optional `out_norm` parameter
- `PartialModel` properly delegates to `nn.ModuleList(layers)`
- Branch pushed to origin

## Offline Constraint
NO SERVER ACCESS. All work is code changes + syntax checks. No GPU testing.
