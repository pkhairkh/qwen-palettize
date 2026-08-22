# RULES: cuda-graphs

## Branch
`agent/cuda-graphs`

## Role
You capture the entire training step as a CUDA Graph and integrate it with the stream double-buffer. You merge LAST (Wave 4).

## Files Owned (EXCLUSIVE)
- `scripts/train_qwen.py` lines 889-918 (data pipeline), 1058-1103 (stream double-buffer), 1095-1250 (training loop — CUDA Graph capture)

## Patches
- **Patch 21:** CUDA Graph capture for full step — Wave 4
- **Patch 22:** Stream double-buffer + CUDA Graphs integration — Wave 4

## What You Can NOT Touch
- `scripts/qwen_model.py` — owned by nn-module-foundation / lora-fusion
- `scripts/triton_*.py` — owned by triton-kernels / layer-fusion / lora-fusion
- `scripts/train_qwen.py` lines OUTSIDE 889-918, 1058-1250 — owned by quality-recipe / nn-module-foundation

## Dependencies
- **Patch 21, 22** depend on ALL other patches being merged. The CUDA Graph capture requires all kernels to be stable (no shape changes, no new intermediates).
- You merge LAST. All other agents must be done before you start.

## Communication
- **Check inbox at start of EVERY wave.**
- Wait for layer-fusion's "layer kernels ready" message AND lora-fusion's "LoRA backward fused" message before starting Patch 21.
- Send to orchestrator: "CUDA Graphs ready, tps target achievable" after Patch 22 merges.

## Research References
- `research-kernel-efficiency/08_recommendations.md` §11 — CUDA Graphs patch
- `research-kernel-efficiency/06_stream_overlap.md` — stream double-buffer pattern

## DoD
- All syntax checks pass
- CUDA Graph capture code present in train_qwen.py
- Static input buffers (batch_ids) pre-allocated
- Stream double-buffer integrated with graph replay
- Branch pushed to origin

## Offline Constraint
NO SERVER ACCESS. Syntax + import checks only. No GPU benchmarking. CUDA Graph capture code can be written but NOT tested (requires GPU).
