# Message: Multi-agent infrastructure created — Round 3 Triton fusion

**TO:** lora-fusion
**FROM:** orchestrator
**TIMESTAMP:** 2026-08-23T00:00:00Z
**SUBJECT:** Round 3 infrastructure ready — you fuse the LoRA backward

The Round 3 multi-agent infrastructure is created. You fuse the LoRA backward into Triton, eliminating the 167ms `aten::add_` overhead from 31 LoRA modules.

## Your Assignment
- **Branch:** `agent/lora-fusion`
- **Patches:** 19 (fused LoRA backward), 20 (fused LoRA + PalettizedLinear backward)
- **Waves:** 3 (Patch 19) + 4 (Patch 20)

## What To Do
1. Read `agent-ctx/ROADMAP.md` for the full plan
2. Read `agent-ctx/agent-lora-fusion/TASKS.md` for your wave structure
3. Read `agent-ctx/agent-lora-fusion/RULES.md` for file ownership
4. Check your inbox at the start of every wave
5. **WAIT** for triton-kernels' "triton_soft_backward.py API stable" message before starting Patch 19

## Dependencies
- Patch 19 depends on Patch 18 (chunked reduction from triton-kernels)
- Patch 20 depends on Patch 19 + triton-kernels' backward API being stable

## Research References
- `docs/papers/2305.14314_QLoRA_Dettmers2023.pdf` — LoRA backward math
- Profiler finding: 167ms `aten::add_` (1070 calls), 31 LoRA modules × 3 matmuls each

**ACTION REQUIRED:** WAIT for triton-kernels. While waiting, read `qwen_model.py:184-265` (QwenLoRA class) and design your fused kernel.
