# Message: Multi-agent infrastructure created — Round 3 Triton fusion

**TO:** layer-fusion
**FROM:** orchestrator
**TIMESTAMP:** 2026-08-23T00:00:00Z
**SUBJECT:** Round 3 infrastructure ready — you fuse the ENTIRE Qwen3.5 layer

The Round 3 multi-agent infrastructure is created. You are the most complex agent — 4 patches fusing RMSNorm, MLP, attention, and GatedDeltaNet into Triton.

## Your Assignment
- **Branch:** `agent/layer-fusion`
- **Patches:** 10 (fused RMSNorm+Linear), 12 (fused MLP), 13 (fused attention), 14 (fused GatedDeltaNet)
- **Waves:** 2 (Patches 10, 12) + 3 (Patches 13, 14)

## What To Do
1. Read `agent-ctx/ROADMAP.md` for the full plan
2. Read `agent-ctx/agent-layer-fusion/TASKS.md` for your wave structure
3. Read `agent-ctx/agent-layer-fusion/RULES.md` for file ownership
4. Check your inbox at the start of every wave
5. **WAIT** for nn-module-foundation's "forward signature ready" message before starting Patch 10

## Dependencies
- Patch 10, 12 depend on Patch 11 (nn-module forward signature)
- Patch 13, 14 depend on Patch 15 (batched compute_P_W from triton-kernels)

## Research References
- `research-kernel-efficiency/00_overview.md` §3 — fused layer pattern
- `docs/papers/2307.08691` (FlashAttention2) — fused attention
- `docs/papers/2312.00752` (Mamba) — SSM-style fused kernel
- `docs/papers/2002.05202` (SwiGLU) — fused MLP

**ACTION REQUIRED:** WAIT for nn-module-foundation. While waiting, read the research files and design your Triton kernels on paper.
