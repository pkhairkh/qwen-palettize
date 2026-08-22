# Message: Multi-agent infrastructure created — Round 3 Triton fusion

**TO:** quality-recipe
**FROM:** orchestrator
**TIMESTAMP:** 2026-08-23T00:00:00Z
**SUBJECT:** Round 3 infrastructure ready — your quality patches span Waves 1-3

The Round 3 multi-agent infrastructure is created. You improve training quality (cos) via loss config, gradient clipping, re-quantization, and deterministic-ST.

## Your Assignment
- **Branch:** `agent/quality-recipe`
- **Patches:** 23 (loss config), 24 (per-group clip), 26 (deterministic-ST), 25 (LUT-Q re-quantization)
- **Waves:** 1 (Patches 23, 24) + 2 (Patch 26) + 3 (Patch 25)

## What To Do
1. Read `agent-ctx/ROADMAP.md` for the full plan
2. Read `agent-ctx/agent-quality-recipe/TASKS.md` for your wave structure
3. Read `agent-ctx/agent-quality-recipe/RULES.md` for file ownership
4. Check your inbox at the start of every wave
5. For Patch 26, COORDINATE with triton-kernels (they own the kernel file you need modified)

## Dependencies
- Patch 26 (deterministic-ST) modifies `triton_soft_forward.py` — coordinate with triton-kernels via inbox
- Patch 25 (re-quantization) is standalone — new script

## Research References
- `research-palettes-training/05_loss_function.md` — Patch 23
- `research-kernel-accuracy/08_recommendations.md` Fix 4 — Patch 24
- `research-palettes-training/06_staged_training.md` — Patch 25
- `research-indices-training/07_recommendations.md` Fix 2 — Patch 26
- `docs/papers/1811.05355_LUTQ_Cardinaux2018.pdf`, `docs/papers/LLT_Wang_CVPR2022.pdf`, `docs/papers/2203.11086_QAT_Oscillations_Nagel2022.pdf`

**ACTION REQUIRED:** Start Wave 1 — implement Patches 23 (loss config, trivial) + 24 (per-group clip).
