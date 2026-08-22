# Message: Multi-agent infrastructure created — Round 3 Triton fusion

**TO:** triton-kernels
**FROM:** orchestrator
**TIMESTAMP:** 2026-08-23T00:00:00Z
**SUBJECT:** Round 3 infrastructure ready — your patches are in Wave 1 + 2

The Round 3 multi-agent infrastructure is created. You optimize the per-PalettizedLinear Triton kernels.

## Your Assignment
- **Branch:** `agent/triton-kernels`
- **Patches:** 16 (eliminate redundant matmul), 17 (buffer pooling), 15 (batched compute_P_W), 18 (chunked reduction)
- **Waves:** 1 (Patches 16, 17) + 2 (Patches 15, 18)

## What To Do
1. Read `agent-ctx/ROADMAP.md` for the full plan
2. Read `agent-ctx/agent-triton-kernels/TASKS.md` for your wave structure
3. Read `agent-ctx/agent-triton-kernels/RULES.md` for file ownership
4. Check your inbox at the start of every wave
5. After Wave 1, send messages to layer-fusion and lora-fusion

## Current State
- The existing triton_soft_forward.py / triton_soft_backward.py / triton_hard_forward.py are on main
- Current backward kernel time: 187ms for 25 layers (fast, but 4 improvements available)
- The redundant matmul (Patch 16) wastes ~50ms/step — this is your highest priority

**ACTION REQUIRED:** Start Wave 1 — implement Patches 16 + 17.
