# Research Filter Consolidation — Index

> Navigation for the filtered, consolidated research output.

## Folder: `research-filter-consolidation/`

| File | Wave | Description |
|------|------|-------------|
| [`00_audit_table.md`](00_audit_table.md) | 1 | Classification of ALL 201 recommendations from 6 agents (101 KEEP, 99 REJECT) |
| [`01_training_recipe.md`](01_training_recipe.md) | 2 | Patches 1-4: τ schedule, LoftQ init, logit clamp, group size |
| [`02_kernel_efficiency.md`](02_kernel_efficiency.md) | 2 | Patches 5-7: fused bwd AoS, stream double-buffer, batched compute_P_W |
| [`03_optimizer_speedup.md`](03_optimizer_speedup.md) | 2 | Patches 8-9: fused AdamW, PartialWrapper→nn.Module |
| [`04_rejected_sidesteps.md`](04_rejected_sidesteps.md) | 3 | 65 rejected recommendations grouped by category |
| [`05_special_notes.md`](05_special_notes.md) | 3 | Critical context (1d-kmeans best, Lloyd-Max bullshit, server offline, ambiguities) |
| [`06_ENHANCEMENT_ROADMAP.md`](06_ENHANCEMENT_ROADMAP.md) | 4 | **Master document** — 9 patches with dependencies, testing plan, risks |
| [`07_INDEX.md`](07_INDEX.md) | 4 | This file |

## Patch Status

| # | Patch | Status | Expected Impact |
|---|-------|--------|-----------------|
| 1 | Polynomial τ schedule (floor 0.5) | RESEARCHED | +0.005-0.01 cos |
| 2 | LoftQ SVD init for LoRA | RESEARCHED | +0.01-0.02 cos |
| 3 | Adaptive logit clamp ±5τ | RESEARCHED | +0.005-0.015 cos |
| 4 | Group size 256→128 | RESEARCHED | +0.005-0.01 cos |
| 5 | Fused bwd with AoS P layout | RESEARCHED | 260ms→60ms backward, -7GB VRAM |
| 6 | Stream double-buffering | RESEARCHED | -69ms/step (teacher hidden) |
| 7 | Batched compute_P_W (25→1) | RESEARCHED | -36ms/step |
| 8 | Fused AdamW (bitsandbytes 8-bit) | RESEARCHED | 113ms→20ms, -14GB VRAM |
| 9 | PartialWrapper → nn.Module | RESEARCHED | Unlocks torch.compile, checkpointing |

**All 9 patches: RESEARCHED. 0 implemented.**

## How to Use This Folder

1. **Read `06_ENHANCEMENT_ROADMAP.md` first** — it's the master document
2. **Read `05_special_notes.md`** before implementing — critical context
3. **For each patch**, read the detail in `01`/`02`/`03` files
4. **Check `04_rejected_sidesteps.md`** if unsure whether something is a sidestep
5. **Refer to `00_audit_table.md`** for the full classification of all 201 recommendations

## Recommended Implementation Order

```
Patch 9 (nn.Module) → Patch 1 (τ) → Patch 2 (LoftQ) → Patch 3 (clamp) →
Patch 4 (GS) → Patch 5 (fused bwd) → Patch 7 (batched) → Patch 6 (streams) → Patch 8 (AdamW)
```

See `06_ENHANCEMENT_ROADMAP.md` §5 for dependency graph.
