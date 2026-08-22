# Research Filter Consolidation — Index

> Navigation for the filtered, consolidated research output.

## Folder: `research-filter-consolidation/`

| File | Description |
|------|-------------|
| [`01_training_recipe.md`](01_training_recipe.md) | Patches 1-4: τ schedule, LoftQ init, logit clamp, group size |
| [`02_kernel_efficiency.md`](02_kernel_efficiency.md) | Patches 5-7: fused bwd AoS, stream double-buffer, batched compute_P_W |
| [`03_optimizer_speedup.md`](03_optimizer_speedup.md) | Patches 8-9: fused AdamW, PartialWrapper→nn.Module |
| [`06_ENHANCEMENT_ROADMAP.md`](06_ENHANCEMENT_ROADMAP.md) | **Master document** — 9 patches with dependencies, testing plan, risks |
| [`07_INDEX.md`](07_INDEX.md) | This file |

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
2. **For each patch**, read the detail in `01`/`02`/`03` files
3. **Refer to original research** in `research-kernel-accuracy/`, `research-kernel-efficiency/`, `research-indices-training/`, `research-palettes-training/`, `research-architecture-review/`, `research-literature-review/` for deeper analysis

## Recommended Implementation Order

```
Patch 9 (nn.Module) → Patch 1 (τ) → Patch 2 (LoftQ) → Patch 3 (clamp) →
Patch 4 (GS) → Patch 5 (fused bwd) → Patch 7 (batched) → Patch 6 (streams) → Patch 8 (AdamW)
```

See `06_ENHANCEMENT_ROADMAP.md` §5 for dependency graph.
