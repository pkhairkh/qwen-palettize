# Global Progress Tracker

> **Updated by:** Orchestrator + each agent (append-only, never edit existing entries)

---

## Agent Status

| Agent | Branch | Wave 1 | Wave 2 | Wave 3 | Merged |
|-------|--------|--------|--------|--------|--------|
| nn-module-foundation | `agent/nn-module-foundation` | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ |
| training-recipe | `agent/training-recipe` | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ |
| kernels | `agent/kernels` | ✅ Done | ✅ Done | 🔄 In Progress | ⬜ |
| optimizer-streams | `agent/optimizer-streams` | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ |

**Legend:** ⬜ Pending | 🔄 In Progress | ✅ Done | ❌ Blocked

---

## Merge Order

1. ⬜ nn-module-foundation (foundation — must merge first)
2. ⬜ training-recipe (rebases on nn-module)
3. ⬜ kernels (independent — can merge anytime after Wave 1)
4. ⬜ optimizer-streams (rebases on nn-module + training-recipe)

---

## Patch Status

| # | Patch | Agent | Status | Branch | Commit |
|---|-------|-------|--------|--------|--------|
| 9 | PartialWrapper → nn.Module | nn-module-foundation | ⬜ | — | — |
| 2 | LoftQ SVD init for LoRA | nn-module-foundation | ⬜ | — | — |
| 1 | Polynomial τ schedule (floor 0.5) | training-recipe | ⬜ | — | — |
| 3 | Adaptive logit clamp ±5τ | training-recipe | ⬜ | — | — |
| 4 | Group size 256→128 | training-recipe | ⬜ | — | — |
| 5 | Fused bwd with AoS P layout | kernels | ✅ | `agent/kernels` | 73558af (5a), fdf5283 (5b), 554348f (5c), 164bd44 (5d) |
| 7 | Batched compute_P_W (25→1) | kernels | ✅ | `agent/kernels` | b26b209 (7a+7b), 9307d79 (7c) |
| 8 | Fused AdamW (bitsandbytes 8-bit) | optimizer-streams | ⬜ | — | — |
| 6 | Stream double-buffering | optimizer-streams | ⬜ | — | — |

---

## Event Log (append-only)

| Timestamp | Agent | Event |
|-----------|-------|-------|
| 2025-08-22T12:00:00Z | orchestrator | Created agent-ctx infrastructure + 4 branches |
| 2026-08-22T11:39:00Z | kernels | Wave 1 complete: Patch 5 (a–d) — AoS P layout + fused bwd kernel re-enabled |
| 2026-08-22T11:55:00Z | kernels | Wave 2 complete: Patch 7 (a–c) — batched compute_P_W kernel (25 launches → 1) |

---

## Inbox Summary (last message per agent)

| Agent | Last Message | From | Subject | Action Required |
|-------|--------------|------|---------|-----------------|
| nn-module-foundation | 2026-08-22T11:55Z | kernels | Batched compute_P_W available | coordinate |
| training-recipe | 2026-08-22T11:39Z | kernels | P layout changed to (K,N,4) AoS | coordinate |
| kernels | — | — | — | — |
| optimizer-streams | 2026-08-22T11:39Z | kernels | P layout changed to (K,N,4) AoS | coordinate |
