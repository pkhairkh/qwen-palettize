# Global Progress Tracker

> **Updated by:** Orchestrator + each agent (append-only, never edit existing entries)

---

## Agent Status

| Agent | Branch | Wave 1 | Wave 2 | Wave 3 | Merged |
|-------|--------|--------|--------|--------|--------|
| nn-module-foundation | `agent/nn-module-foundation` | ✅ Done | 🔄 In Progress | ⬜ Pending | ⬜ |
| training-recipe | `agent/training-recipe` | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ |
| kernels | `agent/kernels` | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ |
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
| 9 | PartialWrapper → nn.Module | nn-module-foundation | ✅ Done | `agent/nn-module-foundation` | `1db364f` (9a), `96a0fdb` (9b), `26b9679` (9c) |
| 2 | LoftQ SVD init for LoRA | nn-module-foundation | 🔄 In Progress | `agent/nn-module-foundation` | — |
| 1 | Polynomial τ schedule (floor 0.5) | training-recipe | ⬜ | — | — |
| 3 | Adaptive logit clamp ±5τ | training-recipe | ⬜ | — | — |
| 4 | Group size 256→128 | training-recipe | ⬜ | — | — |
| 5 | Fused bwd with AoS P layout | kernels | ⬜ | — | — |
| 7 | Batched compute_P_W (25→1) | kernels | ⬜ | — | — |
| 8 | Fused AdamW (bitsandbytes 8-bit) | optimizer-streams | ⬜ | — | — |
| 6 | Stream double-buffering | optimizer-streams | ⬜ | — | — |

---

## Event Log (append-only)

| Timestamp | Agent | Event |
|-----------|-------|-------|
| 2025-08-22T12:00:00Z | orchestrator | Created agent-ctx infrastructure + 4 branches |
| 2026-08-22T11:45:54Z | nn-module-foundation | Wave 1 prep: sent LOCK messages to training-recipe, optimizer-streams, kernels (train_qwen.py:632-736) |
| 2026-08-22T11:52:00Z | nn-module-foundation | Patch 9a: PartialModel → nn.Module (commit `1db364f`) |
| 2026-08-22T11:55:00Z | nn-module-foundation | Patch 9b: PartialWrapper → nn.Module (commit `96a0fdb`) |
| 2026-08-22T11:57:00Z | nn-module-foundation | Patch 9c: verified load_qwen_super_block_only compatibility (commit `26b9679`) |
| 2026-08-22T11:57:57Z | nn-module-foundation | Wave 1 complete: sent RELEASED messages to training-recipe, optimizer-streams, kernels |

---

## Inbox Summary (last message per agent)

| Agent | Last Message | From | Subject | Action Required |
|-------|--------------|------|---------|-----------------|
| nn-module-foundation | — | — | — | — |
| training-recipe | 2026-08-22T11:57:57Z | nn-module-foundation | RELEASED: nn.Module merged — Patch 9 complete | rebase after orchestrator merges agent/nn-module-foundation to main |
| kernels | 2026-08-22T11:57:57Z | nn-module-foundation | Wave 1 complete (Patch 9 nn.Module) — no action needed | nothing |
| optimizer-streams | 2026-08-22T11:57:57Z | nn-module-foundation | RELEASED: nn.Module merged — Patch 9 complete | rebase after orchestrator merges agent/nn-module-foundation to main |
