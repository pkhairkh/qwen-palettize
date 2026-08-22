# Global Progress Tracker

> **Updated by:** Orchestrator + each agent (append-only, never edit existing entries)

---

## Agent Status

| Agent | Branch | Wave 1 | Wave 2 | Wave 3 | Merged |
|-------|--------|--------|--------|--------|--------|
| nn-module-foundation | `agent/nn-module-foundation` | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ |
| training-recipe | `agent/training-recipe` | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ |
| kernels | `agent/kernels` | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ |
| optimizer-streams | `agent/optimizer-streams` | ✅ Done | ✅ Done | ✅ Done | ⬜ |

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
| 5 | Fused bwd with AoS P layout | kernels | ⬜ | — | — |
| 7 | Batched compute_P_W (25→1) | kernels | ⬜ | — | — |
| 8 | Fused AdamW (bitsandbytes 8-bit) | optimizer-streams | ✅ | agent/optimizer-streams | 02cff8e (eps=1e-6) |
| 6 | Stream double-buffering | optimizer-streams | ✅ | agent/optimizer-streams | a2576f5 |

---

## Event Log (append-only)

| Timestamp | Agent | Event |
|-----------|-------|-------|
| 2025-08-22T12:00:00Z | orchestrator | Created agent-ctx infrastructure + 4 branches |
| 2026-08-22T11:40:00Z | optimizer-streams | Wave 1 complete: Patch 8 (fused AdamW via bnb.optim.AdamW8bit) — 3 commits (8a 8b 8c) pushed to agent/optimizer-streams @8c72021. Wave 2 not yet started. |
| 2026-08-22T11:44:00Z | optimizer-streams | Wave 2 started: LOCK message sent to training-recipe for train_qwen.py training loop (commit e778f56). |
| 2026-08-22T11:46:00Z | optimizer-streams | Wave 2 complete: Patch 6 (stream double-buffering) — persistent stream_t + h_out_buf[2] + event_t/s[2] (commit a2576f5). RELEASED message sent to training-recipe. |
| 2026-08-22T11:55:00Z | optimizer-streams | Wave 3 complete: (9a) raised AdamW8bit eps to 1e-6 for 8-bit state NaN safety (commit 02cff8e); (9b) git merge origin/main → "Already up to date" (no conflicts, no rebase needed since main hasn't moved since clone); (9c) branch ready for merge to main. All 3 waves done. Branch: agent/optimizer-streams @ 1c03465. |

---

## Inbox Summary (last message per agent)

| Agent | Last Message | From | Subject | Action Required |
|-------|--------------|------|---------|-----------------|
| nn-module-foundation | — | — | — | — |
| training-recipe | RELEASED: stream double-buffer done | optimizer-streams | RELEASED: stream double-buffer done | nothing |
| kernels | — | — | — | — |
| optimizer-streams | — | — | — | — |
