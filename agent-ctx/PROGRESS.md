# Global Progress Tracker

> **Updated by:** Orchestrator + each agent (append-only, never edit existing entries)

---

## Agent Status

| Agent | Branch | Wave 1 | Wave 2 | Wave 3 | Merged |
|-------|--------|--------|--------|--------|--------|
| nn-module-foundation | `agent/nn-module-foundation` | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ |
| training-recipe | `agent/training-recipe` | ✅ Done | ✅ Done | ✅ Done | ⬜ |
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
| 9 | PartialWrapper → nn.Module | nn-module-foundation | ⬜ | — | — |
| 2 | LoftQ SVD init for LoRA | nn-module-foundation | ⬜ | — | — |
| 1 | Polynomial τ schedule (floor 0.5) | training-recipe | ✅ | `agent/training-recipe` | `012e820` |
| 3 | Adaptive logit clamp ±5τ | training-recipe | ✅ | `agent/training-recipe` | `c0a80e4` |
| 4 | Group size 256→128 | training-recipe | ✅ | `agent/training-recipe` | `d72ba5e` |
| 5 | Fused bwd with AoS P layout | kernels | ⬜ | — | — |
| 7 | Batched compute_P_W (25→1) | kernels | ⬜ | — | — |
| 8 | Fused AdamW (bitsandbytes 8-bit) | optimizer-streams | ⬜ | — | — |
| 6 | Stream double-buffering | optimizer-streams | ⬜ | — | — |

---

## Event Log (append-only)

| Timestamp | Agent | Event |
|-----------|-------|-------|
| 2025-08-22T12:00:00Z | orchestrator | Created agent-ctx infrastructure + 4 branches |
| 2026-08-22T08:30:00Z | training-recipe | Wave 1 / Patch 1a: tau CLI defaults updated (tau_final 0.1->0.5, tau_anneal_steps 4000->6000). Commit `957ad62`. |
| 2026-08-22T08:35:00Z | training-recipe | Wave 1 / Patch 1b: linear tau anneal replaced with piecewise warmup (500 steps) + quadratic decay (6000 steps) + hold at 0.5. Commit `012e820`. Syntax check + math sim both pass. |
| 2026-08-22T08:36:00Z | training-recipe | Wave 1 complete. Pushed to origin/agent/training-recipe. Sent inbox message to nn-module-foundation (informational — no conflicts). |
| 2026-08-22T09:00:00Z | training-recipe | Wave 2 / Patch 3: replaced fixed ±20 logit clamp with adaptive ±5τ at scripts/train_qwen.py:1192 (was 1153 — shifted by Patch 1b's expanded comment). Commit `c0a80e4`. Verified tau in scope via static AST check. Inbox empty (no RELEASED msg yet) — no rebase needed since origin/main unchanged. |
| 2026-08-22T09:25:00Z | training-recipe | Wave 3 / Patch 4a: GROUP_SIZE 256→128 in palettize_core.py:26. Commit `d72ba5e`. Verified all 6 downstream consumers (palettize_core, qwen_model, calib_qwen, calib_stage2, sweep_qwen, convert_trained_to_packed) correctly see GROUP_SIZE=128. |
| 2026-08-22T09:26:00Z | training-recipe | Wave 3 / Patch 4b SKIPPED (optional per-tensor override — adds complexity without clear benefit; orchestrator can apply later if checkpoint preservation becomes important). |
| 2026-08-22T09:27:00Z | training-recipe | Wave 3 / Patch 4c: merge prep done. `git fetch origin` + dry-run merge with origin/main = "Already up to date" (origin/main is strict ancestor). No conflicts. RE-CALIBRATION REQUIRED before resuming training (existing GS=256 checkpoints incompatible with new GS=128). |
| 2026-08-22T09:28:00Z | training-recipe | Wave 3 complete. All 3 patches (1, 3, 4) done. Pushed to origin/agent/training-recipe. Sent final inbox message to orchestrator: "training-recipe ready for merge". |

---

## Inbox Summary (last message per agent)

| Agent | Last Message | From | Subject | Action Required |
|-------|--------------|------|---------|-----------------|
| nn-module-foundation | 2026-08-22T08:36Z | training-recipe | Patch 1 (τ schedule) done on agent/training-recipe | nothing |
| orchestrator | 2026-08-22T09:28Z | training-recipe | training-recipe ready for merge | merge branch |
| training-recipe | — | — | — | — |
| kernels | — | — | — | — |
| optimizer-streams | — | — | — | — |
