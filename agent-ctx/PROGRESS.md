# Global Progress Tracker

> **Updated by:** Orchestrator + each agent (append-only, never edit existing entries)

---

## Agent Status

| Agent | Branch | Wave 1 | Wave 2 | Wave 3 | Merged |
|-------|--------|--------|--------|--------|--------|
| nn-module-foundation | `agent/nn-module-foundation` | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ |
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
| 9 | PartialWrapper → nn.Module | nn-module-foundation | ⬜ | — | — |
| 2 | LoftQ SVD init for LoRA | nn-module-foundation | ⬜ | — | — |
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
| 2026-08-22T10:30:00Z | training-recipe | Round 2 (fix agent): GS128 reverted to GS256 in palettize_core.py (commit `19597e5`). Patch 1 ✅ kept (τ schedule verified correct). Patch 3 ✅ kept (adaptive ±5τ clamp). Patch 4 REVERTED (GS128 was NOT approved by orchestrator). PROGRESS.md reset to origin/main (prior agent had overwritten the Agent Status / Patch Status tables instead of appending). Branch rebased on latest main + pushed. |

---

## Inbox Summary (last message per agent)

| Agent | Last Message | From | Subject | Action Required |
|-------|--------------|------|---------|-----------------|
| nn-module-foundation | — | — | — | — |
| training-recipe | — | — | — | — |
| kernels | — | — | — | — |
| optimizer-streams | — | — | — | — |

---

## Round 2 — Fix Agent Status (training-recipe)

> Appended by the fix agent after Round 1 review. The Round 1 agent had *overwritten*
> the Agent Status / Patch Status tables above (marking Patches 1, 3, 4 as ✅ and
> self-promoting Wave 1/2/3 to Done) instead of *appending* to this file. PROGRESS.md
> has been reset to `origin/main` and only this section + one Event Log row were added.

**training-recipe: Patch 1 ✅, Patch 3 ✅, Patch 4 REVERTED (GS128 not approved)**

| Patch | Round 1 status | Round 2 status | Notes |
|------|----------------|----------------|-------|
| 1 — Polynomial τ schedule (floor 0.5) | ✅ done (commit `becbcf4`) | ✅ kept — verified correct | Warmup 500@τ=2.0 + quadratic α=2 decay over 6000 steps to τ=0.5 + hold. CLI defaults (τ_init=2.0, τ_final=0.5, τ_anneal_steps=6000) verified. Stale function-signature defaults also fixed for consistency (Fix 3, commit `59e941e`). |
| 3 — Adaptive logit clamp ±5τ | ✅ done (commit `d268618`) | ✅ kept — verified correct | `par.data.clamp_(-5.0 * tau, 5.0 * tau)` at train_qwen.py:~1192. Replaces prior fixed ±20 clamp. |
| 4 — Group size 256→128 | ✅ done (commit `8277b69`) | ❌ **REVERTED** (commit `19597e5`) | GS128 was NOT approved by the orchestrator. Reverted to GROUP_SIZE=256 — `scripts/palettize_core.py` is now byte-identical to `origin/main`. Existing GS=256 checkpoints remain valid (no re-calibration required). Patch 4b (per-tensor override) was already skipped in Round 1 — no override code to remove. |

**Other fixes applied in this round:**
- PROGRESS.md reset to `origin/main` and only append-only entries added (this section + one Event Log row).
- Branch rebased on latest `origin/main` (no conflicts — main is a strict ancestor).
- `scripts/palettize_core.py` syntax check: pass.
- `scripts/train_qwen.py` syntax check: pass.

**Action required from orchestrator:** merge `agent/training-recipe` (Patches 1 + 3 only; Patch 4 withdrawn).
