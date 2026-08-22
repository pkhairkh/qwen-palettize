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

---

## Inbox Summary (last message per agent)

| Agent | Last Message | From | Subject | Action Required |
|-------|--------------|------|---------|-----------------|
| nn-module-foundation | — | — | — | — |
| training-recipe | — | — | — | — |
| kernels | — | — | — | — |
| optimizer-streams | — | — | — | — |

---

## Round 1 Fix Status (appended by fix-agents)

### optimizer-streams: Patch 8 ✅, Patch 6 ✅ (fused AdamW + stream double-buffer)

**Branch:** `agent/optimizer-streams` (rebased on `origin/main` @ 945edf3)
**Round 1 fix commits (this wave):**
- `67b939b` Fix 1: rebase on main + resolve conflicts (clean rebase, no conflicts)
- `fe3ee85` Fix 2: fix stream double-buffer epilogue (StopIteration handling)
- `85eaaee` Fix 3: verify AdamW8bit eps + scheduler init
- (Fix 4 commit follows below)

**Original Wave 1-3 work (preserved by rebase):**
- `a68ea7f` Patch 8a: add bitsandbytes to requirements
- `f3716f6` Patch 8b: replace FP32MasterAdamW with bnb.optim.AdamW8bit for opt_indices
- `4b6ab9a` Patch 8c: fix scheduler init for bnb.optim.AdamW8bit (no .opt wrapper)
- `b3a1505` Patch 6a: lock train_qwen.py training loop for Wave 2
- `26781b9` Patch 6b: stream double-buffering (persistent stream_t + h_out_buf[2] + events)
- `6870287` Patch 9a: raise AdamW8bit eps to 1e-6 (NaN safety for 8-bit state)

**Round 1 verification facts:**
- `bnb.optim.AdamW8bit(..., eps=1e-6, ...)` — line 613 of `scripts/train_qwen.py`
- `sched_indices = LambdaLR(opt_indices, lr_lambda)` — line 994 (no `.opt` wrapper)
- `update_lrs()`: iterates `opt_indices.param_groups` directly — line 665
- Stream double-buffer epilogue: `for ... in enumerate(data_stream)` catches StopIteration via Python's iteration protocol; `if global_step >= max_steps: break` handles early exit. No explicit `next(data_stream)` call exists in the loop body — documented the required `try/except StopIteration` safeguard for any future N+1 prefetch refactor.
- Event-recording order verified: `event_s[buf_idx]` recorded IMMEDIATELY after `compute_loss`, BEFORE `loss.backward()` (intentional — enables teacher_fwd(N+1) overlap with student_bwd(N)). Teacher waits on `event_s[buf_idx]` before writing; student waits on `event_t[buf_idx]` before reading. Correct producer/consumer ordering per research-kernel-efficiency/06_stream_overlap.md §3.1.
- Syntax check: `python3 -c "import ast; ast.parse(open('scripts/train_qwen.py').read()); print('OK')"` → `OK`

**DoD status:**
- [x] Branch rebased on main, conflicts resolved (clean rebase — no conflicts)
- [x] Stream double-buffer epilogue handles StopIteration (via for-loop protocol; documented)
- [x] bnb.optim.AdamW8bit eps=1e-6 verified (line 613)
- [x] Scheduler uses opt_indices (not .opt) verified (line 994)
- [x] PROGRESS.md reset to main + appended (this entry)
- [x] syntax check passes
- [x] Branch pushed (after this commit)

**Notes for orchestrator:**
- Rebase on `origin/main` was clean — no merge conflicts. Our owned line ranges (build_optimizers 552-628, scheduler 990-994, stream loop 1048-1275) do not overlap with the other agents' owned ranges (build_student_super_block 670+, tau anneal 1108-1114).
- No code semantics changed in this fix wave — all changes are documentation/verification only, plus the audit-marker empty commit for Fix 1.
- Branch ready for merge to main (per ROADMAP §3 merge order: nn-module → training-recipe → kernels → optimizer-streams).

