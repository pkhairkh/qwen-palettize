# FIX AGENT: optimizer-streams — Fix Conflicts + Verify Stream Logic

> **Branch:** `agent/optimizer-streams`
> **Issues found:** 3 (train_qwen.py conflict potential, stream logic untested, PROGRESS.md overwrite)

## Context

Your Round 1 work replaced FP32MasterAdamW with bnb.optim.AdamW8bit (Patch 8) and added stream double-buffering (Patch 6). The code looks structurally correct, but there are potential issues:

### Issue 1: train_qwen.py conflicts with other agents

**File:** `scripts/train_qwen.py`

**Problem:** You changed lines ~591-627 (build_optimizers), ~989-993 (scheduler), ~1047-1095 (training loop stream). Other agents changed lines ~632-736 (nn-module-foundation) and ~1031-1040 (training-recipe). There WILL be merge conflicts.

**Fix:** Rebase on main first. Resolve conflicts carefully — your changes to build_optimizers and the training loop are in DIFFERENT line ranges than the other agents, but the line numbers shift after their changes.

### Issue 2: Stream double-buffer logic untested

**File:** `scripts/train_qwen.py`, training loop (~line 1047-1095)

**Problem:** The double-buffer logic is complex (h_out_buf[2], event_t[2], event_s[2], buf_idx). Potential issues:
- Prologue: teacher forward for step 0 — does it run before the loop?
- Epilogue: what happens on the last step? Does `next(data_stream)` raise StopIteration?
- Event sync: `torch.cuda.current_stream().wait_event(event_t[curr])` — is this correct?

**Fix:** Review the logic carefully. Add a try/except around `next(data_stream)` for the epilogue. Verify the event recording order.

### Issue 3: PROGRESS.md overwritten

**Fix:** `git checkout origin/main -- agent-ctx/PROGRESS.md`, then append.

## Tasks

### Fix 1: Rebase on main + resolve conflicts
- `git pull origin main`
- Resolve conflicts in `train_qwen.py`:
  - Your lines: 591-627 (build_optimizers), 989-993 (scheduler), 1047-1095 (stream)
  - Other agents' lines: 632-736 (nn-module), 1031-1040 (training-recipe)
  - These are DIFFERENT sections, so conflicts should be minimal (line shifts only)
- Commit: `Fix 1: rebase on main + resolve conflicts`

### Fix 2: Fix stream double-buffer epilogue
- Find the `next(data_stream)` call in the stream double-buffer code
- Wrap in try/except StopIteration:
  ```python
  if step + 1 < max_steps:
      try:
          next_batch = next(data_stream)
      except StopIteration:
          break
  ```
- Verify event recording order: student records event_s AFTER backward, teacher waits on event_s[nxt] before starting
- Commit: `Fix 2: fix stream double-buffer epilogue (StopIteration handling)`

### Fix 3: Verify bnb.optim.AdamW8bit eps
- Verify `eps=1e-6` is set (you raised it in Wave 3 — good)
- Verify the scheduler uses `opt_indices` not `opt_indices.opt` (no wrapper)
- Commit: `Fix 3: verify AdamW8bit eps + scheduler init`

### Fix 4: Reset PROGRESS.md
- `git checkout origin/main -- agent-ctx/PROGRESS.md`
- Append your status
- Commit: `Fix 4: reset PROGRESS.md + append status`

## DoD
- [ ] Branch rebased on main, conflicts resolved
- [ ] Stream double-buffer epilogue handles StopIteration
- [ ] bnb.optim.AdamW8bit eps=1e-6 verified
- [ ] Scheduler uses opt_indices (not .opt)
- [ ] PROGRESS.md reset to main + appended
- [ ] syntax check passes
- [ ] Branch pushed
