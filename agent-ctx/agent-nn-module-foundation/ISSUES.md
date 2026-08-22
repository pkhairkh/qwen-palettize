# FIX AGENT: nn-module-foundation — Fix Issues from Round 1

> **Branch:** `agent/nn-module-foundation`
> **Issues found:** 2 (LoftQ helper device bug, PROGRESS.md overwrite)

## Context

Your Round 1 work is mostly correct — the nn.Module refactor (Patch 9) and LoftQ SVD init (Patch 2) are implemented. However, there are 2 issues that must be fixed before merge:

### Issue 1: `capture_original_weights_from_checkpoint` device bug

**File:** `scripts/qwen_model.py`, function `capture_original_weights_from_checkpoint`

**Problem:** The function has a `device` parameter but the caller in `build_student_super_block` may not pass it correctly. The captured weights must be on CUDA (`device="cuda"`) for the SVD computation in `QwenLoRA.__init__`. Currently the function signature is:
```python
def capture_original_weights_from_checkpoint(sb_idx, model_name="Qwen/Qwen3.5-4B", device="cuda"):
```
But verify the call site in `train_qwen.py` actually passes `device` or relies on the default. If `DEVICE` constant is used, ensure it's `"cuda"`.

**Fix:** Verify the call site. If `device` is not passed, add `device=DEVICE` explicitly.

### Issue 2: PROGRESS.md was overwritten instead of appended

**File:** `agent-ctx/PROGRESS.md`

**Problem:** Each agent has their own divergent copy of PROGRESS.md (4 different versions across 4 branches). The file should be APPEND-ONLY — agents add entries at the bottom, never edit existing content.

**Fix:** Reset PROGRESS.md to the main version, then append your status at the bottom:
```bash
git checkout origin/main -- agent-ctx/PROGRESS.md
# Then append your wave completion entries at the bottom
```

## Tasks

### Fix 1: Verify LoftQ device parameter
- Read `scripts/train_qwen.py` on your branch, find the call to `capture_original_weights_from_checkpoint`
- Verify `device` is passed correctly (should be `DEVICE` which is `"cuda"`)
- If missing, add `device=DEVICE` to the call
- Commit: `Fix 1: verify LoftQ capture device parameter`

### Fix 2: Reset PROGRESS.md to main + append
- `git checkout origin/main -- agent-ctx/PROGRESS.md`
- Append your status entries (Wave 1/2/3 complete, patches 9+2 done)
- Commit: `Fix 2: reset PROGRESS.md to main + append nn-module status`

### Fix 3: Rebase on latest main
- `git pull origin main` (incorporate any orchestrator changes)
- Resolve conflicts in `train_qwen.py` if any (your lines: 632-736)
- Commit: `Fix 3: rebase on latest main`

## DoD
- [ ] LoftQ device parameter verified
- [ ] PROGRESS.md reset to main + appended
- [ ] Branch rebase on main, no conflicts
- [ ] syntax check passes on qwen_model.py + train_qwen.py
- [ ] Branch pushed
