# FIX AGENT: training-recipe — REVERT GS128 + Fix Issues

> **Branch:** `agent/training-recipe`
> **Issues found:** 3 (GS128 applied without approval, PROGRESS.md overwrite, τ schedule needs verification)

## Context

Your Round 1 work has a CRITICAL issue: you applied Patch 4 (GROUP_SIZE 256→128) which was NOT approved by the orchestrator. The user explicitly said this was not planned. The other patches (τ schedule, logit clamp) appear correct but need verification.

### Issue 1: CRITICAL — GS128 was applied without approval

**File:** `scripts/palettize_core.py`, line 26

**Problem:** You changed `GROUP_SIZE = 256` to `GROUP_SIZE = 128`. This was listed as Patch 4 in the research, but the user explicitly said: *"they decided to go down to GS128... This was not planned from my side"*. This change makes ALL existing checkpoints incompatible (they were calibrated with GS=256).

**Fix:** REVERT this change completely:
```python
# REVERT TO:
GROUP_SIZE = 256
```

Remove any per-tensor override code if added. Do NOT change group size.

### Issue 2: PROGRESS.md was overwritten instead of appended

**File:** `agent-ctx/PROGRESS.md`

**Problem:** You have your own divergent copy of PROGRESS.md. Should be append-only.

**Fix:** `git checkout origin/main -- agent-ctx/PROGRESS.md`, then append your status.

### Issue 3: Verify τ schedule is correct

**File:** `scripts/train_qwen.py`, τ anneal logic (~line 1031)

**Problem:** Need to verify the polynomial decay is implemented correctly:
- Warmup: 500 steps at τ=tau_init (2.0)
- Decay: quadratic `(1-progress)^2` from tau_init to tau_final (0.5) over 6000 steps
- Hold: tau_final (0.5) after

Check that the formula matches the research in `research-indices-training/04_tau_schedule.md`.

## Tasks

### Fix 1: REVERT GS128 to GS256
- Edit `scripts/palettize_core.py` line 26
- Change `GROUP_SIZE = 128` back to `GROUP_SIZE = 256`
- Remove any comments about GS128 or per-tensor overrides
- Commit: `Fix 1: REVERT GS128→256 (not approved by orchestrator)`

### Fix 2: Reset PROGRESS.md
- `git checkout origin/main -- agent-ctx/PROGRESS.md`
- Append your status (Patch 1 ✅, Patch 3 ✅, Patch 4 REVERTED)
- Commit: `Fix 2: reset PROGRESS.md + append status`

### Fix 3: Verify τ schedule formula
- Read `scripts/train_qwen.py` τ anneal section on your branch
- Verify it matches: warmup(500) + quadratic decay(6000) + hold(0.5)
- If incorrect, fix to match research
- Commit: `Fix 3: verify τ schedule matches research`

### Fix 4: Rebase on latest main
- `git pull origin main`
- Resolve conflicts (your lines: 1031-1040, 1164-1180, 1279-1285)
- Commit: `Fix 4: rebase on latest main`

## DoD
- [x] GROUP_SIZE = 256 (reverted, NOT 128) — `scripts/palettize_core.py:26`, file now byte-identical to `origin/main`
- [x] No per-tensor override code — grep for `per_tensor|per-tensor|override|GS128` in palettize_core.py returns 0 matches (Patch 4b was already skipped in Round 1)
- [x] PROGRESS.md reset to main + appended — reset via `git checkout origin/main -- agent-ctx/PROGRESS.md`, then appended one Event Log row + a "Round 2 — Fix Agent Status" section at the bottom (no existing table rows modified)
- [x] τ schedule verified correct — 12 anchor checkpoints + 7 spec invariants all pass; matches `research-indices-training/04_tau_schedule.md` §6/§8 exactly. Also fixed stale function-signature defaults at `train_qwen.py:924` (were `tau_init=1.0, tau_final=0.01, tau_anneal_steps=4000`; now `2.0, 0.5, 6000` matching CLI defaults).
- [x] syntax check passes — `palettize_core.py` and `train_qwen.py` both parse cleanly via `ast.parse`
- [x] Branch rebased on main — `git rebase origin/main` succeeded with no conflicts (origin/main's only new commit `945edf3` added ISSUES.md files in `agent-ctx/`, which don't overlap with my changes)
- [x] Branch pushed — see `git log origin/agent/training-recipe`
