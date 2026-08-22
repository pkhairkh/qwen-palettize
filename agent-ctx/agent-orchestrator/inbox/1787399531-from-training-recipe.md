# Message: training-recipe ready for merge

**TO:** orchestrator
**FROM:** training-recipe
**TIMESTAMP:** 2026-08-22T09:30:00Z
**SUBJECT:** training-recipe branch ready for merge — all 3 patches (1, 3, 4) complete

## Summary

All 3 assigned patches are implemented, syntax-checked, math/scope-verified,
and pushed to `origin/agent/training-recipe`. The branch is ready for the
merge step.

| Patch | Description | Commit | File(s) |
|-------|-------------|--------|---------|
| 1 (1a + 1b) | Polynomial τ schedule (warmup + quadratic decay + floor 0.5) | `957ad62` (CLI defaults) + `012e820` (anneal logic) | `scripts/train_qwen.py` (argparse ~1241-1246, anneal ~1034-1062) |
| 3 | Adaptive logit clamp ±5τ (was fixed ±20) | `c0a80e4` | `scripts/train_qwen.py` (~1175→1192) |
| 4a | GROUP_SIZE 256→128 | `d72ba5e` | `scripts/palettize_core.py:26` |

Branch HEAD: `d72ba5e` → 6 commits ahead of `origin/main`.

## Definition of Done — all 3 waves verified

### Wave 1 (Patch 1 — τ schedule) ✅
- [x] τ defaults: tau_init=2.0, tau_final=0.5, tau_anneal_steps=6000
- [x] Piecewise: 500-step warmup at τ=2.0, quadratic (α=2) decay to 0.5, hold
- [x] `python3 -c "import ast; ast.parse(open('scripts/train_qwen.py').read())"` passes
- [x] Branch pushed
- [x] Inbox message sent to nn-module-foundation (informational, no conflicts)
- [x] PROGRESS.md updated (Patch 1 ✅)
- [x] Commit + push
- [x] Bonus: math verified with `/home/z/my-project/scripts/verify_tau_schedule.py` — all 10 sanity cases pass

### Wave 2 (Patch 3 — Logit clamp) ✅
- [x] Clamp is `±5*tau` (adaptive — was fixed ±20)
- [x] syntax check passes
- [x] Branch pushed (no rebase needed — `origin/main` unchanged, nn-module-foundation has not merged yet)
- [x] PROGRESS.md updated (Patch 3 ✅)
- [x] Bonus: tau-in-scope verified with `/home/z/my-project/scripts/check_tau_in_scope.py` — AST walk confirms `opt_indices` is non-None only when `use_soft_indices=True` (per `build_optimizers` line 597), and when `use_soft_indices=True`, the τ-assignment block runs on every branch (warmup/decay/hold) earlier in the same for-loop iteration, before the clamp reads `tau`.

### Wave 3 (Patch 4 — Group size) ✅
- [x] GROUP_SIZE=128 in `palettize_core.py:26`
- [x] Patch 4b (per-tensor override) SKIPPED — optional, adds complexity without clear benefit. Orchestrator can apply later if checkpoint preservation becomes important.
- [x] syntax check passes
- [x] Branch merges cleanly with `origin/main` (`git merge --dry-run` = "Already up to date")
- [x] PROGRESS.md updated (Patch 4 ✅, note about re-calibration)
- [x] Final inbox message to orchestrator (this message)
- [x] Bonus: all 6 downstream consumers verified to correctly see `GROUP_SIZE=128`:
  - `palettize_core.py` (source)
  - `qwen_model.py` (imports from palettize_core)
  - `calib_qwen.py` (imports from palettize_core)
  - `calib_stage2.py` (imports via train_qwen)
  - `sweep_qwen.py` (imports via train_qwen)
  - `convert_trained_to_packed.py` (imports from palettize_core)

## Pre-flight items the orchestrator should be aware of

1. **Re-calibration REQUIRED before resuming training.** Existing trained
   checkpoints (`trained/superblock_0_best/`, `trained/superblock_0_safe_backup/`,
   `palettized/superblock_0/`) were all calibrated with `GROUP_SIZE=256`.
   After Patch 4 (`GROUP_SIZE=128`), these are INCOMPATIBLE — the index
   layout (4 indices per byte, transposed D0D1) is parameterized on
   group count. The orchestrator MUST run `calib_qwen.py --sb_idx 0`
   to re-generate the palettized tensors before `train_qwen.py` can
   resume from any of them.

2. **`fused_lut_linear_cuda.py:425` and `fused_lut_kernel.cu:28` still
   hardcode `GROUP_SIZE=256` / `GROUP_SIZE_DEFAULT 256`.** These are
   owned by the `kernels` agent. The runtime calls to the kernel pass
   `group_size=GROUP_SIZE` (imported from `palettize_core`, now 128), so
   the kernel will use 128 at runtime — but the local constant in
   `fused_lut_linear_cuda.py` should be updated by the kernels agent
   for consistency. The orchestrator should coordinate with `kernels`
   to ensure this is handled in their wave.

3. **`verify_palettize_core.py` has a pre-existing import error** (tries
   to import `pack_indices_transposed`, which was renamed to
   `pack_indices_transposed_2bit`). This was broken before any of my
   changes — not caused by Patch 4. Recommend the orchestrator schedule
   a small cleanup task to update this test file to use the renamed
   function (and replace hardcoded `assert meta["group_size"] == 256`
   at line 72 with `assert meta["group_size"] == GROUP_SIZE`).

4. **`train_qwen.py:924` function-signature defaults** are still
   `tau_init=1.0, tau_final=0.01, tau_anneal_steps=4000`. The CLI
   argparse defaults (lines 1241-1246) now correctly set
   `tau_init=2.0, tau_final=0.5, tau_anneal_steps=6000` and the CLI
   always passes these to `train_super_block(...)`, so the function-
   signature defaults are dead code at runtime. I left them alone
   because line 924 is outside my exclusive territory per RULES.md.
   Recommend the orchestrator schedule a tiny cleanup commit to align
   them (single 1-line change in `train_qwen.py:924`).

5. **No conflicts with other agents' branches.** My exclusive territory
   per ROADMAP.md §2 is `palettize_core.py` (full) + `train_qwen.py`
   (τ anneal logic ~1034-1040, CLI defaults ~1241-1246, logit clamp
   ~1153). None of the other agents own these ranges. The
   `optimizer-streams` branch advanced by 1 commit during my work
   (fetched `d1692c1..1652d84`); their territory is
   `train_qwen.py:540-600` and `train_qwen.py:1058-1103` — no overlap
   with mine.

## Merge order recommendation

Per ROADMAP.md §3:
> Orchestrator merges in order: nn-module → training-recipe → kernels → optimizer-streams

I am step 2. I have NOT seen any "RELEASED" message from
`nn-module-foundation` in my inbox — they have not merged Patch 9 to
`main` yet (verified: `origin/main` HEAD is still at `8770547`, the
pre-agent baseline). The orchestrator should wait for nn-module-foundation
to merge first, then merge my branch second.

That said, since I touched only files/lines in my exclusive territory,
my branch is merge-clean against the current `origin/main` regardless of
whether nn-module-foundation has merged. A rebase later (if
nn-module-foundation's merge conflicts with my comment-block expansions
in train_qwen.py) is straightforward.

## What I need from you

Merge `agent/training-recipe` into `main` (after nn-module-foundation
merges first, per ROADMAP.md §3). Then send a "RELEASED" message to
all agents so they can rebase.

**ACTION REQUIRED:** merge branch (after nn-module-foundation merges first)
