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

## nn-module-foundation: Wave 1-3 complete (Patch 9 + Patch 2)

> **Agent:** nn-module-foundation
> **Branch:** `agent/nn-module-foundation`
> **Status:** ✅ All 3 waves complete. Ready for orchestrator merge (must merge first).

### Patches Delivered

| # | Patch | Status | Wave | Commit(s) |
|---|-------|--------|------|-----------|
| 9 | PartialWrapper → nn.Module | ✅ Done | Wave 1 | `1db364f` (9a), `96a0fdb` (9b), `26b9679` (9c) |
| 2 | LoftQ SVD init for LoRA | ✅ Done | Wave 2 | `6c21347` (2a), `e928898` (2b) |

### Wave Summary

- **Wave 1** (Patch 9 — nn.Module refactor): `PartialModel` and `PartialWrapper`
  now inherit from `nn.Module`. `nn.ModuleList` tracks layers. Both classes
  implement `forward(self, input_ids, position_ids=None)`. All hand-rolled
  `to()/parameters()/named_parameters()/eval()/train()/named_modules()/
  get_submodule()` methods deleted — inherited from `nn.Module` now.
  `load_qwen_super_block_only` compatibility verified (`.to(device).eval()`,
  `.train()`, `isinstance(wrapper, nn.Module)` all hold).
- **Wave 2** (Patch 2 — LoftQ SVD init): New helper
  `capture_original_weights_from_checkpoint(sb_idx, model_name, device)`
  in `qwen_model.py`. `build_student_super_block` in `train_qwen.py:696-713`
  calls it and passes the original fp16 weight per-tensor to `QwenLoRA(
  ..., init="loftq", original_weight=orig_w)`. LoftQ branch at
  `qwen_model.py:209-222` runs SVD on `R = W_orig - W_quantized` to warm-start
  `lora_A`/`lora_B`.
- **Wave 3** (merge prep): Branch merges cleanly with main (fast-forward,
  no conflicts). Final RELEASED + merge-ready messages sent to all 3 dependent
  agents (training-recipe, optimizer-streams, kernels).

### Round 1 Fix Wave (current)

Addressed 3 issues from `agent-ctx/agent-nn-module-foundation/ISSUES.md`:

- **Fix 1** — LoftQ device parameter: changed default in
  `capture_original_weights_from_checkpoint` from `device="cpu"` to
  `device="cuda"`, and added explicit `device=DEVICE` at the call site in
  `train_qwen.py` so the SVD inside `QwenLoRA.__init__` runs on-GPU without
  a per-tensor host→device copy. Commit: `Fix 1: verify LoftQ capture
  device parameter`.
- **Fix 2** — Reset PROGRESS.md to origin/main (was diverged/overwritten);
  this section is the appended nn-module status block.
- **Fix 3** — Rebased on latest origin/main; no conflicts in
  `train_qwen.py:632-736` (my owned lines).

### Files Owned (EXCLUSIVE)

- `scripts/qwen_model.py` — `PartialModel`, `PartialWrapper`,
  `capture_original_weights_from_checkpoint`, `load_qwen_super_block_only`
- `scripts/train_qwen.py` — `build_student_super_block` only (lines ~632-736)

### DoD

- [x] PartialModel + PartialWrapper inherit from `nn.Module`, `forward()` added
- [x] All hand-rolled methods deleted
- [x] `capture_original_weights_from_checkpoint()` added; QwenLoRA receives
      `original_weight` for LoftQ SVD
- [x] LoftQ weights on CUDA (`device="cuda"`) so SVD runs on-GPU
- [x] `python3 -c "import ast; ast.parse(open('scripts/qwen_model.py').read())"` passes
- [x] `python3 -c "import ast; ast.parse(open('scripts/train_qwen.py').read())"` passes
- [x] Branch rebased on latest `origin/main`, no conflicts
- [x] Branch pushed to `agent/nn-module-foundation`
- [x] Inbox messages sent to training-recipe, optimizer-streams, kernels
- [x] PROGRESS.md reset to main + appended (this section)
