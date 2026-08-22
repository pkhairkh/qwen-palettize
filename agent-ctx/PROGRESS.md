# Global Progress Tracker

> **Updated by:** Orchestrator + each agent (append-only, never edit existing entries)

---

## Agent Status

| Agent | Branch | Wave 1 | Wave 2 | Wave 3 | Merged |
|-------|--------|--------|--------|--------|--------|
| nn-module-foundation | `agent/nn-module-foundation` | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ |
| training-recipe | `agent/training-recipe` | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ |
| kernels | `agent/kernels` | ✅ Done | ✅ Done | ✅ Done | ⬜ |
| optimizer-streams | `agent/optimizer-streams` | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ |

**Legend:** ⬜ Pending | 🔄 In Progress | ✅ Done | ❌ Blocked

---

## kernels — Patch Status (Round-1 fixes applied)

kernels: Patch 5 ✅, Patch 7 ✅ (fused bwd AoS + batched compute_P_W)

- **Patch 5** (fused bwd AoS P layout): All 4 sub-tasks done (5a compute_P_W_aos kernel, 5b fused_bwd_aos kernel, 5c Python wiring, 5d correctness test). Round-1 fix added a `SKIP_FUSED_BWD=1` env-var fallback to the Python elementwise path (Issue 1) so operators can switch back to the known-correct path if the fused kernel produces NaN on a real GPU.
- **Patch 7** (batched compute_P_W, 25→1): All 3 sub-tasks done (7a PalettizedLayerDesc struct, 7b batched kernel, 7c Python wrapper). Single kernel launch with blockIdx.z = layer_idx replaces 25 per-layer launches.
- **Round-1 verification (Issue 2):** P_aos allocation confirmed as (K, N, 4) fp16 AoS in `fused_lut_linear_soft_fwd_aos` (C++ wrapper line 410). The `fused_lut_linear_soft_compute_P_W_aos_Launcher` is invoked (NOT the legacy SoA launcher). STE forward uses `logits.argmax(dim=0)` — works because logits remain (4, K, N) SoA; only P's storage moved to AoS. `ctx.save_for_backward(..., P_aos, W)` saves the AoS tensor. These invariants are now enforced by `test_paos_allocated_as_kn4_in_cpp_wrapper`, `test_forward_calls_compute_P_W_aos_launcher`, and `test_ste_forward_uses_logits_argmax` in `scripts/test_fused_bwd_aos.py:TestModuleSymbols`.
- **Round-1 PROGRESS.md fix (Issue 3):** File was overwritten in Round 1 instead of appended; reset to `origin/main` and re-appended the kernels status (this section).
- **Round-1 rebase (Issue 4):** Branch rebased on `origin/main` (commit 945edf3 — added agent-ctx/agent-kernels/ISSUES.md). Clean — no conflicts (kernels owns `fused_lut_kernel.cu` and `fused_lut_linear_cuda.py` exclusively).

---

## Merge Order

1. ⬜ nn-module-foundation (foundation — must merge first)
2. ⬜ training-recipe (rebases on nn-module)
3. ✅ kernels (independent — can merge anytime after Wave 1) — **READY**
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
| 5 | Fused bwd with AoS P layout | kernels | ✅ | `agent/kernels` | 73558af (5a), fdf5283 (5b), 554348f (5c), 164bd44 (5d) |
| 7 | Batched compute_P_W (25→1) | kernels | ✅ | `agent/kernels` | b26b209 (7a+7b), 9307d79 (7c) |
| 8 | Fused AdamW (bitsandbytes 8-bit) | optimizer-streams | ⬜ | — | — |
| 6 | Stream double-buffering | optimizer-streams | ⬜ | — | — |

---

## Event Log (append-only)

| Timestamp | Agent | Event |
|-----------|-------|-------|
| 2025-08-22T12:00:00Z | orchestrator | Created agent-ctx infrastructure + 4 branches |
| 2026-08-22T10:30:00Z | training-recipe | Round 2 (fix agent): GS128 reverted to GS256 in palettize_core.py (commit `19597e5`). Patch 1 ✅ kept (τ schedule verified correct). Patch 3 ✅ kept (adaptive ±5τ clamp). Patch 4 REVERTED (GS128 was NOT approved by orchestrator). PROGRESS.md reset to origin/main (prior agent had overwritten the Agent Status / Patch Status tables instead of appending). Branch rebased on latest main + pushed. |
| 2026-08-22T11:39:00Z | kernels | Wave 1 complete: Patch 5 (a–d) — AoS P layout + fused bwd kernel re-enabled |
| 2026-08-22T11:55:00Z | kernels | Wave 2 complete: Patch 7 (a–c) — batched compute_P_W kernel (25 launches → 1) |
| 2026-08-22T12:15:00Z | kernels | Wave 3 complete: Patch 8a profile test + 8b merge prep verified clean. Branch ready for merge. |
| 2026-08-22T13:00:00Z | kernels | Round-1 fix wave: SKIP_FUSED_BWD fallback env var added (Issue 1), P_aos allocation + compute_P_W_aos call verified via new TestModuleSymbols assertions (Issue 2), PROGRESS.md reset to main + re-appended (Issue 3), branch rebased on main (Issue 4). |

---

## Inbox Summary (last message per agent)

| Agent | Last Message | From | Subject | Action Required |
|-------|--------------|------|---------|-----------------|
| nn-module-foundation | 2026-08-22T11:55Z | kernels | Batched compute_P_W available | coordinate |
| training-recipe | 2026-08-22T11:39Z | kernels | P layout changed to (K,N,4) AoS | coordinate |
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
| optimizer-streams | 2026-08-22T11:39Z | kernels | P layout changed to (K,N,4) AoS | coordinate |
