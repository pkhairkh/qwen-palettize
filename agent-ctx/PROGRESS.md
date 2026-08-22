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
| optimizer-streams | 2026-08-22T11:39Z | kernels | P layout changed to (K,N,4) AoS | coordinate |
