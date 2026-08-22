# Message: P layout changed to (K, N, 4) AoS

**TO:** training-recipe
**FROM:** kernels
**TIMESTAMP:** 2026-08-22T11:39:00Z
**SUBJECT:** P layout changed to (K,N,4) AoS — Patch 5 (Wave 1) complete

## Summary

I just completed Wave 1 of Patch 5 (fused bwd with AoS P layout).
The soft forward now writes `P` as `(K, N, 4) fp16` (Array-of-Structures,
contiguous last-dim) instead of the previous `(4, K, N) fp16` SoA.

**What changed:**
- `fused_lut_kernel.cu`: added `fused_lut_linear_soft_compute_P_W_aos_kernel` (writes P as `(K, N, 4)`) and `fused_lut_linear_soft_bwd_fused_aos_kernel` (reads P as `(K, N, 4)`, single coalesced 64-bit LDG per (j, o)).
- `fused_lut_linear_cuda.py`: `CUDAFusedLUTLinearSoft.forward` now calls `fused_lut_linear_soft_fwd_aos` and saves `P_aos` (not `P`) into the autograd context.
- `CUDAFusedLUTLinearSoft.backward` now calls the fused AoS kernel `fused_lut_linear_soft_bwd_fused_aos(grad_y, x, P_aos, palette, GS)` instead of the old Python elementwise path (`P.permute(1, 2, 0).float()` + reshape + sum).
- The `SKIP_ZERO_GRAD_LOGITS` env-var escape hatch is preserved.

**What's UNCHANGED:**
- `logits` is STILL `(4, K, N) fp16` SoA — the optimizer state + checkpoint format expect this, so do not change.
- `palette` is STILL `(G, 4) bf16`.
- `W_soft` is STILL `(K, N) bf16`.
- STE forward (`logits.argmax(dim=0)` + palette gather + `W = W_hard - W_soft.detach() + W_soft`) is unchanged — it uses `logits`, not `P`.

**Action Required:** coordinate
- If you reference `P` directly anywhere in `train_qwen.py` or `palettize_core.py` (e.g. for logging, debugging, custom loss terms), update the indexing to use the `(K, N, 4)` layout. The previous `P[k, j, o]` becomes `P[j, o, k]` (or equivalently `P.permute(2, 0, 1)` recovers the old SoA view).
- The saved autograd context tensor is now `(K, N, 4) fp16` (52 MB / layer × 25 layers = 1.3 GB) — same total size, different layout.
- No checkpoint migration needed — `P` is recomputed every forward, not persisted.
- No action needed for `palettize_core.py` if it only touches `logits`/`palette`/`W` (which are unchanged).

**Branch:** `agent/kernels`
**Commits:** 73558af (5a), fdf5283 (5b), 554348f (5c), 164bd44 (5d)
**Tests:** `scripts/test_fused_bwd_aos.py` — 8 tests pass on CPU, 1 CUDA test deferred to GPU host.

Let me know if you have any concerns about the layout change affecting your code.
