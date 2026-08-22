# FIX AGENT: kernels — Verify Correctness + Fix PROGRESS.md

> **Branch:** `agent/kernels`
> **Issues found:** 2 (untested fused kernel correctness, PROGRESS.md overwrite)

## Context

Your Round 1 work added the AoS P layout kernels (Patch 5) and batched compute_P_W (Patch 7). The code looks structurally correct, but the fused backward kernel was NOT tested for correctness (no GPU available). The Python backward was switched to call the fused kernel, but if the kernel has a bug, training will produce NaN.

### Issue 1: Fused backward kernel untested

**Files:** `scripts/fused_lut_kernel.cu`, `scripts/fused_lut_linear_cuda.py`

**Problem:** You wrote `test_fused_bwd_aos.py` but couldn't run it (no GPU). The fused kernel `fused_lut_linear_soft_bwd_fused_aos` is now called in the backward, but its output has NOT been verified against the Python reference. If the kernel has a subtle indexing bug, gradients will be wrong and training will diverge.

**Fix:** 
1. Review the kernel code carefully — especially the AoS P indexing: `P_aos[(j * N + o) * 4 + k]`
2. Verify the STE forward is updated to use `P_aos` instead of old `P` (4,K,N)
3. Add a fallback: if `SKIP_FUSED_BWD=1` env var is set, use the Python elementwise path instead
4. Verify the `P_aos` allocation in forward matches the kernel's expected layout

### Issue 2: PROGRESS.md overwritten

**Fix:** `git checkout origin/main -- agent-ctx/PROGRESS.md`, then append status.

## Tasks

### Fix 1: Add SKIP_FUSED_BWD fallback env var
**File:** `scripts/fused_lut_linear_cuda.py`, backward section (~line 912)

Add a fallback so if the fused kernel has issues, the Python path can be used:
```python
if os.environ.get("SKIP_FUSED_BWD", "0") == "1":
    # Fallback to Python elementwise (slower but known-correct)
    grad_W = torch.matmul(x.T, grad_y)
    P_kno = P_aos.view(K, N, 4)  # already AoS, just reshape
    # ... old Python path ...
else:
    grad_logits, grad_palette = mod.fused_lut_linear_soft_bwd_fused_aos(
        grad_y, x, P_aos, palette, GS
    )
```

Commit: `Fix 1: add SKIP_FUSED_BWD fallback env var`

### Fix 2: Verify P_aos allocation in forward
- Read the forward section of `fused_lut_linear_cuda.py` on your branch
- Verify `P_aos` is allocated as `(K, N, 4)` not `(4, K, N)`
- Verify the `compute_P_W_aos` kernel is called (not the old SoA kernel)
- Commit: `Fix 2: verify P_aos allocation + compute_P_W_aos call`

### Fix 3: Reset PROGRESS.md
- `git checkout origin/main -- agent-ctx/PROGRESS.md`
- Append your status
- Commit: `Fix 3: reset PROGRESS.md + append status`

### Fix 4: Rebase on main
- `git pull origin main` — should be clean (you own kernel files exclusively)
- Commit: `Fix 4: rebase on main`

## DoD
- [ ] SKIP_FUSED_BWD fallback env var added
- [ ] P_aos allocation verified as (K, N, 4)
- [ ] compute_P_W_aos kernel is called in forward
- [ ] PROGRESS.md reset to main + appended
- [ ] syntax check passes on .cu and .py
- [ ] Branch rebased on main
- [ ] Branch pushed
