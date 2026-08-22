"""LUT-Q re-quantization — Patch 25 (quality-recipe).

Re-runs k-means on the current W_recon per group at training steps 2000 and
4000 to escape the k-means local optimum that gradient descent on the palette
alone cannot escape. The mechanism is the LUT-Q pattern (Cardinaux et al.
2018, arXiv:1811.05355): periodically refresh the discrete codebook by
re-running k-means on the current (gradient-updated) weight reconstruction,
then re-initialize the trainable index_logits as a one-hot from the new
indices. This breaks out of the local optimum the gradient descent is stuck
in, while re-quantization at carefully chosen step boundaries (rather than
every N steps) avoids the index oscillation pathology documented by Nagel
et al. 2022 (arXiv:2203.11086).

After re-quantization, per PalettizedLinear module:
  - palette.data       <- new k-means centers (cast back to original dtype,
                          typically bf16)
  - indices (buffer)   <- new k-means assignments (int64)
  - indices_int8 (buf) <- same as int8 (for the hard CUDA / Triton kernel)
  - _flat_idx (buffer) <- rebuilt from new indices (fallback path cache:
                          _flat_idx[j, o] = g(o) * palette_size + idx[j, o])
  - index_logits (Param) <- re-initialized as +/-3 one-hot from the new
                          indices (NOT +/-10 — better gradient flow per
                          research-indices-training/01_gumbel_softmax_audit.md
                          Finding 12: at +/-3 gap of 6 with tau=2.0,
                          P_winner ~ 0.87, P_loser ~ 0.043, grad_logits ~300x
                          larger than the +/-10 case where softmax saturates
                          to one-hot and the gradient is ~3e-8, below the
                          AdamW eps=1e-8 floor). Only re-initialized when
                          mod.index_logits exists (soft path).

Research: research-palettes-training/06_staged_training.md Schedule C.
Papers: docs/papers/1811.05355_LUTQ_Cardinaux2018.pdf (LUT-Q pattern),
        docs/papers/2203.11086_QAT_Oscillations_Nagel2022.pdf (oscillation
        prevention via periodic re-quantization).

Usage:
    from re_quantize import re_quantize_indices
    # In the training loop, after step 2000 and 4000:
    if global_step in (2000, 4000) and use_soft_indices:
        print(f"  [step {global_step}] LUT-Q re-quantization...", flush=True)
        n_changed, n_total = re_quantize_indices(student, sb_idx, verbose=True)

Expected behavior (from research-palettes-training/06_staged_training.md §5.5):
  - Step 2000: 5-15% of indices change (palette has shifted enough that the
    k-means assignment updates).
  - Step 4000: <5% of indices change (palette has largely converged; if
    <1% change, re-quantization has converged and future calls can be skipped
    — the caller can check n_changed / n_total < 0.01 to detect this).
"""
import torch


# Logit gap for the +/-3 one-hot re-initialization (Finding 12 sweet spot).
# NOT +/-10 — that saturates softmax to one-hot at any tau > 0.01 and zeros
# all loser gradients (the documented cause of index freeze in the prior
# round, see research-indices-training/01_gumbel_softmax_audit.md Finding 10).
LOGIT_GAP = 3.0


# ═════════════════════════════════════════════════════════════════════════════
# Local 1D weighted k-means (vendored from palettize_pytorch.kmeans1d_weighted
# with the argmin-axis bug fixed).
# ═════════════════════════════════════════════════════════════════════════════
# WHY VENDOR: palettize_pytorch.kmeans1d_weighted (line 63) has a bug:
#     dists = torch.cdist(values.unsqueeze(1), centers.unsqueeze(1)).squeeze(1)
#     assignments = torch.argmin(dists, dim=0)   # ← BUG: dim=0 returns (k,)
#                                               #   should be dim=1 → (N,)
# `cdist((N,1), (k,1))` returns shape (N, k); `squeeze(1)` is a no-op when
# dim 1 has size k>1; `argmin(dists, dim=0)` returns shape (k,) — the index
# of the value closest to each center — instead of the desired (N,) shape
# (the nearest center for each value). The downstream `weights[mask]` then
# raises IndexError because the mask has shape (k,) but weights has shape (N,).
#
# This bug is pre-existing (introduced in commit 5446edf alongside the file
# creation) and affects both calibration (palettize_core.py → palettize_groups
# → kmeans1d_weighted) and this module. Calibration appears to have been run
# before the buggy version was committed (the calib_sb0.log shows successful
# cos=0.92-0.96 per tensor, but a fresh CPU call to kmeans1d_weighted crashes
# immediately).
#
# palettize_pytorch.py is not in any agent's exclusive-ownership list per
# agent-ctx/ROADMAP.md, and no other agent has modified it (verified via
# `git log --all -- scripts/palettize_pytorch.py` — only commit 5446edf
# touches it across all branches). To stay strictly within my owned files
# (NEW scripts/re_quantize.py per RULES.md) and avoid potential merge
# conflicts, I vendor a corrected copy here. The orchestrator should apply
# the same one-line fix (dim=0 -> dim=1) to palettize_pytorch.py separately
# so calibration also works on re-run.
# ═════════════════════════════════════════════════════════════════════════════
def _kmeans1d_weighted_local(values, weights, k, max_iters=100):
    """1D weighted k-means clustering — corrected copy.

    Args:
        values: (N,) tensor — values to cluster
        weights: (N,) tensor — per-value weights (importance)
        k: int — number of clusters (2^bitwidth)
        max_iters: int — max iterations

    Returns:
        centers: (k,) tensor — cluster centers (sorted ascending)
        assignments: (N,) int64 tensor — cluster index per value
    """
    values = values.float().flatten()
    weights = weights.float().flatten()
    n = values.shape[0]

    if n == 0:
        return torch.zeros(k), torch.zeros(0, dtype=torch.long)

    # Initialize centers as quantiles
    sorted_vals, sort_idx = torch.sort(values)
    sorted_w = weights[sort_idx]
    cumw = torch.cumsum(sorted_w, dim=0)
    total_w = cumw[-1]
    quantiles = torch.linspace(0, 1, k + 2)[1:-1]  # k quantile points
    target_cumw = quantiles * total_w
    init_centers = torch.zeros(k)
    for i, tc in enumerate(target_cumw):
        idx = torch.searchsorted(cumw, tc)
        idx = min(idx, n - 1)
        init_centers[i] = sorted_vals[idx]

    centers = init_centers.clone()

    for _ in range(max_iters):
        # Assign each value to nearest center.
        # cdist((N,1), (k,1)) returns (N, k) — distance from each value to
        # each center. argmin along dim=1 (the k axis) gives the nearest
        # center index for each value — shape (N,).
        dists = torch.cdist(values.unsqueeze(1), centers.unsqueeze(1))
        # dists shape: (N, k) — DO NOT squeeze(1); it's a no-op when k>1
        # and the misleading "  # (N,)" comment in the original code is wrong.
        assignments = torch.argmin(dists, dim=1)  # (N,) — FIXED: was dim=0

        # Update centers as weighted mean
        new_centers = centers.clone()
        for c in range(k):
            mask = assignments == c
            if mask.any():
                w = weights[mask]
                v = values[mask]
                new_centers[c] = (v * w).sum() / w.sum().clamp(min=1e-12)

        # Check convergence
        if torch.allclose(new_centers, centers, atol=1e-7):
            break
        centers = new_centers

    # Sort centers ascending
    sorted_centers, sort_order = torch.sort(centers)
    # Remap assignments to sorted order
    remap = torch.zeros(k, dtype=torch.long)
    for i, s in enumerate(sort_order):
        remap[s] = i
    assignments = remap[assignments]

    return sorted_centers, assignments


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════


def re_quantize_indices(model, sb_idx, verbose=True):
    """Re-run k-means on the current W_recon to update indices + palette.

    Walks all PalettizedLinear modules in `model`, reconstructs W_recon from
    the current palette + effective indices (argmax(index_logits) in the soft
    path, or the indices buffer in the hard path), re-runs 1D weighted k-means
    per group with uniform weights, and writes the new palette + indices +
    index_logits back in-place.

    Args:
        model: the student PartialModel / PartialWrapper (any nn.Module whose
               named_modules() yields PalettizedLinear instances).
        sb_idx: super-block index. Currently unused inside the function (the
                re-quantization is purely local to each PalettizedLinear), but
                kept in the signature for API compatibility with the call site
                documented in research-palettes-training/06_staged_training.md
                and to allow future extensions (e.g., per-super-block Hessian
                loading).
        verbose: if True, print per-module change counts and a final summary.

    Returns:
        (n_changed_total, n_total): tuple of ints.
            n_changed_total: total number of indices that changed across all
                             PalettizedLinear modules.
            n_total:         total number of indices across all modules
                             (denominator for the change-rate percentage).
    """
    # Local import to avoid a hard dependency at module load time — re_quantize
    # is only called at step 2000 + 4000, so deferring the import keeps startup
    # fast and avoids circular imports if qwen_model ever imports re_quantize.
    from qwen_model import PalettizedLinear

    n_changed_total = 0
    n_total = 0
    n_modules = 0

    for name, mod in model.named_modules():
        if not isinstance(mod, PalettizedLinear):
            continue
        if getattr(mod, "pre_transposed", False):
            # Pre-transposed modules use a different storage layout (the
            # "indices" axis is the input axis, not the output axis). None of
            # the PalettizedLinear instances in the current super-block setup
            # use pre_transposed=True, but we skip defensively to avoid silent
            # corruption if that ever changes.
            if verbose:
                print(f"    [{name[:60]:<60s}] SKIP — pre_transposed", flush=True)
            continue

        n_modules += 1
        n_changed, n_elems = _re_quantize_one_module(
            mod, name, _kmeans1d_weighted_local, verbose=verbose
        )
        n_changed_total += n_changed
        n_total += n_elems

    if verbose:
        pct = 100.0 * n_changed_total / max(n_total, 1)
        print(
            f"  [re-quant] {n_modules} modules — {n_changed_total:,}/{n_total:,} "
            f"indices changed ({pct:.2f}%)",
            flush=True,
        )
    return n_changed_total, n_total


def _re_quantize_one_module(mod, name, kmeans_fn, verbose=True):
    """Re-quantize a single PalettizedLinear module in-place.

    See module docstring for the full algorithm. This helper is split out so
    that the outer loop in `re_quantize_indices` stays compact and the per-
    module logic can be unit-tested in isolation (test_re_quantize.py, to be
    added in a follow-up).

    Args:
        mod: the PalettizedLinear module to re-quantize.
        name: the module's qualified name (for verbose logging).
        kmeans_fn: the 1D weighted k-means callable. We use the local
                   `_kmeans1d_weighted_local` (vendored, bug-fixed copy —
                   see the comment block above its definition) rather than
                   `palettize_pytorch.kmeans1d_weighted` which has a
                   dim-axis bug (argmin(dim=0) should be dim=1) that causes
                   an IndexError on the downstream `weights[mask]`.
        verbose: if True, print per-module change count.

    Returns:
        (n_changed, n_elems): tuple of ints.
    """
    with torch.no_grad():
        # -- 1. Determine effective indices --------------------------------
        # In the soft path, index_logits is the trainable parameter and its
        # argmax represents the current "best guess" hard indices. In the hard
        # path, index_logits is None and the indices buffer is the source of
        # truth. Either way, we re-derive W_recon from palette + effective
        # indices so the k-means sees the CURRENT reconstructed weight, not
        # the original (pre-training) weight.
        if mod.index_logits is not None:
            # Soft path: argmax over the 4 planes -> (K, N) int64
            eff_indices = mod.index_logits.argmax(dim=0).long()
        else:
            # Hard path: indices buffer is the source of truth
            eff_indices = mod.indices.long()

        K, N = eff_indices.shape
        GS = mod.group_size
        G = mod.n_groups
        PS = mod.palette_size  # typically 4 (2-bit)
        assert N // GS == G, (
            f"[re_quantize] {name}: G mismatch — N//GS={N // GS} vs G={G}"
        )
        assert PS == 4, (
            f"[re_quantize] {name}: Patch 25 expects palette_size=4 (2-bit), "
            f"got PS={PS}"
        )

        # -- 2. Reconstruct W_recon from palette + effective indices --------
        # W_recon[j, o] = palette[g(o), eff_indices[j, o]]
        # g(o) = o // GS
        palette_fp32 = mod.palette.float()  # (G, 4)
        group_idx = torch.arange(N, device=mod.palette.device) // GS  # (N,)
        group_per_col = group_idx.unsqueeze(0).expand(K, N)  # (K, N)
        # Advanced indexing: palette_fp32[group_per_col, eff_indices] gathers
        # palette[g(o), idx[j, o]] for each (j, o) -> shape (K, N) fp32.
        W_recon = palette_fp32[group_per_col.long(), eff_indices]

        # -- 3. Run k-means per group on W_recon ---------------------------
        # Each group covers `GS` output columns and ALL K input rows. The
        # k-means clusters all K * GS values in the group into PS=4 centers.
        # Uniform weights are used (Hessian not available at training time;
        # could be extended to use mod.h_diag buffer if present — SqueezeLLM
        # pattern, see research-indices-training/07_recommendations.md Fix 5).
        new_indices = torch.zeros_like(eff_indices)
        new_palette = torch.zeros_like(palette_fp32)
        for g in range(G):
            start = g * GS
            end = start + GS
            # Extract group g: W_recon[:, start:end] of shape (K, GS)
            W_group = W_recon[:, start:end]
            # kmeans1d_weighted expects 1D input — flatten (K * GS,)
            W_flat = W_group.reshape(-1)
            # Uniform weights
            w_flat = torch.ones_like(W_flat)
            centers, assignments = kmeans_fn(W_flat, w_flat, k=PS)
            # Reshape assignments back to (K, GS) and store
            new_indices[:, start:end] = assignments.view(K, GS)
            new_palette[g] = centers

        # -- 4. Count changes ----------------------------------------------
        changed_mask = (new_indices != eff_indices)
        n_changed = int(changed_mask.sum().item())
        n_elems = int(eff_indices.numel())

        # -- 5. Update module state in-place -------------------------------
        # 5a. Palette (cast back to original dtype, typically bf16)
        mod.palette.data.copy_(new_palette.to(mod.palette.dtype))

        # 5b. Indices buffers (int64 for fallback + int8 for kernels)
        new_indices_int64 = new_indices.long()
        mod.indices.copy_(new_indices_int64)
        mod.indices_int8.copy_(new_indices.to(torch.int8).contiguous())

        # 5c. Rebuild _flat_idx cache (used by the fallback path in
        # PalettizedLinear.forward). _flat_idx[j, o] = g(o) * PS + idx[j, o].
        si, so = new_indices_int64.shape
        device = new_indices_int64.device
        group_idx2 = torch.arange(so, device=device) // GS
        group_idx_2d = group_idx2.unsqueeze(0).expand(si, so)
        new_flat_idx = (group_idx_2d * PS + new_indices_int64).contiguous()
        mod._flat_idx.copy_(new_flat_idx)

        # 5d. Re-init index_logits as +/-3 one-hot from the new indices
        # (only in soft path). +/-3 gap of 6 gives much better gradient flow
        # than the original +/-10 (Finding 12).
        if mod.index_logits is not None:
            new_logits = torch.full(
                (PS, K, N),
                -LOGIT_GAP,
                dtype=mod.index_logits.dtype,
                device=mod.index_logits.device,
            )
            for k in range(PS):
                mask = (new_indices_int64 == k)
                # In-place write to the pre-allocated tensor
                new_logits[k][mask] = LOGIT_GAP
            mod.index_logits.data.copy_(new_logits)

        if verbose:
            pct = 100.0 * n_changed / max(n_elems, 1)
            print(
                f"    [{name[:60]:<60s}] {n_changed:>10,}/{n_elems:,} "
                f"indices changed ({pct:.2f}%)",
                flush=True,
            )

    return n_changed, n_elems
