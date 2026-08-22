"""Smoke test for re_quantize — verifies the function correctly re-quantizes
a mock PalettizedLinear without errors.

This is a minimal correctness check (not a full unit test suite). It:
  1. Builds a small mock PalettizedLinear with known palette + indices.
  2. Calls re_quantize_indices on a wrapper module containing it.
  3. Verifies the palette, indices, indices_int8, _flat_idx, and
     index_logits are all updated consistently.

Run:
    cd /home/z/my-project/workspace/qwen-palettize
    python3 -c "import sys; sys.path.insert(0,'scripts'); import test_re_quantize_smoke"
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn

from qwen_model import PalettizedLinear
from re_quantize import re_quantize_indices, LOGIT_GAP


def _build_mock_plinear(K=64, N=256, GS=64, PS=4, use_soft=True, device="cpu"):
    """Build a mock PalettizedLinear with known palette + indices."""
    # Original linear (just used for in_features/out_features/bias)
    orig = nn.Linear(K, N, bias=True, device=device)
    # Random initial indices in [0, PS)
    indices = torch.randint(0, PS, (K, N), device=device)
    G = N // GS
    # Random initial palette (G, PS) bf16
    palette_init = torch.randn(G, PS, device=device) * 0.1
    mod = PalettizedLinear(
        orig, indices, G, PS, GS,
        pre_transposed=False,
        initial_palette=palette_init,
        use_soft_indices=use_soft,
    )
    return mod


def _verify_module_consistency(mod, name=""):
    """Verify that indices, indices_int8, _flat_idx, index_logits are
    consistent with each other after re-quantization."""
    K, N = mod.indices.shape
    GS = mod.group_size
    PS = mod.palette_size
    G = mod.n_groups

    # 1. indices vs indices_int8 — must match (modulo dtype)
    mismatch_int = (mod.indices.long() != mod.indices_int8.long()).sum().item()
    assert mismatch_int == 0, f"{name}: indices != indices_int8 ({mismatch_int} mismatches)"

    # 2. _flat_idx must equal g(o)*PS + indices[j, o]
    group_idx = torch.arange(N, device=mod.indices.device) // GS
    group_per_col = group_idx.unsqueeze(0).expand(K, N)
    expected_flat = (group_per_col * PS + mod.indices).contiguous()
    mismatch_flat = (mod._flat_idx != expected_flat).sum().item()
    assert mismatch_flat == 0, f"{name}: _flat_idx mismatch ({mismatch_flat} entries)"

    # 3. If index_logits exists, its argmax must match mod.indices
    if mod.index_logits is not None:
        argmax_idx = mod.index_logits.argmax(dim=0).long()
        mismatch_argmax = (argmax_idx != mod.indices.long()).sum().item()
        assert mismatch_argmax == 0, (
            f"{name}: argmax(index_logits) != indices ({mismatch_argmax} mismatches)"
        )

        # 4. Verify ±3 one-hot init: max logit = +3 for winner, -3 for others
        max_logits = mod.index_logits.max(dim=0).values  # (K, N)
        min_logits = mod.index_logits.min(dim=0).values  # (K, N)
        assert torch.allclose(max_logits, torch.full_like(max_logits, LOGIT_GAP)), (
            f"{name}: max(index_logits) != +{LOGIT_GAP}"
        )
        assert torch.allclose(min_logits, torch.full_like(min_logits, -LOGIT_GAP)), (
            f"{name}: min(index_logits) != -{LOGIT_GAP}"
        )


def test_basic_re_quantize():
    """Smoke test: re_quantize a single mock PalettizedLinear."""
    mod = _build_mock_plinear(use_soft=True)
    wrapper = nn.ModuleList([mod])

    # Snapshot original state
    orig_indices = mod.indices.clone()
    orig_palette = mod.palette.clone()

    # Run re-quantization
    n_changed, n_total = re_quantize_indices(wrapper, sb_idx=0, verbose=True)

    # Verify return values
    assert isinstance(n_changed, int), f"n_changed must be int, got {type(n_changed)}"
    assert isinstance(n_total, int), f"n_total must be int, got {type(n_total)}"
    assert n_total == mod.indices.numel(), (
        f"n_total {n_total} != indices.numel {mod.indices.numel()}"
    )
    print(f"  PASS: n_changed={n_changed}, n_total={n_total}")

    # Verify consistency
    _verify_module_consistency(mod, name="mock_plinear")
    print(f"  PASS: module state consistent (indices, indices_int8, _flat_idx, index_logits)")

    # Verify palette was actually updated (k-means centers should differ from
    # original unless the original was already k-means optimal — extremely
    # unlikely with random init)
    palette_diff = (mod.palette.float() - orig_palette.float()).abs().sum().item()
    print(f"  INFO: palette L1 diff after re-quant = {palette_diff:.4f}")

    # Verify indices changed at least somewhat (random init is very unlikely
    # to be k-means optimal)
    indices_changed = (mod.indices.long() != orig_indices.long()).sum().item()
    print(f"  INFO: indices changed = {indices_changed}/{n_total}")
    # Don't assert >0 — k-means could theoretically converge to same assignment
    # (very unlikely with random init, but possible).

    print("test_basic_re_quantize: PASS")


def test_hard_path():
    """Smoke test: re_quantize a hard-path PalettizedLinear (no index_logits)."""
    mod = _build_mock_plinear(use_soft=False)
    wrapper = nn.ModuleList([mod])

    assert mod.index_logits is None, "Hard-path module should have index_logits=None"

    n_changed, n_total = re_quantize_indices(wrapper, sb_idx=0, verbose=True)
    assert n_total == mod.indices.numel()
    print(f"  PASS: hard-path n_changed={n_changed}, n_total={n_total}")

    _verify_module_consistency(mod, name="mock_plinear_hard")
    print(f"  PASS: hard-path module state consistent")

    print("test_hard_path: PASS")


def test_mixed_wrapper():
    """Smoke test: wrapper with multiple PalettizedLinear + non-Palettized modules."""
    mod1 = _build_mock_plinear(K=32, N=128, GS=32, use_soft=True)
    mod2 = _build_mock_plinear(K=64, N=256, GS=64, use_soft=True)
    mod3 = _build_mock_plinear(K=16, N=64, GS=16, use_soft=False)
    # Add a non-PalettizedLinear module to verify the isinstance filter
    plain_linear = nn.Linear(10, 10)
    wrapper = nn.ModuleList([mod1, mod2, mod3, plain_linear])

    n_changed, n_total = re_quantize_indices(wrapper, sb_idx=0, verbose=True)
    expected_total = mod1.indices.numel() + mod2.indices.numel() + mod3.indices.numel()
    assert n_total == expected_total, f"n_total {n_total} != expected {expected_total}"

    _verify_module_consistency(mod1, name="mod1")
    _verify_module_consistency(mod2, name="mod2")
    _verify_module_consistency(mod3, name="mod3")

    print(f"  PASS: mixed wrapper n_changed={n_changed}, n_total={n_total} (3 PalettizedLinear + 1 plain)")
    print("test_mixed_wrapper: PASS")


if __name__ == "__main__":
    print("Running re_quantize smoke tests...")
    print()
    test_basic_re_quantize()
    print()
    test_hard_path()
    print()
    test_mixed_wrapper()
    print()
    print("All smoke tests PASSED.")
