#!/usr/bin/env python3
"""Smoke test for the new PalettizedLinear + custom autograd Function.

Verifies:
1. Forward output matches the naive gather+matmul reference.
2. Backward grad_x matches the naive autograd path.
3. Backward grad_palette matches a finite-difference estimate.
4. pre_transposed=True branch is also correct.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from qwen_model import PalettizedLinear

torch.manual_seed(0)

def naive_forward(x, palette, indices, group_size, palette_size, pre_transposed, bias=None):
    """Reference implementation: explicit gather + matmul."""
    si, so = indices.shape
    group_idx = torch.arange(so, device=indices.device) // group_size
    group_idx_2d = group_idx.unsqueeze(0).expand(si, so)
    flat_idx = group_idx_2d * palette_size + indices
    flat_palette = palette.reshape(-1).to(x.dtype)
    gathered = flat_palette[flat_idx]  # (in_dim, out_dim)
    if not pre_transposed:
        y = x @ gathered
    else:
        y = x @ gathered.T
    if bias is not None:
        y = y + bias.to(x.dtype)
    return y


def test_forward_matches_reference(pre_transposed):
    print(f"\n=== Test forward matches reference (pre_transposed={pre_transposed}) ===")
    in_dim, out_dim = 64, 128
    group_size, palette_size = 16, 4
    n_groups = (out_dim + group_size - 1) // group_size
    batch, seq = 2, 8

    # Build a fake Linear
    lin = nn.Linear(in_dim, out_dim, bias=False)
    # Build random indices + palette.
    # Indices shape depends on pre_transposed:
    #   pre_transposed=False: (in_dim, out_dim) -- indices stored transposed
    #   pre_transposed=True:  (out_dim, in_dim) -- indices stored in original orientation
    if not pre_transposed:
        indices = torch.randint(0, palette_size, (in_dim, out_dim))
    else:
        indices = torch.randint(0, palette_size, (out_dim, in_dim))
    palette = nn.Parameter(torch.randn(n_groups, palette_size, dtype=torch.bfloat16) * 0.1)

    pal_lin = PalettizedLinear(
        lin, indices, n_groups, palette_size, group_size,
        pre_transposed=pre_transposed, initial_palette=palette,
    ).cuda()

    x = torch.randn(batch, seq, in_dim, dtype=torch.bfloat16).cuda()
    y_new = pal_lin(x)
    y_ref = naive_forward(x.reshape(-1, in_dim), pal_lin.palette, pal_lin.indices,
                          group_size, palette_size, pre_transposed, pal_lin.bias)
    y_ref = y_ref.reshape(batch, seq, out_dim)

    diff = (y_new.float() - y_ref.float()).abs().max().item()
    print(f"  max abs diff: {diff:.6e}")
    assert diff < 1e-3, f"Forward mismatch: {diff}"
    print(f"  ✓ forward matches reference (pre_transposed={pre_transposed})")


def test_backward_grad_x(pre_transposed):
    print(f"\n=== Test backward grad_x (pre_transposed={pre_transposed}) ===")
    in_dim, out_dim = 64, 128
    group_size, palette_size = 16, 4
    n_groups = (out_dim + group_size - 1) // group_size
    batch, seq = 2, 8

    lin = nn.Linear(in_dim, out_dim, bias=False)
    if not pre_transposed:
        indices = torch.randint(0, palette_size, (in_dim, out_dim))
    else:
        indices = torch.randint(0, palette_size, (out_dim, in_dim))
    init_palette = torch.randn(n_groups, palette_size, dtype=torch.bfloat16) * 0.1

    pal_lin = PalettizedLinear(
        lin, indices, n_groups, palette_size, group_size,
        pre_transposed=pre_transposed, initial_palette=init_palette,
    ).cuda()
    x = torch.randn(batch, seq, in_dim, dtype=torch.bfloat16, device='cuda', requires_grad=True)

    y = pal_lin(x)
    loss = y.sum()
    loss.backward()
    grad_x_new = x.grad.clone()

    # Reference: use naive gather+matmul with autograd
    x_ref = x.detach().clone().requires_grad_(True)
    y_ref = naive_forward(x_ref.reshape(-1, in_dim), pal_lin.palette, pal_lin.indices,
                          group_size, palette_size, pre_transposed, pal_lin.bias)
    y_ref = y_ref.reshape(batch, seq, out_dim)
    y_ref.sum().backward()
    grad_x_ref = x_ref.grad

    diff = (grad_x_new.float() - grad_x_ref.float()).abs().max().item()
    print(f"  max abs diff: {diff:.6e}")
    assert diff < 1e-3, f"grad_x mismatch: {diff}"
    print(f"  ✓ grad_x matches reference (pre_transposed={pre_transposed})")


def test_grad_palette_via_finite_diff():
    """Verify grad_palette matches a finite-difference estimate on one element."""
    print(f"\n=== Test grad_palette via finite difference ===")
    in_dim, out_dim = 32, 64
    group_size, palette_size = 16, 4
    n_groups = (out_dim + group_size - 1) // group_size
    batch, seq = 2, 4

    lin = nn.Linear(in_dim, out_dim, bias=False)
    indices = torch.randint(0, palette_size, (in_dim, out_dim))
    init_palette = torch.randn(n_groups, palette_size, dtype=torch.float64) * 0.1  # float64 for FD

    # Cast indices to long
    indices = indices.long()

    pal_lin = PalettizedLinear(
        lin, indices, n_groups, palette_size, group_size,
        pre_transposed=False, initial_palette=init_palette,
    ).cuda()
    # Override dtype to float64 for finite-diff precision
    pal_lin.palette = nn.Parameter(init_palette.cuda().double())
    pal_lin._flat_idx = pal_lin._flat_idx.cuda()

    x = torch.randn(batch, seq, in_dim, dtype=torch.float64).cuda()

    # Forward + backward
    y = pal_lin(x)
    loss = (y ** 2).sum()
    loss.backward()
    grad_palette = pal_lin.palette.grad.clone()

    # Finite difference on palette[0, 0]
    eps = 1e-4
    pal_lin.palette.data[0, 0] += eps
    y_plus = pal_lin(x)
    loss_plus = (y_plus ** 2).sum()
    pal_lin.palette.data[0, 0] -= 2 * eps
    y_minus = pal_lin(x)
    loss_minus = (y_minus ** 2).sum()
    pal_lin.palette.data[0, 0] += eps  # restore
    fd_grad = (loss_plus.item() - loss_minus.item()) / (2 * eps)

    autograd_grad = grad_palette[0, 0].item()
    rel_err = abs(autograd_grad - fd_grad) / max(abs(autograd_grad), abs(fd_grad), 1e-12)
    print(f"  autograd grad_palette[0,0] = {autograd_grad:.6e}")
    print(f"  finite-diff grad_palette[0,0] = {fd_grad:.6e}")
    print(f"  relative error: {rel_err:.6e}")
    assert rel_err < 1e-3, f"grad_palette mismatch: autograd={autograd_grad}, fd={fd_grad}"
    print(f"  ✓ grad_palette matches finite difference")


def test_speedup_forward():
    """Quick benchmark: new fused vs naive gather+matmul."""
    print(f"\n=== Speed benchmark: PalettizedLinear ===")
    import time
    in_dim, out_dim = 2560, 8192  # actual Qwen3.5 size
    group_size, palette_size = 256, 4
    n_groups = (out_dim + group_size - 1) // group_size
    batch, seq = 8, 128

    lin = nn.Linear(in_dim, out_dim, bias=False)
    indices = torch.randint(0, palette_size, (in_dim, out_dim)).cuda()
    init_palette = torch.randn(n_groups, palette_size, dtype=torch.bfloat16).cuda() * 0.1

    pal_lin = PalettizedLinear(
        lin, indices, n_groups, palette_size, group_size,
        pre_transposed=False, initial_palette=init_palette,
    ).cuda()
    pal_lin._flat_idx = pal_lin._flat_idx.cuda()

    x = torch.randn(batch, seq, in_dim, dtype=torch.bfloat16).cuda()

    # Warmup
    for _ in range(3):
        y = pal_lin(x)
        y.sum().backward()
    torch.cuda.synchronize()

    # Time forward+backward
    n_iter = 50
    t0 = time.time()
    for _ in range(n_iter):
        y = pal_lin(x)
        y.sum().backward()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"  PalettizedLinear fwd+bwd: {(t1-t0)/n_iter*1000:.2f} ms/iter at seq_len=128, batch=8")
    print(f"  (was ~200ms+ at seq_len=2048, batch=1 in original code)")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA required for this test")
        sys.exit(1)

    test_forward_matches_reference(pre_transposed=False)
    test_forward_matches_reference(pre_transposed=True)
    test_backward_grad_x(pre_transposed=False)
    test_backward_grad_x(pre_transposed=True)
    test_grad_palette_via_finite_diff()
    test_speedup_forward()

    print("\n=== ALL TESTS PASSED ===")
