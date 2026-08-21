#!/usr/bin/env python3
"""Direct micro-benchmark: OLD PalettizedLinear.forward vs NEW custom autograd Function.
Goal: figure out if 986ms/iter is real or a measurement artifact.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn

# Reproduce OLD PalettizedLinear.forward logic (eager gather + matmul + autograd)
def old_forward(x, palette, indices, group_size, palette_size, pre_transposed, bias):
    orig_ndim = x.ndim
    if x.ndim == 3:
        B, S, _ = x.shape
        x = x.reshape(-1, x.shape[-1])
    si, so = indices.shape
    gs = group_size
    if not hasattr(old_forward, '_cache') or old_forward._cache is None or old_forward._cache.shape != indices.shape:
        group_idx = torch.arange(so, device=indices.device) // gs
        group_idx_2d = group_idx.unsqueeze(0).expand(si, so)
        flat_idx = group_idx_2d * palette_size + indices
        old_forward._cache = flat_idx
    flat_palette = palette.reshape(-1).to(x.dtype)
    gathered = flat_palette[old_forward._cache]
    if not pre_transposed:
        y = x @ gathered
    else:
        y = x @ gathered.T
    if bias is not None:
        y = y + bias.to(x.dtype)
    if orig_ndim == 3:
        y = y.reshape(B, S, -1)
    return y


def bench(label, fn, n_iter=20, warmup=3):
    for _ in range(warmup):
        y = fn()
        y.sum().backward()
        torch.cuda.synchronize()
    # Zero grads between iters (avoid blowup)
    t0 = time.time()
    for _ in range(n_iter):
        y = fn()
        y.sum().backward()
        torch.cuda.synchronize()
    t1 = time.time()
    print(f"  {label:30s}: {(t1-t0)/n_iter*1000:.2f} ms/iter")


def main():
    if not torch.cuda.is_available():
        print("CUDA required"); sys.exit(1)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    in_dim, out_dim = 2560, 8192
    group_size, palette_size = 256, 4
    n_groups = (out_dim + group_size - 1) // group_size
    batch, seq = 8, 128

    indices = torch.randint(0, palette_size, (in_dim, out_dim), device='cuda')
    palette = nn.Parameter(torch.randn(n_groups, palette_size, dtype=torch.bfloat16, device='cuda') * 0.1)
    x = torch.randn(batch, seq, in_dim, dtype=torch.bfloat16, device='cuda', requires_grad=True)

    print(f"\nConfig: in_dim={in_dim}, out_dim={out_dim}, batch={batch}, seq={seq}")
    print(f"x shape: ({batch*seq}, {in_dim}) bf16")
    print(f"indices shape: {tuple(indices.shape)} dtype={indices.dtype}")
    print(f"palette shape: {tuple(palette.shape)} dtype={palette.dtype}")
    print()

    # OLD: eager gather + matmul + native autograd
    old_forward._cache = None  # reset cache
    bench("OLD eager gather+matmul", lambda: old_forward(x, palette, indices, group_size, palette_size, False, None))

    # NEW: custom autograd Function (PalettizedLinear)
    from qwen_model import PalettizedLinear
    lin = nn.Linear(in_dim, out_dim, bias=False)
    pal_lin = PalettizedLinear(
        lin, indices.cpu(), n_groups, palette_size, group_size,
        pre_transposed=False, initial_palette=palette.detach().cpu(),
    ).cuda()
    # Re-attach the same palette as a Parameter (so we compare apples to apples)
    pal_lin.palette = nn.Parameter(palette.detach().clone())
    bench("NEW custom autograd Function", lambda: pal_lin(x))

    # Baseline: just a regular bf16 matmul (no palettization)
    W_direct = torch.randn(out_dim, in_dim, dtype=torch.bfloat16, device='cuda', requires_grad=True)
    bench("BASELINE plain bf16 matmul", lambda: x @ W_direct.T)

    # Baseline: gather alone (no matmul) — to see if the gather is the bottleneck
    flat_palette = palette.reshape(-1)
    group_idx = torch.arange(out_dim, device='cuda') // group_size
    group_idx_2d = group_idx.unsqueeze(0).expand(in_dim, out_dim)
    flat_idx = (group_idx_2d * palette_size + indices).contiguous()
    bench("BASELINE gather only (no matmul)", lambda: flat_palette[flat_idx])

    # Try int32 indices to see if it speeds up gather
    flat_idx_int32 = flat_idx.to(torch.int32)
    bench("BASELINE gather (int32 indices)", lambda: flat_palette[flat_idx_int32])

    # Try torch.index_select (might have a faster kernel than fancy indexing)
    flat_idx_1d = flat_idx.reshape(-1)
    bench("BASELINE torch.index_select", lambda: torch.index_select(flat_palette, 0, flat_idx_1d).reshape(in_dim, out_dim))

    # Try using the same dtype as palette (no .to(x.dtype) cast)
    bench("BASELINE gather no dtype cast", lambda: palette.reshape(-1)[flat_idx])


if __name__ == "__main__":
    main()
