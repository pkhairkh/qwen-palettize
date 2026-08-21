"""Performance benchmark for the fused LUT-quantized linear layer.

Compares end-to-end wall time across:
  - Reference PyTorch (gather + cuBLAS matmul, scatter_add backward)
  - Triton fused kernel
  - CUDA fused kernel (production)
  - cuBLAS bf16 baseline (for reference — uses MATERIALIZE W, then matmul)

Reports per-kernel fwd/bwd latency, relative speedups, and memory footprint.

Usage
-----
    python benchmark.py
    python benchmark.py --reps 50 --warmup 10
    python benchmark.py --shape 2560,2560 --shape 9216,2560
"""
from __future__ import annotations
import argparse
import sys
import time

import torch
from torch import Tensor


DEFAULT_SHAPES = [
    (2560, 2560),
    (2560, 8192),
    (2560, 9216),
    (9216, 2560),     # largest K — backward bottleneck per spec
]
DEFAULT_M = 1024
DEFAULT_GROUP_SIZE = 256


def _bench(fn, warmup: int = 10, reps: int = 50) -> float:
    """Returns median wall time in ms."""
    torch.cuda.synchronize()
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    # Time
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def bench_one_shape(K: int, N: int, M: int, group_size: int, device: str,
                    warmup: int, reps: int, skip_cuda: bool) -> None:
    G = N // group_size
    print(f"\n=== K={K}, N={N}, M={M}, G={G} ===")
    print(f"  Weight matrix (K×N bf16) = {K*N*2/1024/1024:.1f} MB if materialized")
    print(f"  Indices (K×N int8)      = {K*N/1024/1024:.1f} MB")
    print(f"  Palette (G×4 bf16)       = {G*4*2} bytes")

    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.5
    base = torch.randn(G, 1, dtype=torch.bfloat16, device=device) * 0.2
    offsets = torch.tensor([-0.3, -0.1, 0.1, 0.3], dtype=torch.bfloat16, device=device)
    palette = (base + offsets.expand(G, 4)).detach().requires_grad_(True)
    indices = torch.randint(0, 4, (K, N), dtype=torch.int8, device=device)
    bias = torch.randn(N, dtype=torch.bfloat16, device=device) * 0.1
    grad_y = torch.randn(M, N, dtype=torch.bfloat16, device=device) * 0.1

    # ── Reference PyTorch (gather + cuBLAS + scatter_add) ───────────────────
    from reference_pytorch import lut_linear_forward as ref_fwd, lut_linear_backward as ref_bwd
    t_ref_fwd = _bench(lambda: ref_fwd(x, palette, indices, bias, group_size), warmup, reps)
    t_ref_bwd = _bench(lambda: ref_bwd(grad_y, x, palette, indices, bias, group_size), warmup, reps)
    print(f"  REF FWD: {t_ref_fwd:.3f} ms   BWD: {t_ref_bwd:.3f} ms   TOTAL: {t_ref_fwd+t_ref_bwd:.3f} ms")

    # ── Triton fused ───────────────────────────────────────────────────────
    from triton_lut_linear import triton_lut_linear as tri_fn

    # Forward
    def tri_fwd_bwd():
        y = tri_fn(x, palette, indices, bias, group_size)
        y.backward(grad_y, retain_graph=True)
    t_tri_total = _bench(tri_fwd_bwd, warmup, reps)
    # Separate timing
    t_tri_fwd = _bench(lambda: tri_fn(x, palette, indices, bias, group_size), warmup, reps)
    t_tri_bwd = t_tri_total - t_tri_fwd
    print(f"  TRI FWD: {t_tri_fwd:.3f} ms   BWD: {t_tri_bwd:.3f} ms   TOTAL: {t_tri_total:.3f} ms"
          f"   (speedup vs ref total: {t_ref_fwd+t_ref_bwd:.2f}x → {t_tri_total:.2f}x)")

    # ── CUDA fused ─────────────────────────────────────────────────────────
    if not skip_cuda:
        try:
            from fused_lut_linear_cuda import fused_lut_linear as cuda_fn

            # Warmup compile
            _ = cuda_fn(x, palette, indices, bias, group_size)
            torch.cuda.synchronize()

            def cuda_fwd_bwd():
                y = cuda_fn(x, palette, indices, bias, group_size)
                y.backward(grad_y, retain_graph=True)
            t_cuda_total = _bench(cuda_fwd_bwd, warmup, reps)
            t_cuda_fwd = _bench(lambda: cuda_fn(x, palette, indices, bias, group_size), warmup, reps)
            t_cuda_bwd = t_cuda_total - t_cuda_fwd
            print(f"  CUDA FWD: {t_cuda_fwd:.3f} ms   BWD: {t_cuda_bwd:.3f} ms   TOTAL: {t_cuda_total:.3f} ms"
                  f"   (speedup vs ref total: {(t_ref_fwd+t_ref_bwd)/t_cuda_total:.2f}x,"
                  f"  vs triton: {t_tri_total/t_cuda_total:.2f}x)")
        except Exception as e:
            print(f"  CUDA benchmark failed: {e}")

    # ── cuBLAS bf16 baseline (materializes W via gather — apples-to-apples memory comparison) ──
    # Note: this measures cuBLAS GEMM ONLY, not the gather/scatter overhead.
    with torch.no_grad():
        flat_palette = palette.detach().reshape(-1)
        o_idx = torch.arange(N, device=device, dtype=torch.long)
        g = o_idx // group_size
        flat_idx = g.unsqueeze(0) * 4 + indices.long()
        W_mat = flat_palette[flat_idx]  # (K, N) bf16
    def cublas_fwd():
        return x @ W_mat + bias
    t_cublas_fwd = _bench(cublas_fwd, warmup, reps)
    def cublas_bwd():
        gx = grad_y @ W_mat.T
        dW = x.T.float() @ grad_y.float()
        return gx, dW
    t_cublas_bwd = _bench(cublas_bwd, warmup, reps)
    print(f"  cuBLAS FWD: {t_cublas_fwd:.3f} ms   BWD: {t_cublas_bwd:.3f} ms"
          f"   (no gather/scatter; reference floor for the matmul portion)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", type=str, default=None)
    parser.add_argument("--M", type=int, default=DEFAULT_M)
    parser.add_argument("--group-size", type=int, default=DEFAULT_GROUP_SIZE)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--skip-cuda", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available — benchmark requires GPU.")
        sys.exit(1)

    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: M={args.M}, group_size={args.group_size}, warmup={args.warmup}, reps={args.reps}")

    if args.shapes:
        shapes = []
        for pair in args.shapes.split(";"):
            k, n = pair.split(",")
            shapes.append((int(k), int(n)))
    else:
        shapes = DEFAULT_SHAPES

    for K, N in shapes:
        bench_one_shape(K, N, args.M, args.group_size, device,
                        args.warmup, args.reps, args.skip_cuda)


if __name__ == "__main__":
    main()


def bench_soft(shape, M, group_size, device, warmup, reps):
    """Benchmark soft (Gumbel-Softmax) kernel vs hard kernel."""
    K, N = shape
    G = N // group_size
    print(f"\n=== SOFT benchmark: K={K}, N={N}, M={M}, G={G} ===")

    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.5
    palette = torch.randn(G, 4, dtype=torch.bfloat16, device=device).requires_grad_(True)
    logits = torch.randn(4, K, N, dtype=torch.float16, device=device) * 0.5
    grad_y = torch.randn(M, N, dtype=torch.bfloat16, device=device) * 0.1

    from fused_lut_linear_cuda import fused_lut_linear_soft

    # Warmup (first call compiles)
    _ = fused_lut_linear_soft(x, palette, logits, None, group_size, tau=1.0)
    torch.cuda.synchronize()

    # FWD only
    def fwd():
        return fused_lut_linear_soft(x, palette, logits, None, group_size, tau=1.0)
    t_fwd = _bench(fwd, warmup, reps)
    print(f"  SOFT FWD: {t_fwd:.3f} ms")

    # FWD + BWD
    def fwd_bwd():
        y = fused_lut_linear_soft(x, palette, logits, None, group_size, tau=1.0)
        y.backward(grad_y, retain_graph=True)
    t_total = _bench(fwd_bwd, warmup, reps)
    print(f"  SOFT TOTAL (FWD+BWD): {t_total:.3f} ms")

    return t_fwd, t_total


def main_soft():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", type=str, default=None)
    p.add_argument("--M", type=int, default=256)   # Use smaller M for soft (memory)
    p.add_argument("--group-size", type=int, default=256)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--reps", type=int, default=30)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)

    device = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: M={args.M}, group_size={args.group_size}, warmup={args.warmup}, reps={args.reps}")

    if args.shapes:
        shapes = []
        for pair in args.shapes.split(";"):
            k, n = pair.split(",")
            shapes.append((int(k), int(n)))
    else:
        shapes = [(2560, 2560), (2560, 1024), (2560, 8192)]

    for shape in shapes:
        K, N = shape
        if N % 256 != 0:
            continue
        try:
            bench_soft(shape, args.M, args.group_size, device, args.warmup, args.reps)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__" and "__file__" in dir():
    # Only run if invoked directly as a script
    if len(sys.argv) > 1 and sys.argv[1] == "--soft":
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        main_soft()
