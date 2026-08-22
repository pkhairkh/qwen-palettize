"""Isolated microbenchmark for each Triton kernel vs torch reference.

For each kernel:
  1. CORRECTNESS: max|triton - torch_ref| (must pass thresholds)
  2. SPEED: median latency over 50 iters (after 10 warmup) + GFLOPS
  3. VS TORCH: ratio vs equivalent torch op (cuBLAS matmul / gather)

Realistic shapes from Qwen3.5-4B PalettizedLinears:
  K=2560, N=1024 (k_proj/v_proj), N=4096 (in_proj_z), N=8192 (q_proj), N=9216 (gate/up_proj)
  K=4096, N=2560 (out_proj/o_proj)
  K=9216, N=2560 (down_proj)
  M=2048 (batch=8, seq=256) or M=4096 (batch=32, seq=128)

Run on server:  python3 bench_triton_kernels.py
"""
from __future__ import annotations
import sys, time, statistics, math
import torch
import triton
import triton.language as tl

sys.path.insert(0, '/root/qwen35_palettize/scripts')

from triton_soft_forward import (
    compute_P_W_ste_triton, fused_soft_matmul_triton, TritonSoftLinear,
)
from triton_soft_backward import (
    fused_soft_bwd_grad_x_triton, fused_soft_bwd_grad_W_triton,
    fused_soft_bwd_elementwise_triton, fused_soft_bwd_chunked_triton,
)
from triton_hard_forward import (
    compute_hard_W_triton, fused_hard_matmul_triton, TritonHardLinear,
)


# ═════════════════════════════════════════════════════════════════════════════
# Timing helper
# ═════════════════════════════════════════════════════════════════════════════
def bench(fn, warmup=10, iters=50) -> float:
    """Returns median latency in milliseconds."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms
    return statistics.median(times)


def gflops(m, k, n, ms) -> float:
    """GFLOPS for a (m,k)@(k,n) matmul = 2*m*k*n FLOPs."""
    if ms <= 0: return 0.0
    return (2.0 * m * k * n) / (ms * 1e-3) / 1e9


# ═════════════════════════════════════════════════════════════════════════════
# Reference implementations (pure torch, no Triton)
# ═════════════════════════════════════════════════════════════════════════════
def torch_soft_forward(x, palette, logits, bias, group_size, tau):
    """Pure-torch soft forward: Gumbel + softmax + STE + matmul."""
    n_planes, K, N = logits.shape
    G, _ = palette.shape
    # Gumbel noise + softmax
    g = torch.distributions.Gumbel(0.0, 1.0).sample((4, K, N)).to(logits.device)
    logits_noisy = (logits.float() + g) / tau
    P = torch.softmax(logits_noisy, dim=0)  # (4, K, N) fp32
    # W_soft = Σ_k P[k] * palette[g, k]
    g_idx = torch.arange(N, device=palette.device) // group_size
    pal_per_col = palette[g_idx.long()].float()  # (N, 4)
    W_soft = torch.einsum('kjo,ok->jo', P, pal_per_col)  # (K, N) fp32 — using 'kjo' for (4,K,N)
    # W_hard = palette[g, argmax(logits, dim=0)]  (NO Gumbel — plain argmax)
    argmax_idx = logits.argmax(dim=0)  # (K, N)
    g_per_col = g_idx[None, :].expand(K, N)
    W_hard = palette[g_per_col.long(), argmax_idx].float()
    # STE: W_ste = W_hard (forward value)
    W_ste = W_hard.to(torch.bfloat16)
    y = torch.matmul(x, W_ste)  # bf16 matmul (cuBLAS, fp32 acc)
    if bias is not None:
        y = y + bias.to(torch.bfloat16)
    return y, W_ste, P


def torch_soft_backward(grad_y, x, palette, P, W_ste, group_size):
    """Pure-torch soft backward (matches our math)."""
    M, K = x.shape
    _, _, N = P.shape
    G, _ = palette.shape
    g_idx = torch.arange(N, device=palette.device) // group_size
    pal_per_col = palette[g_idx.long()].float()  # (N, 4)

    # grad_x = grad_y @ W_ste.T (bf16 matmul)
    grad_x = torch.matmul(grad_y, W_ste.to(torch.bfloat16).T)

    # grad_W_soft = x.T @ grad_y (fp32)
    grad_W = (x.T.float() @ grad_y.float())  # (K, N) fp32

    # P_aos: (K, N, 4) — permute from (4, K, N)
    P_aos = P.permute(1, 2, 0).contiguous()  # (K, N, 4)
    W_soft = (P_aos * pal_per_col[None, :, :]).sum(dim=-1)  # (K, N)

    # grad_logits[k,j,o] = grad_W[j,o] * P[k,j,o] * (pal[o,k] - W_soft[j,o])
    pal_b = pal_per_col[None, :, :].expand(K, N, 4)
    delta = pal_b - W_soft.unsqueeze(-1)
    grad_logits_aos = grad_W.unsqueeze(-1) * P_aos * delta  # (K, N, 4)
    grad_logits = grad_logits_aos.permute(2, 0, 1).contiguous().to(torch.float16)  # (4, K, N)

    # grad_palette[g,k] = Σ_{j,o in g} grad_W[j,o] * P[k,j,o]
    grad_palette = torch.zeros((G, 4), dtype=torch.float32, device=palette.device)
    contrib = grad_W.unsqueeze(-1) * P_aos  # (K, N, 4)
    contrib_sum_j = contrib.sum(dim=0)  # (N, 4)
    for k in range(4):
        grad_palette[:, k] = grad_palette[:, k].scatter_add(0, g_idx.long(), contrib_sum_j[:, k])

    return grad_x.to(torch.bfloat16), grad_logits, grad_palette


def torch_hard_forward(x, palette, indices, bias, group_size):
    K, N = indices.shape
    G, _ = palette.shape
    g_idx = torch.arange(N, device=palette.device) // group_size
    g_per_col = g_idx[None, :].expand(K, N)
    W = palette[g_per_col.long(), indices.long()].to(torch.bfloat16)
    y = torch.matmul(x, W)
    if bias is not None:
        y = y + bias.to(torch.bfloat16)
    return y, W


# ═════════════════════════════════════════════════════════════════════════════
# Test configs (K, N) — from real Qwen3.5-4B PalettizedLinears
# ═════════════════════════════════════════════════════════════════════════════
TEST_SHAPES = [
    (2560, 1024),   # k_proj, v_proj (small)
    (2560, 4096),   # in_proj_z
    (2560, 8192),   # q_proj, in_proj_qkv
    (2560, 9216),   # gate_proj, up_proj (largest N)
    (4096, 2560),   # out_proj, o_proj
    (9216, 2560),   # down_proj (largest K)
]
M_VALS = [2048, 4096]   # batch=8/seq=256, batch=32/seq=128
GROUP_SIZE = 256


def make_inputs(K, N, M, device='cuda', seed=0):
    torch.manual_seed(seed)
    G = N // GROUP_SIZE
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16) * 0.5
    palette = torch.randn(G, 4, device=device, dtype=torch.bfloat16) * 0.3
    logits = torch.full((4, K, N), -10.0, device=device, dtype=torch.float16)
    init_idx = torch.randint(0, 4, (K, N), device=device)
    for k in range(4):
        logits[k][init_idx == k] = 10.0
    logits = logits + torch.randn_like(logits) * 0.5
    indices = torch.randint(0, 4, (K, N), device=device).to(torch.int8)
    bias = torch.randn(N, device=device, dtype=torch.bfloat16) * 0.05
    grad_y = torch.randn(M, N, device=device, dtype=torch.bfloat16) * 0.1
    return x, palette, logits, indices, bias, grad_y, G


# ═════════════════════════════════════════════════════════════════════════════
# Per-kernel benchmark
# ═════════════════════════════════════════════════════════════════════════════
def bench_compute_P_W_ste(K, N, M):
    x, palette, logits, _, bias, _, G = make_inputs(K, N, M)
    # Triton
    def triton_fn():
        compute_P_W_ste_triton(logits, palette, GROUP_SIZE, tau=1.5, step_seed=42)
    # Torch ref (no direct equivalent — closest is manual softmax + einsum)
    def torch_fn():
        # Match the kernel math: Gumbel + softmax + W_soft + argmax + W_ste
        g = torch.distributions.Gumbel(0.0, 1.0).sample((4, K, N)).to(logits.device)
        logits_noisy = (logits.float() + g) / 1.5
        P = torch.softmax(logits_noisy, dim=0)
        g_idx = torch.arange(N, device=palette.device) // GROUP_SIZE
        pal_per_col = palette[g_idx.long()].float()
        _ = (P * pal_per_col.permute(1, 0).unsqueeze(1)).sum(dim=0)  # W_soft
        argmax_idx = logits.argmax(dim=0)
        g_per_col = g_idx[None, :].expand(K, N)
        _ = palette[g_per_col.long(), argmax_idx]  # W_hard
    # Correctness — note: P values won't bit-match (different Gumbel RNG),
    # so just verify W_ste is in palette range + has correct argmax distribution
    P_aos, W_ste = compute_P_W_ste_triton(logits, palette, GROUP_SIZE, 1.5, 42)
    argmax_ref = logits.argmax(dim=0)
    g_idx = torch.arange(N, device=palette.device) // GROUP_SIZE
    g_per_col = g_idx[None, :].expand(K, N)
    W_ste_ref = palette[g_per_col.long(), argmax_ref].to(torch.bfloat16)
    err = (W_ste.float() - W_ste_ref.float()).abs().max().item()
    pal_min, pal_max = palette.float().min().item(), palette.float().max().item()
    wste_min, wste_max = W_ste.float().min().item(), W_ste.float().max().item()
    in_range = wste_min >= pal_min - 1e-3 and wste_max <= pal_max + 1e-3
    t_tri = bench(triton_fn)
    t_tor = bench(torch_fn)
    return {
        'kernel': 'compute_P_W_ste', 'K': K, 'N': N, 'M': M,
        'err_W_ste': err, 'correctness': 'PASS' if (err < 1e-3 and in_range) else 'FAIL',
        'triton_ms': t_tri, 'torch_ms': t_tor, 'speedup': t_tor / t_tri if t_tri > 0 else 0,
        'note': 'P_aos RNG differs from torch ref (expected)',
    }


def bench_soft_matmul(K, N, M):
    x, palette, logits, _, bias, _, G = make_inputs(K, N, M)
    _, W_ste = compute_P_W_ste_triton(logits, palette, GROUP_SIZE, 1.5, 42)
    def triton_fn():
        fused_soft_matmul_triton(x, W_ste, bias)
    def torch_fn():
        y = torch.matmul(x, W_ste)
        if bias is not None:
            y = y + bias
    y_tri = fused_soft_matmul_triton(x, W_ste, bias)
    y_tor = torch.matmul(x, W_ste) + (bias if bias is not None else 0)
    err = (y_tri.float() - y_tor.float()).abs().max().item()
    y_abs_max = y_tor.float().abs().max().item()
    rel_err = err / max(y_abs_max, 1e-6)
    t_tri = bench(triton_fn)
    t_tor = bench(torch_fn)
    return {
        'kernel': 'fused_soft_matmul (x@W_ste+bias)', 'K': K, 'N': N, 'M': M,
        'err_y': err, 'rel_err': rel_err,
        # bf16 has ~3 decimal digits of precision. For K=9216 accumulation,
        # ULP at |y|=80 is 0.5, so err/|y| < 0.01 (1%) is the right threshold.
        'correctness': 'PASS' if rel_err < 0.01 else 'FAIL',
        'triton_ms': t_tri, 'torch_ms': t_tor, 'speedup': t_tor / t_tri if t_tri > 0 else 0,
        'gflops_triton': gflops(M, K, N, t_tri), 'gflops_torch': gflops(M, K, N, t_tor),
    }


def bench_soft_bwd_grad_x(K, N, M):
    x, palette, logits, _, bias, grad_y, G = make_inputs(K, N, M)
    _, W_ste = compute_P_W_ste_triton(logits, palette, GROUP_SIZE, 1.5, 42)
    def triton_fn():
        fused_soft_bwd_grad_x_triton(grad_y, W_ste)
    def torch_fn():
        torch.matmul(grad_y, W_ste.T)
    gx_tri = fused_soft_bwd_grad_x_triton(grad_y, W_ste)
    gx_tor = torch.matmul(grad_y, W_ste.T)
    err = (gx_tri.float() - gx_tor.float()).abs().max().item()
    gx_abs_max = gx_tor.float().abs().max().item()
    rel_err = err / max(gx_abs_max, 1e-6)
    t_tri = bench(triton_fn)
    t_tor = bench(torch_fn)
    # grad_x = grad_y @ W_ste.T: FLOPs = 2 * M * N * K
    return {
        'kernel': 'fused_soft_bwd_grad_x (grad_y@W.T)', 'K': K, 'N': N, 'M': M,
        'err_gx': err, 'rel_err': rel_err,
        'correctness': 'PASS' if rel_err < 0.01 else 'FAIL',
        'triton_ms': t_tri, 'torch_ms': t_tor, 'speedup': t_tor / t_tri if t_tri > 0 else 0,
        'gflops_triton': gflops(M, N, K, t_tri), 'gflops_torch': gflops(M, N, K, t_tor),
    }


def bench_soft_bwd_grad_W(K, N, M):
    x, palette, logits, _, _, grad_y, G = make_inputs(K, N, M)
    def triton_fn():
        fused_soft_bwd_grad_W_triton(x, grad_y)
    def torch_fn():
        x.T.float() @ grad_y.float()
    gW_tri = fused_soft_bwd_grad_W_triton(x, grad_y)
    gW_tor = (x.T.float() @ grad_y.float())
    err = (gW_tri - gW_tor).abs().max().item()
    gW_abs_max = gW_tor.abs().max().item()
    rel_err = err / max(gW_abs_max, 1e-6)
    t_tri = bench(triton_fn)
    t_tor = bench(torch_fn)
    return {
        'kernel': 'fused_soft_bwd_grad_W (x.T@grad_y)', 'K': K, 'N': N, 'M': M,
        'err_gW': err, 'rel_err': rel_err,
        'correctness': 'PASS' if rel_err < 0.01 else 'FAIL',
        'triton_ms': t_tri, 'torch_ms': t_tor, 'speedup': t_tor / t_tri if t_tri > 0 else 0,
        'gflops_triton': gflops(K, M, N, t_tri), 'gflops_torch': gflops(K, M, N, t_tor),
    }


def bench_soft_bwd_elementwise(K, N, M):
    x, palette, logits, _, _, grad_y, G = make_inputs(K, N, M)
    P_aos, _ = compute_P_W_ste_triton(logits, palette, GROUP_SIZE, 1.5, 42)
    grad_W = fused_soft_bwd_grad_W_triton(x, grad_y)
    def triton_fn():
        fused_soft_bwd_elementwise_triton(grad_W, P_aos, palette, GROUP_SIZE)
    # Torch ref
    def torch_fn():
        g_idx = torch.arange(N, device=palette.device) // GROUP_SIZE
        pal_per_col = palette[g_idx.long()].float()
        P_aos_f = P_aos.float()
        W_soft = (P_aos_f * pal_per_col[None, :, :]).sum(dim=-1)
        pal_b = pal_per_col[None, :, :].expand(K, N, 4)
        delta = pal_b - W_soft.unsqueeze(-1)
        _ = (grad_W.unsqueeze(-1) * P_aos_f * delta)  # grad_logits_aos
        # grad_palette (skip — atomic, just do the matmul part for fair compare)
    gl_tri, gp_tri = fused_soft_bwd_elementwise_triton(grad_W, P_aos, palette, GROUP_SIZE)
    # Correctness check vs torch ref
    g_idx = torch.arange(N, device=palette.device) // GROUP_SIZE
    pal_per_col = palette[g_idx.long()].float()
    P_aos_f = P_aos.float()
    W_soft = (P_aos_f * pal_per_col[None, :, :]).sum(dim=-1)
    pal_b = pal_per_col[None, :, :].expand(K, N, 4)
    delta = pal_b - W_soft.unsqueeze(-1)
    gl_ref = (grad_W.unsqueeze(-1) * P_aos_f * delta).permute(2, 0, 1).contiguous().to(torch.float16)
    err = (gl_tri.float() - gl_ref.float()).abs().max().item()
    gl_abs_max = gl_ref.float().abs().max().item()
    rel_err = err / max(gl_abs_max, 1e-6)
    t_tri = bench(triton_fn)
    t_tor = bench(torch_fn)
    return {
        'kernel': 'fused_soft_bwd_elementwise (grad_logits+grad_palette)', 'K': K, 'N': N, 'M': M,
        'err_gl': err, 'rel_err': rel_err,
        'correctness': 'PASS' if rel_err < 0.02 else 'FAIL',
        'triton_ms': t_tri, 'torch_ms': t_tor, 'speedup': t_tor / t_tri if t_tri > 0 else 0,
    }


def bench_soft_bwd_chunked(K, N, M):
    """Patch 18: fused grad_W + elementwise — no HBM grad_W intermediate."""
    x, palette, logits, _, _, grad_y, G = make_inputs(K, N, M)
    P_aos, _ = compute_P_W_ste_triton(logits, palette, GROUP_SIZE, 1.5, 42)
    def triton_fn():
        fused_soft_bwd_chunked_triton(x, grad_y, P_aos, palette, GROUP_SIZE)
    # Torch ref (same as bench_soft_bwd_elementwise — two-step: grad_W + elementwise)
    def torch_fn():
        grad_W = x.T.float() @ grad_y.float()
        g_idx = torch.arange(N, device=palette.device) // GROUP_SIZE
        pal_per_col = palette[g_idx.long()].float()
        P_aos_f = P_aos.float()
        W_soft = (P_aos_f * pal_per_col[None, :, :]).sum(dim=-1)
        pal_b = pal_per_col[None, :, :].expand(K, N, 4)
        delta = pal_b - W_soft.unsqueeze(-1)
        _ = (grad_W.unsqueeze(-1) * P_aos_f * delta)  # grad_logits_aos
    gl_tri, gp_tri = fused_soft_bwd_chunked_triton(x, grad_y, P_aos, palette, GROUP_SIZE)
    # Correctness: compare grad_logits to two-step reference
    grad_W_ref = fused_soft_bwd_grad_W_triton(x, grad_y)
    gl_ref, _ = fused_soft_bwd_elementwise_triton(grad_W_ref, P_aos, palette, GROUP_SIZE)
    err = (gl_tri.float() - gl_ref.float()).abs().max().item()
    gl_abs_max = gl_ref.float().abs().max().item()
    rel_err = err / max(gl_abs_max, 1e-6)
    t_tri = bench(triton_fn)
    t_tor = bench(torch_fn)
    return {
        'kernel': 'fused_soft_bwd_chunked (grad_W+elementwise fused, no HBM grad_W)',
        'K': K, 'N': N, 'M': M,
        'err_gl': err, 'rel_err': rel_err,
        'correctness': 'PASS' if rel_err < 0.02 else 'FAIL',
        'triton_ms': t_tri, 'torch_ms': t_tor, 'speedup': t_tor / t_tri if t_tri > 0 else 0,
        'note': 'Patch 18 — eliminates grad_W HBM intermediate (52MB write + 52MB read per layer)',
    }


def bench_hard_forward(K, N, M):
    x, palette, _, indices, bias, _, G = make_inputs(K, N, M)
    def triton_fn():
        TritonHardLinear.apply(x, palette, indices, bias, GROUP_SIZE)
    def torch_fn():
        W, _ = torch_hard_forward(x, palette, indices, bias, GROUP_SIZE)
    y_tri = TritonHardLinear.apply(x, palette, indices, bias, GROUP_SIZE)
    y_tor, _ = torch_hard_forward(x, palette, indices, bias, GROUP_SIZE)
    err = (y_tri.float() - y_tor.float()).abs().max().item()
    y_abs_max = y_tor.float().abs().max().item()
    rel_err = err / max(y_abs_max, 1e-6)
    t_tri = bench(triton_fn)
    t_tor = bench(torch_fn)
    return {
        'kernel': 'hard_forward (gather+matmul+bias)', 'K': K, 'N': N, 'M': M,
        'err_y': err, 'rel_err': rel_err,
        'correctness': 'PASS' if rel_err < 0.01 else 'FAIL',
        'triton_ms': t_tri, 'torch_ms': t_tor, 'speedup': t_tor / t_tri if t_tri > 0 else 0,
        'gflops_triton': gflops(M, K, N, t_tri), 'gflops_torch': gflops(M, K, N, t_tor),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 100)
    print("TRITON KERNEL MICROBENCHMARK vs TORCH REFERENCE")
    print("=" * 100)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Triton: {triton.__version__}  PyTorch: {torch.__version__}")
    print(f"Test shapes (K, N) from real Qwen3.5-4B PalettizedLinears, M from batch×seq")
    print(f"GROUP_SIZE={GROUP_SIZE}")
    print()

    all_results = []
    for K, N in TEST_SHAPES:
        for M in M_VALS:
            print(f"\n--- (K={K}, N={N}, M={M}) ---")
            for fn in [bench_compute_P_W_ste, bench_soft_matmul,
                       bench_soft_bwd_grad_x, bench_soft_bwd_grad_W,
                       bench_soft_bwd_elementwise, bench_soft_bwd_chunked,
                       bench_hard_forward]:
                try:
                    r = fn(K, N, M)
                    all_results.append(r)
                    sp = r.get('speedup', 0)
                    print(f"  {r['kernel']:50s}  {r['correctness']:4s}  "
                          f"triton={r['triton_ms']:.3f}ms  torch={r['torch_ms']:.3f}ms  "
                          f"speedup={sp:.2f}x", end='')
                    if 'gflops_triton' in r:
                        print(f"  GFLOPS: triton={r['gflops_triton']:.0f}  torch={r['gflops_torch']:.0f}")
                    else:
                        print()
                    if r['correctness'] == 'FAIL':
                        err_key = [k for k in r if k.startswith('err_')][0]
                        print(f"    !!! {err_key}={r[err_key]} — FAILED CORRECTNESS")
                except Exception as e:
                    print(f"  {fn.__name__:50s}  ERROR: {e}")
                    import traceback
                    traceback.print_exc()

    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"{'Kernel':50s} {'K':>5} {'N':>5} {'M':>5} {'corr':>5} {'triton_ms':>10} {'torch_ms':>10} {'speedup':>8}")
    for r in all_results:
        print(f"{r['kernel']:50s} {r['K']:5d} {r['N']:5d} {r['M']:5d} {r['correctness']:>5} "
              f"{r['triton_ms']:>10.3f} {r['torch_ms']:>10.3f} {r.get('speedup', 0):>8.2f}x")


if __name__ == "__main__":
    main()
