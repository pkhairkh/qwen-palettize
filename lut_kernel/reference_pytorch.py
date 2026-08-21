"""Reference (slow) PyTorch implementation of the fused LUT-quantized linear layer.

This is the GROUND TRUTH for correctness. Both the Triton and CUDA kernels must
produce identical output (to within atol=1e-3, rtol=1e-3) on forward and backward.

Matches the spec's `PalettizedLinear` semantics:
    W[j, o] = palette[o // group_size, indices[j, o]]
    y = x @ W + bias
    backward:
        grad_x = grad_y @ W.T
        grad_palette = scatter_add(x.T @ grad_y, _flat_idx)   # fp32 accumulation
        grad_bias = grad_y.sum(0)
"""
from __future__ import annotations
import torch
from torch import Tensor


GROUP_SIZE = 256  # always 256 per spec


def lut_linear_forward(
    x: Tensor,           # (M, K) bf16
    palette: Tensor,      # (G, 4) bf16
    indices: Tensor,      # (K, N) int8
    bias: Tensor | None,  # (N,) bf16 or None
    group_size: int = GROUP_SIZE,
) -> Tensor:
    """Reference forward — materializes W via gather, then matmul.

    This is intentionally slow (the gather is the bottleneck we want to eliminate
    in the fused kernels), but is mathematically correct.
    """
    assert x.dtype == torch.bfloat16
    assert palette.dtype == torch.bfloat16
    assert indices.dtype == torch.int8
    assert x.dim() == 2 and palette.dim() == 2 and indices.dim() == 2
    M, K = x.shape
    K_, N = indices.shape
    G, P = palette.shape
    assert K == K_, f"x K={K} != indices K={K_}"
    assert P == 4, f"palette must have 4 entries per group, got {P}"
    assert N % group_size == 0, f"N={N} must be divisible by group_size={group_size}"
    assert N // group_size == G, f"N//group_size={N//group_size} != G={G}"

    flat_palette = palette.reshape(-1)                          # (G*4,)
    o_idx = torch.arange(N, device=x.device, dtype=torch.long)
    g = o_idx // group_size                                     # (N,)
    flat_idx = g.unsqueeze(0) * 4 + indices.long()             # (K, N) — int64

    # THE GATHER (slow path) — materializes (K, N) bf16 weight matrix
    W = flat_palette[flat_idx]                                  # (K, N) bf16

    # Use fp32 accumulation then cast to bf16 — matches Triton/CUDA kernel
    # behavior (which accumulate in fp32 for numerical stability).
    # PyTorch's bf16 matmul (x @ W) internally uses fp32 accum on H100/L4
    # tensor cores, but the cast-to-bf16 rounding differs slightly.
    # Using explicit fp32 matmul removes this discrepancy in the correctness test.
    y = (x.float() @ W.float()).to(torch.bfloat16)
    if bias is not None:
        y = y + bias
    return y


def lut_linear_backward(
    grad_y: Tensor,      # (M, N) bf16
    x: Tensor,           # (M, K) bf16
    palette: Tensor,     # (G, 4) bf16
    indices: Tensor,     # (K, N) int8
    bias: Tensor | None, # (N,) bf16 or None
    group_size: int = GROUP_SIZE,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """Reference backward — autograd-style decomposition.

    Returns (grad_x, grad_palette, grad_bias).
    grad_palette is bf16 (matches model param dtype) but accumulation done in fp32.
    """
    M, K = x.shape
    K_, N = indices.shape
    G, P = palette.shape

    flat_palette = palette.reshape(-1)
    o_idx = torch.arange(N, device=x.device, dtype=torch.long)
    g = o_idx // group_size
    flat_idx = g.unsqueeze(0) * 4 + indices.long()             # (K, N)
    W = flat_palette[flat_idx]                                  # (K, N) bf16 — THE GATHER

    # grad_x: (M, K) bf16
    grad_x = grad_y @ W.T

    # dW: (K, N) bf16 — materialized (THIS is what the fused kernel avoids)
    dW = x.T.float() @ grad_y.float()                          # fp32 for accuracy

    # grad_palette: scatter_add along flat_idx
    grad_palette_flat = torch.zeros(G * 4, dtype=torch.float32, device=x.device)
    grad_palette_flat.scatter_add_(0, flat_idx.reshape(-1), dW.reshape(-1))
    grad_palette = grad_palette_flat.reshape(G, P).to(torch.bfloat16)

    grad_bias = grad_y.sum(dim=0) if bias is not None else None
    return grad_x, grad_palette, grad_bias


class ReferencePalettizedLinear(torch.nn.Module):
    """Slow reference Module — mirrors the user's PalettizedLinear API."""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = GROUP_SIZE,
        bias: bool = False,
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        assert out_features % group_size == 0
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.n_groups = out_features // group_size

        # Trainable palette — init via small k-means-style centroid spread
        # (in real use, comes from offline GPTQ; here just randn spread for testing)
        palette = torch.randn(self.n_groups, 4, dtype=torch.bfloat16, device=device) * 0.1
        # Make 4 entries per group span a small range so different indices matter
        offsets = torch.tensor([-1.5, -0.5, 0.5, 1.5], dtype=torch.bfloat16, device=device)
        self.palette = torch.nn.Parameter(palette + offsets)

        # Frozen indices — random 0..3
        indices = torch.randint(0, 4, (in_features, out_features), dtype=torch.int8, device=device)
        self.register_buffer("indices", indices)

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.bfloat16, device=device))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        return lut_linear_forward(x, self.palette, self.indices, self.bias, self.group_size)


if __name__ == "__main__":
    # Sanity check: just verify shapes and dtypes
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mod = ReferencePalettizedLinear(2560, 2560, bias=True, device=device).to(device)
    x = torch.randn(8, 128, 2560, dtype=torch.bfloat16, device=device).view(1024, 2560)
    y = mod(x)
    print(f"y.shape={y.shape}, dtype={y.dtype}")
    assert y.shape == (1024, 2560)
    print("OK")
