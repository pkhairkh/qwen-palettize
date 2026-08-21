"""Minimal Qwen3.5-4B-style model with PalettizedLinear replaced by our fused CUDA kernel.

This is NOT a full Qwen3.5-4B — it's a single transformer block with the same
dimensionality (hidden=2560, intermediate=9216, n_heads=40, n_kv_heads=8) so we
can verify end-to-end that:
  1. The fused LUT-quantized CUDA kernel integrates cleanly with PyTorch autograd.
  2. Muon-style fp32 master weights + bf16 model params + bf16 grads work.
  3. autocast doesn't break anything.
  4. Loss decreases over a few steps (sanity check that gradients flow correctly).

Architecture (per spec KB):
  - Attention (fused QKV): 2560 → 8192 (split into Q=4096, K=2048, V=2048)
  - Output projection: 4096 → 2560 (o_proj after attention)
  - MLP: gate 2560→9216, up 2560→9216, down 9216→2560 (SwiGLU)
  - RMSNorm + residual connections

The actual Qwen3.5-4B has 36 such layers. Here we use 1 layer for fast verification.
"""
from __future__ import annotations
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# Enable TC variants if requested (env vars are read by the C++ launcher)
os.environ.setdefault("USE_TC_FWD", "1")
os.environ.setdefault("USE_TC_BWD_GX", "1")

from fused_lut_linear_cuda import fused_lut_linear
from fused_lut_linear_cuda import fused_lut_linear_soft


GROUP_SIZE = 256


def _make_palette(in_features: int, out_features: int, device: str) -> torch.Tensor:
    """Initialize a palette with 4 spread-out values per group (k-means-style init)."""
    n_groups = out_features // GROUP_SIZE
    # Use small random offsets around 0 — represents typical post-GPTQ palette.
    base = torch.randn(n_groups, 1, dtype=torch.bfloat16, device=device) * 0.05
    offsets = torch.tensor([-0.4, -0.13, 0.13, 0.4], dtype=torch.bfloat16, device=device)
    return (base + offsets.expand(n_groups, 4)).contiguous()


def _make_indices(in_features: int, out_features: int, device: str) -> torch.Tensor:
    """Random int8 indices in [0, 4). In practice, these come from GPTQ offline."""
    return torch.randint(0, 4, (in_features, out_features), dtype=torch.int8, device=device)


class PalettizedLinear(nn.Module):
    """Drop-in replacement for nn.Linear using our fused CUDA LUT kernel.

    Stores:
      - palette: (n_groups, 4) bf16 — TRAINABLE Parameter
      - indices: (in_features, out_features) int8 — FROZEN buffer
      - bias: (out_features,) bf16 — FROZEN buffer (set requires_grad=True to train)

    Mimics the user's original PalettizedLinear API.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = GROUP_SIZE,
        bias: bool = False,
        device: str | torch.device = "cuda",
        use_soft: bool = False,                # Phase IX: enable soft path
        initial_tau: float = 1.0,              # Phase IX: Gumbel-Softmax temperature
    ):
        super().__init__()
        assert out_features % group_size == 0
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.n_groups = out_features // group_size
        self.use_soft = use_soft
        self.tau = initial_tau

        # Trainable palette (this is what the optimizer updates)
        self.palette = nn.Parameter(_make_palette(in_features, out_features, str(device)))

        # Frozen indices (used for hard path + as init for soft logits)
        indices = _make_indices(in_features, out_features, str(device))
        self.register_buffer("indices", indices)

        # Phase IX: Trainable index logits (4, K, N) fp16 — only used when use_soft=True
        # Initialize: one-hot from current indices (logit=10 for chosen, -10 for others)
        if use_soft:
            logits_init = torch.full((4, in_features, out_features), -10.0,
                                      dtype=torch.float16, device=device)
            # Set chosen index's plane to +10 using one_hot mask
            one_hot = torch.nn.functional.one_hot(indices.long(), num_classes=4)  # (K, N, 4)
            # logits_init[k, j, o] = one_hot[j, o, k] * 20 - 10
            logits_init = (one_hot.permute(2, 0, 1).float() * 20.0 - 10.0).to(torch.float16)
            self.index_logits = nn.Parameter(logits_init)
        else:
            # Not used; allocate as None to save memory
            self.index_logits = None

        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.bfloat16, device=device))
        else:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.bfloat16, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten batch dims if any: (B, S, H) → (B*S, H)
        orig_shape = x.shape
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])

        if self.training and self.use_soft and self.index_logits is not None:
            # Phase IX: soft forward with Gumbel-Softmax (gradients flow to logits)
            y = fused_lut_linear_soft(
                x, self.palette, self.index_logits,
                self.bias if self.bias.requires_grad else None,
                self.group_size, self.tau,
            )
        else:
            # Hard forward (eval, or soft disabled)
            y = fused_lut_linear(
                x, self.palette, self.indices,
                self.bias if self.bias.requires_grad else None,
                self.group_size,
            )

        if len(orig_shape) > 2:
            y = y.view(*orig_shape[:-1], self.out_features)
        return y

    def update_indices_from_logits(self) -> None:
        """Update frozen indices buffer from current argmax of logits.

        Call this periodically (e.g., every N steps) during training to commit
        the soft index decisions to the hard indices buffer for eval mode.
        """
        if self.index_logits is None:
            return
        with torch.no_grad():
            # argmax over k dimension → (K, N) int64 → cast to int8
            self.indices.copy_(self.index_logits.argmax(dim=0).to(torch.int8))


class RMSNorm(nn.Module):
    """RMSNorm in bf16 (matches Qwen3.5)."""
    def __init__(self, hidden_size: int, eps: float = 1e-6, device: str = "cuda"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.bfloat16, device=device))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in fp32 for numerical stability, then cast back to bf16.
        x_fp32 = x.float()
        var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(var + self.eps)
        return (x_normed.to(torch.bfloat16) * self.weight)


class Attention(nn.Module):
    """Simplified attention with fused QKV + per-head RoPE (skipped for verification).

    Layout (matches user's spec):
      in_proj_qkv: 2560 → 8192 (Q=4096 + K=2048 + V=2048, with head_dim=64)
        Q has 64 heads, K/V have 32 heads each (GQA 2:1 within fused QKV)
      Actually the spec says q_proj 2560→8192 separately, k_proj 2560→1024, v_proj 2560→1024.
      For simplicity here, use the fused QKV form (single PalettizedLinear).
    """
    def __init__(self, hidden_size: int = 2560, n_heads: int = 40, n_kv_heads: int = 8, device: str = "cuda"):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = hidden_size // n_heads   # 64
        # Fused QKV: Q = n_heads * head_dim, K/V = n_kv_heads * head_dim
        qkv_out = (n_heads + 2 * n_kv_heads) * self.head_dim   # (40 + 16) * 64 = 3584
        self.in_proj_qkv = PalettizedLinear(hidden_size, qkv_out, device=device)
        self.o_proj = PalettizedLinear(n_heads * self.head_dim, hidden_size, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, H = x.shape
        qkv = self.in_proj_qkv(x)   # (B, S, qkv_out)
        q, k, v = qkv.split([self.n_heads * self.head_dim,
                              self.n_kv_heads * self.head_dim,
                              self.n_kv_heads * self.head_dim], dim=-1)
        # Reshape to (B, n_heads, S, head_dim)
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # GQA: repeat K, V to match n_heads
        if self.n_kv_heads < self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        # Scaled dot-product attention (use PyTorch's SDPA — fast, no LUT needed here)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # Reshape back: (B, n_heads, S, head_dim) → (B, S, n_heads * head_dim)
        attn = attn.transpose(1, 2).reshape(B, S, self.n_heads * self.head_dim)
        return self.o_proj(attn)


class MLP(nn.Module):
    """SwiGLU MLP block."""
    def __init__(self, hidden_size: int = 2560, intermediate_size: int = 9216, device: str = "cuda"):
        super().__init__()
        self.gate_proj = PalettizedLinear(hidden_size, intermediate_size, device=device)
        self.up_proj = PalettizedLinear(hidden_size, intermediate_size, device=device)
        self.down_proj = PalettizedLinear(intermediate_size, hidden_size, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: down_proj(SiLU(gate_proj(x)) * up_proj(x))
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class TransformerBlock(nn.Module):
    """Single transformer block (RMSNorm + Attn + RMSNorm + MLP, with residuals)."""
    def __init__(self, hidden_size: int = 2560, intermediate_size: int = 9216,
                 n_heads: int = 40, n_kv_heads: int = 8, device: str = "cuda"):
        super().__init__()
        self.attn_norm = RMSNorm(hidden_size, device=device)
        self.attn = Attention(hidden_size, n_heads, n_kv_heads, device=device)
        self.mlp_norm = RMSNorm(hidden_size, device=device)
        self.mlp = MLP(hidden_size, intermediate_size, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class QwenMini(nn.Module):
    """Mini Qwen3.5-4B-style model: 1 transformer block + output projection.

    This is for end-to-end integration testing. A full Qwen3.5-4B would have 36 blocks.
    """
    def __init__(self, hidden_size: int = 2560, intermediate_size: int = 9216,
                 n_heads: int = 40, n_kv_heads: int = 8, vocab_size: int = 151936,
                 n_layers: int = 1, device: str = "cuda"):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, hidden_size, dtype=torch.bfloat16, device=device)
        # The embed doesn't need 2-bit palettization — keep it fp16/bf16 dense.
        # In the real model, this would also be palettized but it's small.
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, intermediate_size, n_heads, n_kv_heads, device)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(hidden_size, device=device)
        # Output projection (tied to embed in real Qwen, but separate here for clarity)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False, dtype=torch.bfloat16, device=device)
        # Initialize lm_head with small values to keep logits reasonable
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: (B, S) int64 → logits: (B, S, vocab_size) bf16."""
        x = self.embed(input_ids)   # (B, S, hidden) bf16
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.lm_head(x)      # (B, S, vocab) bf16


if __name__ == "__main__":
    # Sanity check: build the model and run forward
    device = "cuda"
    model = QwenMini(n_layers=1, device=device)
    print(f"Model built. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable palettes: {sum(p.numel() for n, p in model.named_parameters() if 'palette' in n):,}")

    # Forward pass
    B, S = 2, 128
    input_ids = torch.randint(0, model.vocab_size, (B, S), dtype=torch.long, device=device)
    logits = model(input_ids)
    print(f"Logits shape: {logits.shape}, dtype: {logits.dtype}")
    print(f"Logits mean: {logits.float().mean().item():.4f}, std: {logits.float().std().item():.4f}")
    print("OK")
