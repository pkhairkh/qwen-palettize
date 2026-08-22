"""Python wrapper for the CUDA fused LUT-quantized linear layer.

Loads fused_lut_kernel.cu via torch.utils.cpp_extension.load_inline (or load from
file for cleaner separation). Exposes a `torch.autograd.Function` so it slots
into PyTorch's autograd + autocast + DDP/FSDP stack.

Usage
-----
    from fused_lut_linear_cuda import fused_lut_linear, CUDAFusedLUTLinear

    # Functional
    y = fused_lut_linear(x, palette, indices, bias, group_size=256)

    # Autograd-friendly: gradients flow to `palette` only.
    y = CUDAFusedLUTLinear.apply(x, palette, indices, bias, group_size)

Build
-----
First import will compile via `load_inline` (cached under ~/.cache/torch_extensions/).
To force recompile, delete that cache directory.

Compile flags target sm_89 (L4 / Ada Lovelace) by default, with sm_80 + sm_90
fallbacks so the same .cu file also runs on A100 / H100.
"""
from __future__ import annotations
import os
import os
import torch
from torch import Tensor
from torch.utils.cpp_extension import load_inline


HERE = os.path.dirname(os.path.abspath(__file__))
CU_KERNEL_PATH = os.path.join(HERE, "fused_lut_kernel.cu")


# ─────────────────────────────────────────────────────────────────────────────
#  Read .cu source
# ─────────────────────────────────────────────────────────────────────────────
def _read_cu_source() -> str:
    with open(CU_KERNEL_PATH, "r") as f:
        return f.read()


# ─────────────────────────────────────────────────────────────────────────────
#  C++ host-side wrapper declarations (matches Pybind signatures)
# ─────────────────────────────────────────────────────────────────────────────
CPP_SOURCE = r"""
#include <torch/extension.h>
#include <c10/util/BFloat16.h>
#include <cstdint>

// Launcher prototypes — use c10::BFloat16 (host-side, layout-compatible with __nv_bfloat16).
// The .cu launchers cast to __nv_bfloat16* internally.
void fused_lut_linear_fwdLauncher(
    const c10::BFloat16* x, const c10::BFloat16* palette, const uint8_t* indices,
    const c10::BFloat16* bias, c10::BFloat16* y,
    int M, int K, int N, int group_size);

void fused_lut_linear_bwd_grad_xLauncher(
    const c10::BFloat16* grad_y, const c10::BFloat16* palette, const uint8_t* indices,
    c10::BFloat16* grad_x, int M, int K, int N, int group_size);

void fused_lut_linear_bwd_grad_paletteLauncher(
    const c10::BFloat16* x, const c10::BFloat16* grad_y,
    const c10::BFloat16* palette,  // ADDED in Phase I
    const uint8_t* indices,
    float* grad_palette, int M, int K, int N, int group_size);

void fused_lut_linear_bwd_grad_biasLauncher(
    const c10::BFloat16* grad_y, c10::BFloat16* grad_bias, int M, int N);

// ── Pybind wrappers (allocate output tensors then call launchers) ────────────
torch::Tensor fused_lut_linear_fwd(
    torch::Tensor x,        // (M, K) bf16
    torch::Tensor palette,   // (G, 4) bf16
    torch::Tensor indices,   // (K, N) int8
    c10::optional<torch::Tensor> bias,  // (N,) bf16 or None
    int64_t group_size
) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kBFloat16, "x must be bf16 cuda");
    TORCH_CHECK(palette.is_cuda() && palette.dtype() == torch::kBFloat16, "palette must be bf16 cuda");
    TORCH_CHECK(indices.is_cuda() && (indices.dtype() == torch::kInt8 || indices.dtype() == torch::kUInt8),
                "indices must be int8 or uint8 cuda");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(palette.is_contiguous(), "palette must be contiguous");
    TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");

    int M = x.size(0);
    int K = x.size(1);
    int N = indices.size(1);
    int G = palette.size(0);

    TORCH_CHECK(palette.size(1) == 4, "palette must have 4 entries per group");
    TORCH_CHECK(indices.size(0) == K, "indices K mismatch");
    TORCH_CHECK(N % group_size == 0, "N must be divisible by group_size");
    TORCH_CHECK(N / group_size == G, "G mismatch: N/group_size != palette.size(0)");

    auto y = torch::empty({M, N}, x.options());

    const c10::BFloat16* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().defined()) {
        TORCH_CHECK(bias.value().dtype() == torch::kBFloat16, "bias must be bf16");
        TORCH_CHECK(bias.value().is_contiguous(), "bias must be contiguous");
        bias_ptr = bias.value().data_ptr<c10::BFloat16>();
    }

    fused_lut_linear_fwdLauncher(
        x.data_ptr<c10::BFloat16>(),
        palette.data_ptr<c10::BFloat16>(),
        reinterpret_cast<const uint8_t*>(indices.data_ptr()),
        bias_ptr,
        y.data_ptr<c10::BFloat16>(),
        M, K, N, (int)group_size
    );
    return y;
}

std::vector<torch::Tensor> fused_lut_linear_bwd(
    torch::Tensor grad_y,   // (M, N) bf16
    torch::Tensor x,        // (M, K) bf16
    torch::Tensor palette,  // (G, 4) bf16
    torch::Tensor indices,  // (K, N) int8
    c10::optional<torch::Tensor> bias,  // (N,) bf16 or None — only used to decide if grad_bias is returned
    int64_t group_size,
    bool needs_grad_x,
    bool needs_grad_palette
) {
    TORCH_CHECK(grad_y.is_cuda() && grad_y.dtype() == torch::kBFloat16);
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kBFloat16);
    TORCH_CHECK(palette.is_cuda() && palette.dtype() == torch::kBFloat16);
    TORCH_CHECK(indices.is_cuda() && (indices.dtype() == torch::kInt8 || indices.dtype() == torch::kUInt8));

    int M = x.size(0);
    int K = x.size(1);
    int N = indices.size(1);
    int G = palette.size(0);

    torch::Tensor grad_x;
    torch::Tensor grad_palette;
    torch::Tensor grad_bias;

    if (needs_grad_x) {
        grad_x = torch::empty({M, K}, x.options());
        fused_lut_linear_bwd_grad_xLauncher(
            grad_y.data_ptr<c10::BFloat16>(),
            palette.data_ptr<c10::BFloat16>(),
            reinterpret_cast<const uint8_t*>(indices.data_ptr()),
            grad_x.data_ptr<c10::BFloat16>(),
            M, K, N, (int)group_size
        );
    } else {
        grad_x = torch::Tensor();
    }

    if (needs_grad_palette) {
        // Allocate fp32 accumulator
        grad_palette = torch::zeros({G, 4}, x.options().dtype(torch::kFloat32));
        fused_lut_linear_bwd_grad_paletteLauncher(
            x.data_ptr<c10::BFloat16>(),
            grad_y.data_ptr<c10::BFloat16>(),
            palette.data_ptr<c10::BFloat16>(),  // ADDED in Phase I
            reinterpret_cast<const uint8_t*>(indices.data_ptr()),
            grad_palette.data_ptr<float>(),
            M, K, N, (int)group_size
        );
        // Cast back to bf16 (matching palette dtype) for autograd compatibility
        grad_palette = grad_palette.to(torch::kBFloat16);
    } else {
        grad_palette = torch::Tensor();
    }

    if (bias.has_value() && bias.value().defined()) {
        grad_bias = torch::empty({N}, x.options());
        fused_lut_linear_bwd_grad_biasLauncher(
            grad_y.data_ptr<c10::BFloat16>(),
            grad_bias.data_ptr<c10::BFloat16>(),
            M, N
        );
    } else {
        grad_bias = torch::Tensor();
    }

    return {grad_x, grad_palette, grad_bias};
}

// ═══════════════════════════════════════════════════════════════════════════
//  PHASE IX — Soft (Gumbel-Softmax) launcher prototypes + wrappers
// ═══════════════════════════════════════════════════════════════════════════

// Soft compute_P_W launcher prototype
void fused_lut_linear_soft_compute_P_W_Launcher(
    const c10::Half* logits, const c10::BFloat16* palette,
    c10::Half* P, c10::BFloat16* W_out,
    int K, int N, int group_size, float tau, uint32_t step_seed);

// Soft bwd_grad_logits launcher prototype
void fused_lut_linear_soft_bwd_grad_logits_Launcher(
    const float* grad_W, const c10::Half* P, const c10::BFloat16* palette,
    c10::Half* grad_logits,
    int K, int N, int group_size);

// Soft bwd_grad_palette launcher prototype
void fused_lut_linear_soft_bwd_grad_palette_Launcher(
    const float* grad_W, const c10::Half* P,
    float* grad_palette,
    int K, int N, int group_size);

// PHASE IX.b: Fused soft backward launcher prototype
// (computes grad_logits + grad_palette in one pass, with grad_W on-the-fly)
void fused_lut_linear_soft_bwd_fused_Launcher(
    const c10::BFloat16* grad_y, const c10::BFloat16* x,
    const c10::Half* P, const c10::BFloat16* palette,
    c10::Half* grad_logits, float* grad_palette,
    int M, int K, int N, int group_size);

// Patch 5a — AoS-output compute_P_W launcher prototype (P written as (K,N,4))
void fused_lut_linear_soft_compute_P_W_aos_Launcher(
    const c10::Half* logits, const c10::BFloat16* palette,
    c10::Half* P_aos, c10::BFloat16* W_out,
    int K, int N, int group_size, float tau, uint32_t step_seed);

// Patch 5b — AoS-reading fused backward launcher prototype
// (reads P as (K,N,4) AoS; grad_logits output still (4,K,N) SoA)
void fused_lut_linear_soft_bwd_fused_aos_Launcher(
    const c10::BFloat16* grad_y, const c10::BFloat16* x,
    const c10::Half* P_aos, const c10::BFloat16* palette,
    c10::Half* grad_logits, float* grad_palette,
    int M, int K, int N, int group_size);

// ── Soft forward wrapper ─────────────────────────────────────────────────────
// Returns: (y, P, W)
//   y: (M, N) bf16
//   P: (4, K, N) fp16 — saved for backward
//   W: (K, N) bf16 — materialized weights, saved for backward
//
// Note: the actual matmul y = x @ W is done in Python via torch.matmul (cuBLAS).
// This wrapper only calls the compute_P_W kernel.
std::vector<torch::Tensor> fused_lut_linear_soft_fwd(
    torch::Tensor x,         // (M, K) bf16 — used only to infer shapes/options
    torch::Tensor palette,   // (G, 4) bf16
    torch::Tensor logits,    // (4, K, N) fp16
    int64_t group_size,
    double tau,
    int64_t step_seed
) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kBFloat16);
    TORCH_CHECK(palette.is_cuda() && palette.dtype() == torch::kBFloat16);
    TORCH_CHECK(logits.is_cuda() && logits.dtype() == torch::kHalf,
                "logits must be fp16 (torch.kHalf)");
    TORCH_CHECK(logits.dim() == 3 && logits.size(0) == 4, "logits must be (4, K, N)");

    int K = logits.size(1);
    int N = logits.size(2);
    int G = palette.size(0);
    TORCH_CHECK(palette.size(1) == 4, "palette must be (G, 4)");
    TORCH_CHECK(N % group_size == 0, "N must be divisible by group_size");
    TORCH_CHECK(N / group_size == G, "G mismatch: N/group_size != palette.size(0)");

    // Allocate P (4, K, N) fp16 and W (K, N) bf16
    auto P = torch::empty({4, K, N}, logits.options());
    auto W = torch::empty({K, N}, x.options());

    fused_lut_linear_soft_compute_P_W_Launcher(
        logits.data_ptr<c10::Half>(),
        palette.data_ptr<c10::BFloat16>(),
        P.data_ptr<c10::Half>(),
        W.data_ptr<c10::BFloat16>(),
        K, N, (int)group_size, (float)tau, (uint32_t)step_seed);

    // y = x @ W + bias (cuBLAS, done in Python)
    auto y = torch::matmul(x, W);

    return {y, P, W};
}

// ── Soft backward wrapper ────────────────────────────────────────────────────
// Returns: (grad_logits, grad_palette)
//   grad_logits: (4, K, N) fp16
//   grad_palette: (G, 4) bf16 (after fp32 accumulation)
//
// Note: grad_W and grad_x are computed in Python via torch.matmul (cuBLAS).
// This wrapper only calls the bwd_grad_logits + bwd_grad_palette kernels.
std::vector<torch::Tensor> fused_lut_linear_soft_bwd(
    torch::Tensor grad_W,    // (K, N) fp32 — computed in Python as (x.T @ grad_y).float()
    torch::Tensor P,         // (4, K, N) fp16
    torch::Tensor palette,   // (G, 4) bf16
    int64_t group_size
) {
    TORCH_CHECK(grad_W.is_cuda() && grad_W.dtype() == torch::kFloat32);
    TORCH_CHECK(P.is_cuda() && P.dtype() == torch::kHalf);
    TORCH_CHECK(palette.is_cuda() && palette.dtype() == torch::kBFloat16);
    TORCH_CHECK(P.dim() == 3 && P.size(0) == 4, "P must be (4, K, N)");

    int K = grad_W.size(0);
    int N = grad_W.size(1);
    int G = palette.size(0);

    // Allocate grad_logits (4, K, N) fp16
    auto grad_logits = torch::empty({4, K, N}, P.options());

    // Allocate grad_palette (G, 4) fp32 (atomic accum), then cast to bf16 at the end
    auto grad_palette_fp32 = torch::zeros({G, 4}, grad_W.options());

    fused_lut_linear_soft_bwd_grad_logits_Launcher(
        grad_W.data_ptr<float>(),
        P.data_ptr<c10::Half>(),
        palette.data_ptr<c10::BFloat16>(),
        grad_logits.data_ptr<c10::Half>(),
        K, N, (int)group_size);

    fused_lut_linear_soft_bwd_grad_palette_Launcher(
        grad_W.data_ptr<float>(),
        P.data_ptr<c10::Half>(),
        grad_palette_fp32.data_ptr<float>(),
        K, N, (int)group_size);

    // Cast grad_palette fp32 → bf16 for autograd compatibility
    auto grad_palette = grad_palette_fp32.to(torch::kBFloat16);

    return {grad_logits, grad_palette};
}

// ── PHASE IX.b: Fused soft backward wrapper ─────────────────────────────────
// Computes grad_logits + grad_palette in a SINGLE kernel pass.
// grad_W is computed on-the-fly inside the kernel (no intermediate tensor).
//
// Inputs:
//   grad_y: (M, N) bf16
//   x: (M, K) bf16
//   P: (4, K, N) fp16
//   palette: (G, 4) bf16
//
// Returns: (grad_logits, grad_palette)
//   grad_logits: (4, K, N) fp16
//   grad_palette: (G, 4) bf16 (after fp32 atomic accumulation)
std::vector<torch::Tensor> fused_lut_linear_soft_bwd_fused(
    torch::Tensor grad_y,   // (M, N) bf16
    torch::Tensor x,        // (M, K) bf16
    torch::Tensor P,        // (4, K, N) fp16
    torch::Tensor palette,  // (G, 4) bf16
    int64_t group_size
) {
    TORCH_CHECK(grad_y.is_cuda() && grad_y.dtype() == torch::kBFloat16);
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kBFloat16);
    TORCH_CHECK(P.is_cuda() && P.dtype() == torch::kHalf);
    TORCH_CHECK(palette.is_cuda() && palette.dtype() == torch::kBFloat16);
    TORCH_CHECK(P.dim() == 3 && P.size(0) == 4, "P must be (4, K, N)");

    int M = x.size(0);
    int K = x.size(1);
    int N = grad_y.size(1);
    int G = palette.size(0);

    // Allocate grad_logits (4, K, N) fp16
    auto grad_logits = torch::empty({4, K, N}, P.options());

    // Allocate grad_palette (G, 4) fp32 (atomic accum), then cast to bf16 at the end
    auto grad_palette_fp32 = torch::zeros({G, 4}, grad_y.options().dtype(torch::kFloat32));

    fused_lut_linear_soft_bwd_fused_Launcher(
        grad_y.data_ptr<c10::BFloat16>(),
        x.data_ptr<c10::BFloat16>(),
        P.data_ptr<c10::Half>(),
        palette.data_ptr<c10::BFloat16>(),
        grad_logits.data_ptr<c10::Half>(),
        grad_palette_fp32.data_ptr<float>(),
        M, K, N, (int)group_size);

    // Cast grad_palette fp32 → bf16 for autograd compatibility
    auto grad_palette = grad_palette_fp32.to(torch::kBFloat16);

    return {grad_logits, grad_palette};
}

// ═══════════════════════════════════════════════════════════════════════════
//  PATCH 5 — AoS variants (forward writes P_aos as (K, N, 4); backward reads it)
// ═══════════════════════════════════════════════════════════════════════════

// ── Patch 5c.1: Soft forward (AoS P output) ─────────────────────────────────
// Returns: (y, P_aos, W)
//   y:     (M, N)    bf16
//   P_aos: (K, N, 4) fp16 — saved for backward (contiguous, AoS)
//   W:     (K, N)    bf16 — materialized weights, saved for backward
//
// Wave 1 optimization: NO matmul here — caller does STE matmul in Python.
// Returns only {P_aos, W_soft} so the caller can compute W_ste and do ONE matmul.
std::vector<torch::Tensor> fused_lut_linear_soft_fwd_aos(
    torch::Tensor x,         // (M, K) bf16 — used only to infer shapes/options
    torch::Tensor palette,   // (G, 4) bf16
    torch::Tensor logits,    // (4, K, N) fp16
    int64_t group_size,
    double tau,
    int64_t step_seed
) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kBFloat16);
    TORCH_CHECK(palette.is_cuda() && palette.dtype() == torch::kBFloat16);
    TORCH_CHECK(logits.is_cuda() && logits.dtype() == torch::kHalf,
                "logits must be fp16 (torch.kHalf)");
    TORCH_CHECK(logits.dim() == 3 && logits.size(0) == 4, "logits must be (4, K, N)");

    int K = logits.size(1);
    int N = logits.size(2);
    int G = palette.size(0);
    TORCH_CHECK(palette.size(1) == 4, "palette must be (G, 4)");
    TORCH_CHECK(N % group_size == 0, "N must be divisible by group_size");
    TORCH_CHECK(N / group_size == G, "G mismatch: N/group_size != palette.size(0)");

    // Allocate P_aos (K, N, 4) fp16 — contiguous last-dim, AoS layout
    auto P_aos = torch::empty({K, N, 4}, logits.options());
    auto W = torch::empty({K, N}, x.options());

    fused_lut_linear_soft_compute_P_W_aos_Launcher(
        logits.data_ptr<c10::Half>(),
        palette.data_ptr<c10::BFloat16>(),
        P_aos.data_ptr<c10::Half>(),
        W.data_ptr<c10::BFloat16>(),
        K, N, (int)group_size, (float)tau, (uint32_t)step_seed);

    // NO matmul — caller does STE: y = x @ W_ste (one matmul, not two)
    return {P_aos, W};
}

// ── Patch 5c.2: Fused soft backward (AoS P input) ───────────────────────────
// Reads P_aos in (K, N, 4) AoS layout. Outputs grad_logits in (4, K, N) SoA
// (optimizer + checkpoint format expect SoA logits/gradients).
//
// Inputs:
//   grad_y:    (M, N)    bf16
//   x:         (M, K)    bf16
//   P_aos:     (K, N, 4) fp16 — AoS layout (from fused_lut_linear_soft_fwd_aos)
//   palette:   (G, 4)    bf16
//
// Returns: (grad_logits, grad_palette)
//   grad_logits:  (4, K, N) fp16  — SoA (matches existing autograd contract)
//   grad_palette: (G, 4)    bf16  (after fp32 atomic accumulation)
std::vector<torch::Tensor> fused_lut_linear_soft_bwd_fused_aos(
    torch::Tensor grad_y,   // (M, N) bf16
    torch::Tensor x,        // (M, K) bf16
    torch::Tensor P_aos,    // (K, N, 4) fp16 — AoS
    torch::Tensor palette,  // (G, 4) bf16
    int64_t group_size
) {
    TORCH_CHECK(grad_y.is_cuda() && grad_y.dtype() == torch::kBFloat16);
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kBFloat16);
    TORCH_CHECK(P_aos.is_cuda() && P_aos.dtype() == torch::kHalf);
    TORCH_CHECK(palette.is_cuda() && palette.dtype() == torch::kBFloat16);
    TORCH_CHECK(P_aos.dim() == 3 && P_aos.size(2) == 4,
                "P_aos must be (K, N, 4) fp16 (AoS layout)");

    int M = x.size(0);
    int K = x.size(1);
    int N = grad_y.size(1);
    int G = palette.size(0);

    // Allocate grad_logits (4, K, N) fp16 — SoA OUTPUT (matches optimizer contract)
    auto grad_logits = torch::empty({4, K, N}, P_aos.options());

    // Allocate grad_palette (G, 4) fp32 (atomic accum), then cast to bf16 at the end
    auto grad_palette_fp32 = torch::zeros({G, 4}, grad_y.options().dtype(torch::kFloat32));

    fused_lut_linear_soft_bwd_fused_aos_Launcher(
        grad_y.data_ptr<c10::BFloat16>(),
        x.data_ptr<c10::BFloat16>(),
        P_aos.data_ptr<c10::Half>(),
        palette.data_ptr<c10::BFloat16>(),
        grad_logits.data_ptr<c10::Half>(),
        grad_palette_fp32.data_ptr<float>(),
        M, K, N, (int)group_size);

    // Cast grad_palette fp32 → bf16 for autograd compatibility
    auto grad_palette = grad_palette_fp32.to(torch::kBFloat16);

    return {grad_logits, grad_palette};
}

// ═══════════════════════════════════════════════════════════════════════════
//  PATCH 7c — Batched compute_P_W host-side wrapper
//
//  Replaces 25 separate `fused_lut_linear_soft_fwd_aos` calls (one per
//  PalettizedLinear) with a SINGLE batched kernel launch that processes
//  all layers via blockIdx.z = layer_idx. See Patch 7a + 7b in
//  fused_lut_kernel.cu for the kernel + launcher.
//
//  The host-side launcher (fused_compute_P_W_batched_Launcher, defined in
//  the .cu file) takes raw host-side pointer arrays + per-layer shapes.
//  This wrapper accepts std::vector<torch::Tensor> (so Python lists work
//  naturally), allocates the per-layer P_aos + W outputs (managed by
//  PyTorch), and builds the host-side pointer arrays to pass to the
//  launcher.
//
//  Returns: flattened list of (P_aos_0, W_0, P_aos_1, W_1, ...) so the
//  caller can zip them into pairs.
// ═══════════════════════════════════════════════════════════════════════════

// Patch 7c — launcher prototype (defined in fused_lut_kernel.cu).
// Note: the PalettizedLayerDesc struct is NOT visible here — the launcher
// takes raw pointer arrays so the wrapper doesn't need to construct the
// struct itself.
void fused_compute_P_W_batched_Launcher(
    const c10::Half* const*     logits_ptrs,    // host array [n_layers]
    const c10::BFloat16* const* palette_ptrs,   // host array [n_layers]
    c10::Half* const*           P_aos_ptrs,     // host array [n_layers]
    c10::BFloat16* const*       W_out_ptrs,     // host array [n_layers]
    const int*                  Ks,             // host array [n_layers]
    const int*                  Ns,             // host array [n_layers]
    const int*                  Gs,             // host array [n_layers]
    int n_layers,
    int group_size,
    float tau,
    uint32_t step_seed);

std::vector<torch::Tensor> fused_compute_P_W_batched(
    std::vector<torch::Tensor> logits_list,    // each (4, K_l, N_l) fp16
    std::vector<torch::Tensor> palette_list,  // each (G_l, 4)    bf16
    int64_t group_size,
    double tau,
    int64_t step_seed
) {
    int n_layers = (int)logits_list.size();
    TORCH_CHECK(n_layers > 0, "logits_list must be non-empty");
    TORCH_CHECK((int)palette_list.size() == n_layers,
                "palette_list size must match logits_list size");
    TORCH_CHECK(n_layers <= 64, "max 64 layers in batched kernel (got ", n_layers, ")");

    // Validate inputs and allocate outputs in one pass.
    // We hold pointers in std::vector (heap-allocated — n_layers is dynamic).
    std::vector<const c10::Half*>     logits_ptrs(n_layers);
    std::vector<const c10::BFloat16*> palette_ptrs(n_layers);
    std::vector<c10::Half*>           P_aos_ptrs(n_layers);
    std::vector<c10::BFloat16*>       W_out_ptrs(n_layers);
    std::vector<int>                  Ks(n_layers);
    std::vector<int>                  Ns(n_layers);
    std::vector<int>                  Gs(n_layers);

    std::vector<torch::Tensor> P_aos_list;
    std::vector<torch::Tensor> W_list;
    P_aos_list.reserve(n_layers);
    W_list.reserve(n_layers);

    for (int i = 0; i < n_layers; ++i) {
        auto& logits  = logits_list[i];
        auto& palette = palette_list[i];

        TORCH_CHECK(logits.is_cuda() && logits.dtype() == torch::kHalf,
                    "logits[", i, "] must be fp16 cuda");
        TORCH_CHECK(palette.is_cuda() && palette.dtype() == torch::kBFloat16,
                    "palette[", i, "] must be bf16 cuda");
        TORCH_CHECK(logits.is_contiguous(),  "logits[", i, "] must be contiguous");
        TORCH_CHECK(palette.is_contiguous(), "palette[", i, "] must be contiguous");
        TORCH_CHECK(logits.dim() == 3 && logits.size(0) == 4,
                    "logits[", i, "] must be (4, K, N)");
        TORCH_CHECK(palette.dim() == 2 && palette.size(1) == 4,
                    "palette[", i, "] must be (G, 4)");

        int K = (int)logits.size(1);
        int N = (int)logits.size(2);
        int G = (int)palette.size(0);

        TORCH_CHECK(N % group_size == 0,
                    "layer ", i, ": N=", N, " not divisible by group_size=", group_size);
        TORCH_CHECK(N / group_size == G,
                    "layer ", i, ": G mismatch (N/GS=", N / group_size, " vs palette.size(0)=", G, ")");

        // Allocate outputs: P_aos (K, N, 4) fp16 + W (K, N) bf16
        auto P_aos = torch::empty({K, N, 4}, logits.options());
        auto W     = torch::empty({K, N}, palette.options());

        logits_ptrs[i]  = logits.data_ptr<c10::Half>();
        palette_ptrs[i] = palette.data_ptr<c10::BFloat16>();
        P_aos_ptrs[i]   = P_aos.data_ptr<c10::Half>();
        W_out_ptrs[i]   = W.data_ptr<c10::BFloat16>();
        Ks[i] = K;
        Ns[i] = N;
        Gs[i] = G;

        P_aos_list.push_back(std::move(P_aos));
        W_list.push_back(std::move(W));
    }

    // Single batched launch (host-side wrapper handles cudaMemcpyToSymbol
    // + grid.z = n_layers, see fused_lut_kernel.cu).
    fused_compute_P_W_batched_Launcher(
        logits_ptrs.data(),
        palette_ptrs.data(),
        P_aos_ptrs.data(),
        W_out_ptrs.data(),
        Ks.data(),
        Ns.data(),
        Gs.data(),
        n_layers,
        (int)group_size,
        (float)tau,
        (uint32_t)step_seed);

    // Return flattened (P_aos_0, W_0, P_aos_1, W_1, ...) — caller can zip
    // consecutive pairs.
    std::vector<torch::Tensor> result;
    result.reserve(2 * n_layers);
    for (int i = 0; i < n_layers; ++i) {
        result.push_back(std::move(P_aos_list[i]));
        result.push_back(std::move(W_list[i]));
    }
    return result;
}
"""


# ─────────────────────────────────────────────────────────────────────────────
#  Compile & cache via load_inline
#  (Launcher functions are defined in fused_lut_kernel.cu itself.)
# ─────────────────────────────────────────────────────────────────────────────
def _build_module():
    cu_source = _read_cu_source()

    # Compile for sm_89 (L4 / Ada) primary; add sm_80 (A100), sm_86 (Ampere-Ada compat),
    # and sm_90 (H100) for portability. PyTorch 2.9 ships with sm_86 binaries so we MUST
    # include sm_86 to link against torch's prebuilt objects.
    default_arch = "compute_120,code=sm_120;compute_89,code=sm_89;compute_86,code=sm_86;compute_80,code=sm_80;compute_90,code=sm_90"
    arch_str = os.environ.get("FUSED_LUT_CUDA_ARCH", default_arch)
    gencode_flags = []
    for entry in arch_str.split(";"):
        entry = entry.strip()
        if entry:
            gencode_flags.append(f"-gencode=arch={entry}")

    extra_cuda_cflags = [
        "-O3",
        "-std=c++17",
        "--use_fast_math",
        "-DCUDA_HAS_BF16",
        "-D__CUDA_NO_HALF_OPERATORS__",
        # NOTE: -Xptxas=-v is great for debugging register spilling but slows compile.
        # Keep it on for now since we're still tuning tile sizes.
        *gencode_flags,
    ]

    extra_cflags = ["-O3", "-std=c++17"]

    print("[fused_lut_linear_cuda] Compiling (may take ~30s on first run)...")
    print(f"  arches: {arch_str}")
    print(f"  cuda flags: {' '.join(extra_cuda_cflags)}")

    return load_inline(
        name="fused_lut_linear_cuda_ext",
        cpp_sources=[CPP_SOURCE],
        cuda_sources=[cu_source],
        functions=[
            "fused_lut_linear_fwd", "fused_lut_linear_bwd",
            "fused_lut_linear_soft_fwd", "fused_lut_linear_soft_bwd",
            "fused_lut_linear_soft_bwd_fused",
            # Patch 5: AoS variants — PalettizedLinear.forward/backward use these.
            "fused_lut_linear_soft_fwd_aos",
            "fused_lut_linear_soft_bwd_fused_aos",
            # Patch 7: batched compute_P_W — replaces 25 per-layer launches with 1.
            "fused_compute_P_W_batched",
        ],
        extra_cuda_cflags=extra_cuda_cflags,
        extra_cflags=extra_cflags,
        verbose=True,
    )


_module = None


def _get_module():
    global _module
    if _module is None:
        _module = _build_module()
    return _module


# ─────────────────────────────────────────────────────────────────────────────
#  Autograd Function
# ─────────────────────────────────────────────────────────────────────────────
GROUP_SIZE = 256


class CUDAFusedLUTLinear(torch.autograd.Function):
    """Autograd Function wrapping the CUDA kernels.

    forward(ctx, x, palette, indices, bias, group_size) -> y
    backward(ctx, grad_y) -> (grad_x, grad_palette, None, grad_bias, None)

    Gradients flow ONLY to `palette`. `indices` is a frozen int8 buffer.
    `bias` is also frozen (treated as buffer, not Parameter) — grad_bias is
    computed but only used externally if `bias.requires_grad` was set.
    """

    @staticmethod
    def forward(ctx, x, palette, indices, bias, group_size):
        # Validate / coerce contiguity
        x = x.contiguous()
        palette = palette.contiguous()
        indices = indices.to(torch.int8).contiguous()
        if bias is not None:
            bias = bias.contiguous()

        mod = _get_module()
        y = mod.fused_lut_linear_fwd(
            x, palette, indices, bias if bias is not None else None, group_size
        )

        ctx.save_for_backward(x, palette, indices)
        ctx.group_size = group_size
        ctx.has_bias = bias is not None
        ctx.bias_requires_grad = (bias is not None and bias.requires_grad)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, palette, indices = ctx.saved_tensors
        grad_y = grad_y.contiguous()

        needs_grad_x = ctx.needs_input_grad[0]
        needs_grad_palette = ctx.needs_input_grad[1]
        # grad_bias only needed if user declared bias.requires_grad=True
        needs_grad_bias = ctx.has_bias and ctx.bias_requires_grad

        mod = _get_module()
        # We pass `bias` as None to the bwd call when needs_grad_bias is False
        # so the C++ wrapper skips the grad_bias kernel.
        bias_arg = (
            torch.empty(0, dtype=torch.bfloat16, device=grad_y.device)
            if needs_grad_bias else None
        )
        grad_x, grad_palette, grad_bias = mod.fused_lut_linear_bwd(
            grad_y, x, palette, indices, bias_arg,
            ctx.group_size, needs_grad_x, needs_grad_palette
        )

        # grad_palette was cast to bf16 in C++ wrapper — pass through.
        # Cast back to None if not needed.
        if not needs_grad_x:
            grad_x = None
        if not needs_grad_palette:
            grad_palette = None
        if not needs_grad_bias:
            grad_bias = None

        # Return tuple must match forward input order: (x, palette, indices, bias, group_size)
        return grad_x, grad_palette, None, grad_bias, None


def fused_lut_linear(
    x: Tensor,
    palette: Tensor,
    indices: Tensor,
    bias: Tensor | None = None,
    group_size: int = GROUP_SIZE,
) -> Tensor:
    """Functional interface — supports autograd."""
    return CUDAFusedLUTLinear.apply(x, palette, indices, bias, group_size)


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE IX — Soft (Gumbel-Softmax) autograd Function + functional interface
# ─────────────────────────────────────────────────────────────────────────────

# Global step counter for Gumbel seed determinism (one increment per forward call)
_SOFT_STEP_SEED = 0


def _next_soft_step_seed() -> int:
    global _SOFT_STEP_SEED
    _SOFT_STEP_SEED = (_SOFT_STEP_SEED + 1) & 0xFFFFFFFF
    return _SOFT_STEP_SEED


class CUDAFusedLUTLinearSoft(torch.autograd.Function):
    """Soft forward with Gumbel-Softmax relaxation.

    Forward:
      - Calls `fused_lut_linear_soft_fwd` which:
          * Runs `compute_P_W_kernel`: samples Gumbel noise, computes softmax probs P,
            computes W = Σ_k P[k] * palette[g, k]
          * Runs `torch.matmul(x, W)` for the actual matmul (cuBLAS, tensor cores)
      - Saves: x (input), P (4, K, N) fp16, W (K, N) bf16, palette (G, 4) bf16

    Backward:
      - grad_x = grad_y @ W.T         (cuBLAS via torch.matmul)
      - grad_W = x.T @ grad_y          (cuBLAS via torch.matmul, fp32 accum)
      - grad_logits  = bwd_grad_logits_kernel (NEW CUDA kernel)
      - grad_palette = bwd_grad_palette_soft_kernel (NEW CUDA kernel, P-weighted)
      - grad_bias = grad_y.sum(dim=0)

    Args (forward):
      x: (M, K) bf16 — input activations
      palette: (G, 4) bf16 — LUT entries per group (TRAINABLE)
      logits: (4, K, N) fp16 — index logits (TRAINABLE)
      bias: (N,) bf16 or None
      group_size: int (256)
      tau: float — temperature for Gumbel-Softmax

    Returns (forward):
      y: (M, N) bf16

    Returns (backward, matching forward input order):
      (grad_x, grad_palette, grad_logits, grad_bias, None, None)
      — None for group_size and tau (non-tensor inputs)
    """

    @staticmethod
    def forward(ctx, x, palette, logits, bias, group_size, tau):
        # Validate / coerce
        x = x.contiguous()
        palette = palette.contiguous()
        logits = logits.contiguous()
        if bias is not None:
            bias = bias.contiguous()

        # Verify shapes
        M, K = x.shape
        G, P_size = palette.shape
        n_planes, K_, N = logits.shape
        assert P_size == 4, f"palette last dim must be 4, got {P_size}"
        assert n_planes == 4, f"logits first dim must be 4, got {n_planes}"
        assert K == K_, f"x K={K} != logits K={K_}"
        assert N % group_size == 0
        assert N // group_size == G, f"G mismatch: N//GS={N//group_size} vs palette.size(0)={G}"

        mod = _get_module()
        step_seed = _next_soft_step_seed()

        # Wave 1 optimization: C++ wrapper returns {P_aos, W_soft} — NO y_soft matmul.
        # The caller does ONE matmul: y = x @ W_ste (not two like before).
        P_aos, W_soft = mod.fused_lut_linear_soft_fwd_aos(
            x, palette, logits, group_size, float(tau), step_seed
        )

        # ── STE: Straight-Through Gumbel-Softmax ──────────────────────────
        # Forward uses HARD weight: W_hard = palette[argmax(logits)]
        # Backward flows through SOFT weight: W_soft (non-zero gradients)
        # W = W_hard - W_soft.detach() + W_soft
        #   forward value = W_hard (exact one-hot → cos preserved)
        #   backward grad  = through W_soft (indices actually train)
        with torch.no_grad():
            argmax_idx = logits.argmax(dim=0)  # (K, N) — hard index assignment
            # Gather: W_hard[k, n] = palette[n // group_size, argmax_idx[k, n]]
            group_idx = torch.arange(N, device=palette.device) // group_size
            group_per_col = group_idx.unsqueeze(0).expand(K, N)  # (K, N)
            W_hard = palette[group_per_col.long(), argmax_idx.long()].to(W_soft.dtype)  # (K, N) bf16
        # STE trick: forward = W_hard, backward = through W_soft
        W = W_hard - W_soft.detach() + W_soft
        # ONE matmul (was TWO: y_soft + y)
        y = torch.matmul(x, W)

        # Add bias if provided
        if bias is not None:
            y = y + bias

        # ctx saves P_aos as (K, N, 4) fp16 — Patch 5 layout.
        ctx.save_for_backward(x, palette, logits, P_aos, W)
        ctx.group_size = group_size
        ctx.tau = tau
        ctx.has_bias = bias is not None
        return y

    @staticmethod
    def backward(ctx, grad_y):
        # Patch 5c: ctx saved P as (K, N, 4) fp16 AoS — alias to P_aos for clarity.
        x, palette, logits, P_aos, W = ctx.saved_tensors
        grad_y = grad_y.contiguous()

        needs_grad_x = ctx.needs_input_grad[0]
        needs_grad_palette = ctx.needs_input_grad[1]
        needs_grad_logits = ctx.needs_input_grad[2]
        needs_grad_bias = ctx.has_bias and ctx.needs_input_grad[3]

        M, K = x.shape
        _, _, N = logits.shape
        G = palette.shape[0]
        GS = ctx.group_size

        # grad_x = grad_y @ W.T (cuBLAS — still the fastest way to do this matmul)
        grad_x = None
        if needs_grad_x:
            grad_x = torch.matmul(grad_y, W.T)

        # ── Patch 5c: re-enable the fused CUDA backward kernel (AoS P variant) ──
        #
        # Previously (PHASE IX.c), the fused CUDA bwd kernel was disabled because
        # its 4 strided loads to P[k * plane_size + idx] for k = 0..3 each hit
        # a separate L2 cache line 26 MB apart, making the kernel 3.8× slower
        # than the PyTorch elementwise path. The PyTorch path materialised a
        # (K, N, 4) fp32 intermediate via P.permute(1, 2, 0).float() — ~100 MB
        # of intermediate traffic per layer × 25 layers = ~2.5 GB / step.
        #
        # With the AoS layout (Patch 5a + 5b), P_aos[idx * 4 + 0..3] loads the
        # 4 fp16 values via a single coalesced 64-bit LDG — eliminating the
        # strided access entirely. The fused kernel is now faster than the
        # Python elementwise path AND avoids the (K, N, 4) fp32 intermediate.
        #
        # Path:
        #   1. grad_x   = grad_y @ W.T                (cuBLAS, computed above)
        #   2. grad_W (on-the-fly) + grad_logits + grad_palette
        #        = mod.fused_lut_linear_soft_bwd_fused_aos(grad_y, x, P_aos, palette, GS)
        grad_logits = None
        grad_palette = None
        if needs_grad_logits or needs_grad_palette:
            # ── Fused AoS CUDA backward kernel ──────────────────────────────
            # grad_W is computed INSIDE the kernel (on-the-fly per thread),
            # so we don't materialise the (K, N) bf16 grad_W on the Python side.
            mod = _get_module()
            grad_logits, grad_palette = mod.fused_lut_linear_soft_bwd_fused_aos(
                grad_y, x, P_aos, palette, GS
            )

            if not needs_grad_logits:
                grad_logits = None
            if not needs_grad_palette:
                grad_palette = None

        # grad_bias = grad_y.sum(dim=0)
        grad_bias = None
        if needs_grad_bias:
            grad_bias = grad_y.sum(dim=0)

        # Return tuple matches forward input order: (x, palette, logits, bias, group_size, tau)
        return grad_x, grad_palette, grad_logits, grad_bias, None, None


def fused_lut_linear_soft(
    x: Tensor,
    palette: Tensor,
    logits: Tensor,
    bias: Tensor | None = None,
    group_size: int = GROUP_SIZE,
    tau: float = 1.0,
) -> Tensor:
    """Soft (Gumbel-Softmax) functional interface — supports autograd.

    Args:
        x: (M, K) bf16
        palette: (G, 4) bf16 — trainable
        logits: (4, K, N) fp16 — trainable (Gumbel-Softmax input)
        bias: (N,) bf16 or None
        group_size: int (256)
        tau: float — temperature

    Returns:
        y: (M, N) bf16
    """
    return CUDAFusedLUTLinearSoft.apply(x, palette, logits, bias, group_size, tau)


# ─────────────────────────────────────────────────────────────────────────────
#  PATCH 7c — Python wrapper for batched compute_P_W
#
#  Computes P_aos and W_soft for ALL PalettizedLinear submodules of `student`
#  in a SINGLE kernel launch (blockIdx.z = layer_idx, see Patch 7a + 7b in
#  fused_lut_kernel.cu). Replaces the 25 separate compute_P_W launches that
#  happen when PalettizedLinear.forward is called per-layer.
#
#  PalettizedLinear.forward can OPTIONALLY use this batched path:
#    1. Pre-compute (P_aos, W) for all 25 layers via this function.
#    2. For each PalettizedLinear, look up its pre-computed W, apply STE
#       (W_hard - W_soft.detach() + W_soft), and call torch.matmul(x, W).
#    3. For backward, the autograd context saves the pre-computed P_aos.
#
#  Note: integrating this into PalettizedLinear.forward requires touching
#  qwen_model.py — that file is owned by agent-nn-module-foundation, so
#  the actual integration happens in their branch. This wrapper is the
#  API surface they will call.
# ─────────────────────────────────────────────────────────────────────────────
def fused_compute_P_W_batched(
    student,
    tau: float,
    step_seed: int,
    group_size: int | None = None,
):
    """Compute P_aos + W_soft for all PalettizedLinears in one kernel launch.

    Args:
        student: an nn.Module containing PalettizedLinear submodules. We walk
                 `student.named_modules()` and collect every PalettizedLinear
                 that has `index_logits is not None` and `_use_cuda == True`.
        tau: float — Gumbel-Softmax temperature (shared across all layers).
        step_seed: int — per-step Gumbel seed (must be incremented per forward
                   for noise diversity across steps; the batched kernel
                   decorrelates per-layer noise via XOR with the layer index).
        group_size: optional int — if None, uses each layer's own group_size
                    (all layers MUST share the same GS for the batched
                    kernel to work — we assert this).

    Returns:
        A list of `(P_aos, W)` tuples, one per PalettizedLinear (in the order
        returned by `student.named_modules()`). Each `P_aos` is `(K, N, 4)`
        fp16 AoS, each `W` is `(K, N)` bf16. Empty list if no PalettizedLinear
        with index_logits is found.
    """
    # Deferred import: qwen_model.py is owned by agent-nn-module-foundation.
    # If they rename PalettizedLinear, this will break — but that's their
    # responsibility to coordinate via inbox.
    from qwen_model import PalettizedLinear

    layers = [
        mod for _, mod in student.named_modules()
        if isinstance(mod, PalettizedLinear)
        and mod.index_logits is not None
        and getattr(mod, "_use_cuda", False)
    ]

    if not layers:
        return []

    logits_list  = [mod.index_logits for mod in layers]
    palette_list = [mod.palette      for mod in layers]

    # All layers must share the same group_size (kernel constraint).
    if group_size is None:
        group_size = layers[0].group_size
    for i, mod in enumerate(layers):
        if mod.group_size != group_size:
            raise ValueError(
                f"All PalettizedLinears must share group_size for the "
                f"batched compute_P_W kernel; layer {i} has "
                f"group_size={mod.group_size} vs expected {group_size}"
            )

    mod_ext = _get_module()
    flat = mod_ext.fused_compute_P_W_batched(
        logits_list, palette_list, int(group_size),
        float(tau), int(step_seed)
    )
    # flat is [P_aos_0, W_0, P_aos_1, W_1, ...] — zip consecutive pairs.
    return [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
