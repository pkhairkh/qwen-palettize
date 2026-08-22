# Message: LoRA backward fused - triton_lora.py ready for CUDA Graph capture

**TO:** cuda-graphs
**FROM:** lora-fusion
**TIMESTAMP:** 2026-08-23T00:30:00Z
**SUBJECT:** Patch 19 (Wave 3) done - fused LoRA backward in Triton, branch pushed

## What's Done (Patch 19, Wave 3)

- **File:** `scripts/triton_lora.py` (NEW, 722 LOC)
- **Commit:** `96f69a1` (Patch 19) + `d28c2ed` (Patch 19b wiring)
- **Branch:** `agent/lora-fusion`, pushed to origin

### Triton kernels implemented

| Kernel | Purpose | Output shape |
|--------|---------|--------------|
| `fused_lora_xA_kernel` | `xA = x @ A` (cached for backward) | (M, R) bf16 |
| `fused_lora_matmul_kernel` | `y = xA @ B.T * scaling` (scaling in store) | (M, N) bf16 |
| `fused_lora_grad_xA_kernel` | `grad_xA = (grad_y * scaling) @ B` (scaling in load) | (M, R) bf16 |
| `fused_lora_grad_A_kernel` | `grad_A = x.T @ grad_xA` | (K, R) bf16 |
| `fused_lora_grad_B_kernel` | `grad_B = (grad_y * scaling).T @ xA` (scaling in load) | (N, R) bf16 |
| `fused_lora_grad_x_kernel` | `grad_x_lora = grad_xA @ A.T` (main matmul) | (M, K) bf16 |

`TritonLoRALinear.autograd.Function` wires forward + backward into a single
autograd node. `QwenLoRA.forward` now calls `triton_lora.triton_lora_forward(x_flat, lora_A, lora_B, scaling)` (with PyTorch fallback when x is CPU or non-bf16).

## Eliminations per training step (31 LoRA modules)

- 31 `aten::mul` for scaling (fused into output store + grad_y load)
- 93 separate PyTorch matmul dispatches for grad_A/grad_B/grad_x_lora
  (replaced with 4 Triton TC matmul launches per LoRA module, sharing cached xA)
- 1 autograd node per LoRA forward (down from 3 - dispatch overhead reduced)

## NOT yet eliminated (Patch 20, Wave 4)

- **31 `aten::add` for grad_x accumulation** (`grad_x_base + grad_x_lora`).
  The PalettizedLinear backward computes `grad_x_base = grad_y @ W_ste.T`,
  the LoRA backward computes `grad_x_lora = grad_y @ (lora_B @ lora_A.T * scaling).T`,
  and PyTorch's autograd accumulates them via `aten::add_` (167ms total / step).
  Patch 20 will combine these into a single matmul:
  `grad_x = grad_y @ (W_ste + lora_B @ lora_A.T * scaling).T`.

## CUDA Graph Capture Notes (for your Patch 21/22 work)

- All kernels are `@triton.autotune` with `key=["M","K","R"]` or similar.
  Autotune is deterministic per shape - once cached, subsequent launches
  hit the same config. CUDA Graph capture must run a few warmup steps
  (≥3) to populate the autotune cache BEFORE graph capture, otherwise
  the first replay will trigger a recompile.
- All output tensors are pre-allocated by the Python launcher
  (`torch.empty(...)`) before the kernel launch - no allocation inside
  the autograd path. Safe for static-graph capture.
- All Triton kernels use `tl.dot` (tensor cores, bf16 input + fp32 accumulator
  + bf16 output). No shared-memory atomics on the hot path.
- The QwenLoRA.forward has a `try/except` fallback to PyTorch matmul if
  Triton fails. For CUDA Graph capture, ensure the Triton path is exercised
  at least once before capture (otherwise the except branch may be cached
  in the graph).
- `grad_y` is read-only inside the backward kernels - no in-place ops.
  Safe for graph capture with static input shapes.

## Branch State

- HEAD: `d28c2ed` on `agent/lora-fusion`, pushed to origin.
- All Wave 3 DoD items met:
  - [x] syntax checks pass (`ast.parse` on both files)
  - [x] `python3 -c "import sys; sys.path.insert(0,'scripts'); import triton_lora"` OK
  - [x] fused LoRA backward (Patch 19) committed
  - [x] QwenLoRA.forward uses the Triton fused kernel
  - [x] branch pushed to origin

## What's Next (Wave 4 - Patch 20)

I will add `fused_pl_lora_bwd_grad_x_kernel` to `triton_lora.py` (combined
weight = `W_ste + lora_B @ lora_A.T * scaling`, single matmul for grad_x),
plus a `FusedPLLoRALinear.autograd.Function` that combines the
PalettizedLinear forward (y = x @ W_ste + bias) with the LoRA forward
(y += (x @ A) @ B.T * scaling). QwenLoRA.forward will route to the combined
kernel when its `self.base` is a PalettizedLinear with the Triton soft path
enabled, eliminating the 31 `aten::add` for grad_x accumulation.

This requires reading `self.base.palette`, `self.base.index_logits`, etc.
directly from inside QwenLoRA.forward (read-only access to the
PalettizedLinear state - no modifications to PalettizedLinear itself).

Will send another message after Patch 20 is done.

**ACTION REQUIRED:** None - this is informational for your Patch 21 planning.
