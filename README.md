# Qwen3.5-4B 2-bit Palettization

2-bit per-group (GS=256) scalar palettization of Qwen3.5-4B with **trainable indices** via Gumbel-Softmax + LoRA rank-16 on all palettized Linears.

## Architecture

- **2-bit LUT quantization** with group size 256 (4 palette entries per 256 weights)
- **Trainable indices** via Gumbel-Softmax relaxation (soft forward, temperature annealing)
- **LoRA rank-16** on all palettized Linears (compensates for residual quantization error)
- **No dense correction layer** — indices + LoRA do the same job with 10x fewer params
- **CUDA kernel** with Tensor Cores for both hard (inference) and soft (training) forward/backward

## Structure

```
scripts/           — Training + calibration scripts
  qwen_model.py     — PalettizedLinear + QwenLoRA + model loading
  train_qwen.py     — Training loop (Muon + AdamW + SGD, fp32 masters)
  calib_qwen.py     — Stage 1 calibration (GPTQ + kmeans1d)
  palettize_core.py — 2-bit packing + kmeans1d
  fused_lut_kernel.cu        — CUDA kernel (hard + soft, Tensor Cores)
  fused_lut_linear_cuda.py   — Python wrapper + autograd Functions

lut_kernel/        — CUDA kernel development + tests
  test_correctness.py — 53 numerical tests
  benchmark.py        — Performance benchmarks
  reference_pytorch.py — PyTorch reference (correctness oracle)
  triton_lut_linear.py — Triton reference kernel
  train_step.py       — End-to-end training verification

docs/              — Documentation
  SPEC.md            — Full project specification
  trainable_indices_kb.md — Literature review on trainable indices
  gumbel_softmax_kernel_work_instruction.md — Soft kernel spec
```

## Quick Start

```bash
# Stage 1: Calibrate (produces 2-bit indices + initial palette)
python3 scripts/calib_qwen.py --sb_idx 0 --n_seqs 8192 --seq_len 2048

# Stage 2: Train (soft indices + LoRA + temperature annealing)
USE_TC_FWD=1 USE_TC_BWD_GX=1 \
python3 scripts/train_qwen.py \
  --sb_idx 0 --max_steps 5000 \
  --seq_len 128 --batch_size 4 \
  --lora_rank 16 --lora_alpha 32 \
  --use_soft_indices 1 \
  --tau_init 1.0 --tau_final 0.01 --tau_anneal_steps 4000
```

## CUDA Kernel

The fused LUT-quantized linear layer kernel supports two modes:

- **Hard** (eval/inference): frozen int8 indices -> palette gather -> TC matmul
- **Soft** (training): Gumbel-Softmax logits -> probability-weighted W -> TC matmul

Both use Tensor Cores (mma.sync m16n8k16) via ldmatrix for the matmul portion.

### Tests

```bash
cd lut_kernel && python3 test_correctness.py  # 53/53 tests pass
```

### Benchmark (L4, sm_89)

| Shape (KxN) | REF | CUDA (hard) | cuBLAS floor |
|---|---|---|---|
| 2560x2560 | 6.5ms | 2.9ms | 1.7ms |
| 2560x8192 | 17.7ms | 9.1ms | 5.2ms |
| 9216x2560 | 23.1ms | 10.4ms | 5.9ms |

## GPU Requirements

- **L4 (24GB)**: Works with batch_size=4 + plain SGD for indices (tight memory)
- **RTX 6000 Ada (48GB)**: Works with batch_size=8 + AdamW for indices (recommended)
- **A100 (80GB)**: Full headroom

Compile target: sm_89 (primary) + sm_80 + sm_90 (portability)
