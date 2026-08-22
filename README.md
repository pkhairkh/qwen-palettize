# qwen-palettize

2-bit LUT (lookup table) quantization of Qwen3.5-4B with trainable indices via Gumbel-Softmax + LoRA compensation.

## Current Status

- **Cos achieved:** 0.9530 (training, target: >0.999)
- **Hardware:** NVIDIA RTX PRO 6000 Blackwell (sm_120, 96GB VRAM, 600W TDP)
- **GPU utilization:** 99% (417W power draw)
- **Throughput:** 0.8 steps/sec at batch=32, seq=512 (25.6K tokens/sec)

## Architecture

- **2-bit palettization** (4 values per group of 256 weights)
- **Trainable indices** via Gumbel-Softmax + STE (Straight-Through Estimator)
- **Trainable palettes** via AdamW (bf16 params, fp32 master)
- **LoRA rank-16** on all Linears (rank-32 on 5 worst-cosine Linears)
- **Distillation loss:** 1-cos + norm_mse (equal weights)
- **CUDA kernels:** hard forward (TC mma.sync.m16n8k16) + soft forward (Gumbel-Softmax)
- **Optimizer:** Muon (layernorms) + FP32MasterAdamW (palettes, LoRA, indices)

## Repository Structure

```
qwen-palettize/
├── SPEC.md                          # Project specification
├── scripts/                         # Training + CUDA kernel source code
│   ├── train_qwen.py                # Training loop
│   ├── qwen_model.py                # PalettizedLinear + QwenLoRA
│   ├── fused_lut_kernel.cu          # CUDA kernels (forward + backward)
│   ├── fused_lut_linear_cuda.py     # Python autograd wrappers
│   ├── palettize_core.py            # 2-bit calibration (weighted k-means)
│   └── ...
├── palettized/superblock_0/         # Initial 2-bit calibration data
├── trained/superblock_0_best/       # Best trained checkpoint (cos=0.9464, step=8000)
├── logs/                            # Training + calibration logs
├── cached_tokens.pt                 # Cached training tokens
├── eval_tokens.pt                   # Held-out eval set (256 seqs × 512 tokens)
└── research-*/                      # Research agent output (5 agents, 568 pages total)
```

## Research Reports

Six research agents investigated specific deficiencies. All reports are in their respective folders:

| Agent | Folder | Pages | Focus |
|-------|--------|-------|-------|
| Kernel Accuracy | `research-kernel-accuracy/` | 88 | Numerical precision, STE correctness, convergence analysis |
| Kernel Efficiency | `research-kernel-efficiency/` | 107 | 530ms step decomposition, fused backward fix, sm_120 optimization |
| Indices Training | `research-indices-training/` | 96 | Gumbel-Softmax audit, gradient flow, tau schedule, optimizer analysis |
| Palettes Training | `research-palettes-training/` | 115 | Palette gradient correctness, precision, k-means vs gradient, loss function |
| Architecture Review | `research-architecture-review/` | 162 | Systemic architecture issues, PartialWrapper, memory waste, refactoring roadmap |
| **Literature Review** | **`research-literature-review/`** | **75** | **Comparative survey of 20 SOTA LLM quantization methods (GPTQ, AWQ, SqueezeLLM, AQLM, QuIP#, GPTVQ, BitNet, QLoRA, llama.cpp, ExLlamaV2, etc.) × 10 dimensions; gap analysis vs our approach; prioritized recommendations** |

**Note:** Literature Review (`research-literature-review/`) was completed last, covering 20 SOTA methods across 10 dimensions. See `research-literature-review/00_executive_summary.md` for the gap analysis and `research-literature-review/08_recommendations.md` for the prioritized implementation plan.

## Quick Start

```bash
# Resume training from checkpoint
cd scripts
USE_TC_FWD=1 USE_TC_BWD_GX=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 train_qwen.py --max_steps 12000 --batch_size 32 --seq_len 512 \
  --tau_init 2.0 --tau_final 0.1 --tau_anneal_steps 4000 \
  --resume_from ../trained/superblock_0_best
```

## Live Hyperparameter Tuning

Edit `/tmp/hyperparams_qwen.json` during training to adjust LRs, loss weights, and freeze/unfreeze groups:

```json
{
  "lrs": {"palettes": 3e-3, "lora": 1e-3, "indices": 1e-2, "layernorms": 3e-4},
  "groups": {"palettes": true, "lora": true, "indices": true, "layernorms": true},
  "loss_type": "1-cos+norm_mse",
  "loss_weights": {"cos": 0.5, "mse": 0.5}
}
```

## Key Findings from Research

1. **STE is correct** but ineffective at low tau — gradients vanish when P is one-hot
2. **Fused CUDA backward kernel** exists but is 10x slower (strided P access) — needs coalesced rewrite
3. **cos plateau at 0.95** is a representation problem — missing 6 SOTA techniques from literature (GPTQ, AWQ, SqueezeLLM, etc.)
4. **PartialWrapper** breaks torch.compile, gradient checkpointing, and framework integration
5. **VRAM utilization** is only 37% (36/96GB) — batch=64 OOMs due to Python intermediate tensors

See `research-*/00_overview.md` or `research-*/0*_executive_summary.md` in each folder for details.
