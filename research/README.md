# Research Reports Index

This directory contains the output of 5 research agents that investigated deficiencies in the qwen-palettize training system. Total: **568 pages** of analysis across 5 folders.

## Agent Status

| # | Agent | Folder | Pages | Status | Key Finding |
|---|-------|--------|-------|--------|-------------|
| 1 | Kernel Accuracy | [`../research-kernel-accuracy/`](../research-kernel-accuracy/) | 88 | ✅ Complete | STE is mathematically correct but ineffective at low tau |
| 2 | Kernel Efficiency | [`../research-kernel-efficiency/`](../research-kernel-efficiency/) | 107 | ✅ Complete | Fused backward kernel 10x slower (strided P access); 530ms step breakdown |
| 3 | Indices Training | [`../research-indices-training/`](../research-indices-training/) | 96 | ✅ Complete | Gumbel-Softmax gradients vanish at logits=±10; need softening + STE |
| 4 | Palettes Training | [`../research-palettes-training/`](../research-palettes-training/) | 115 | ✅ Complete | bf16 sufficient for 2,208 params; k-means re-quantization (LUT-Q) may outperform gradient descent |
| 5 | Architecture Review | [`../research-architecture-review/`](../research-architecture-review/) | 162 | ✅ Complete | PartialWrapper breaks everything; no gradient checkpointing; monolithic 1200-line training loop |
| 6 | Literature Review | — | 0 | ❌ Missing | Agent was never started — needs to be launched separately |

## File Organization

Each agent's output follows this naming convention:
- `00_overview.md` or `00_executive_summary.md` — TL;DR + key findings
- `01_*.md` through `09_*.md` — detailed analysis (wave-by-wave)
- Last numbered file — references/bibliography

## How to Read

1. Start with each agent's `00_*.md` for executive summary
2. Read `*_recommendations.md` for actionable code patches
3. Read `*_references.md` for arxiv papers and GitHub repos

## Top 5 Cross-Cutting Findings

1. **STE + soft logits is the winning combination** — forward=hard (cos preserved), backward=soft (gradients flow). See `research-indices-training/02_ste_correctness.md`.

2. **Fused backward kernel needs coalesced P access** — current strided (4,K,N) layout causes 10x slowdown. See `research-kernel-efficiency/02_fused_bwd_fix.md`.

3. **cos plateau at 0.95 is a representation problem** — not a kernel-numerics problem. Missing 6 SOTA techniques from literature. See `research-kernel-accuracy/05_convergence_analysis.md`.

4. **PartialWrapper must be replaced with nn.Module** — breaks torch.compile, gradient checkpointing, FSDP. See `research-architecture-review/02_partial_wrapper_problem.md`.

5. **100-step warmup prevents STE divergence** — without warmup, STE + full LR causes index flips that destroy the model. See training log in `logs/train_sb0.log`.
