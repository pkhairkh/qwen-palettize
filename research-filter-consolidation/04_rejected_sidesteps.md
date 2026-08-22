# 04 — Rejected Sidesteps Register

> **Wave 3 deliverable.** Documents every recommendation that was REJECTED because it replaces our current approach. Grouped by category. Future agents should NOT re-suggest these.

**Our approach (DO NOT REPLACE):**
- 2-bit per-group palettization (GS=256, 4 LUT values per group)
- 1d-kmeans calibration (BEST non-GPTQ — already tested)
- Gumbel-Softmax trainable indices + STE
- Trainable palettes via AdamW (gradient descent, NOT k-means re-quantization)
- LoRA rank-16/32
- PartialWrapper (converting to nn.Module IS a speed enhancement — see Patch 9, NOT rejected)

---

## Category 1: GPTQ Alternatives (10 rejections)

These replace our 1d-kmeans calibration with GPTQ Hessian-based closed-form quantization.

| # | Source | Recommendation | Why Rejected |
|---|--------|---------------|--------------|
| 25 | kernel-accuracy/04_literature | Add GPTQ-style second-order calibration | Replaces 1d-kmeans with GPTQ Hessian inverse. Our 1d-kmeans is already tested as the BEST non-GPTQ approach. GPTQ is a different calibration method entirely. |
| 37 | kernel-efficiency/00_overview | Add GPTQ-style second-order calibration | Duplicate of #25. |
| 114 | palettes-training/00_overview | Add GPTQ-style second-order calibration | Duplicate of #25. |
| 130 | palettes-training/04_kmeans_vs_gradient | Try GPTQ with fixed k-means LUT | Adds GPTQ column-by-column update. Even with fixed LUT, this is the GPTQ algorithm — a sidestep. |
| 176 | literature-review/00_executive_summary | GPTQ Hessian-inverse error propagation | Duplicate of #25. |
| 184 | literature-review/01_gptq_family | Activation-ordered group processing (desc_act) | GPTQ-specific heuristic. Only makes sense with GPTQ (rejected). |
| 43 | kernel-efficiency/00_overview | AWQ-style activation-aware scaling | AWQ is a different calibration method. Replaces 1d-kmeans. |
| 27 | kernel-accuracy/04_literature | AWQ-style activation-aware scaling | Duplicate of #43. |
| 173 | literature-review/00_executive_summary | AWQ per-channel activation-aware scale | Duplicate of #43. |
| 186 | literature-review/02_awq_smoothquant | SmoothQuant per-channel activation smoothing | Adds SmoothQuant scaling. Calibration approach change. |

**Rationale:** GPTQ and AWQ are alternative quantization methods. We use 1d-kmeans (already tested as best without GPTQ). Switching to GPTQ/AWQ would be a fundamental calibration change.

---

## Category 2: LLT / k-means Re-quantization Alternatives (8 rejections)

These replace Gumbel-Softmax with LLT deterministic softmax, or replace gradient-trained palettes with LUT-Q k-means re-quantization.

| # | Source | Recommendation | Why Rejected |
|---|--------|---------------|--------------|
| 82 | indices-training/00_overview | Switch to deterministic-ST (remove Gumbel noise) | Replaces Gumbel-Softmax with LLT deterministic softmax. Our approach uses Gumbel noise (Jang et al. 2017). |
| 178 | literature-review/00_executive_summary | Drop Gumbel noise (deterministic softmax) | Duplicate of #82. |
| 86 | indices-training/00_overview | Switch to LUT-Q pattern (FP shadow + k-means reassignment) | Replaces Gumbel-Softmax index training with LUT-Q k-means reassignment. Fundamental approach change. |
| 182 | literature-review/00_executive_summary | LUT-Q FP shadow + k-means reassignment | Duplicate of #86. |
| 128 | palettes-training/04_kmeans_vs_gradient | Hybrid GD + periodic k-means re-quantization | Replaces gradient-trained palettes with periodic k-means. LUT-Q pattern. |
| 5 | kernel-accuracy/00_overview | Implement LUT-Q-style re-quantization every 2000 steps | Duplicate of #128. |
| 92 | indices-training/02_ste_correctness | A/B test vanilla STE (Bengio 2013) | Replaces Gumbel-ST with vanilla STE (no softmax relaxation). Different gradient routing. |
| 96 | indices-training/03_gradient_flow_analysis | Direct P parameterization (drop softmax) | Replaces Gumbel-Softmax entirely with direct probability parameterization. |

**Rationale:** Our approach uses Gumbel-Softmax + STE (state of the art for differentiable quantization). LLT/LUT-Q/vanilla-STE are alternative approaches that replace Gumbel-Softmax. We enhance Gumbel-Softmax (e.g., τ schedule, logit clamp), not replace it.

---

## Category 3: VQ / Codebook Architecture Changes (10 rejections)

These replace our scalar 2-bit LUT (4 values per group) with vector quantization, additive codebooks, or lattice codebooks.

| # | Source | Recommendation | Why Rejected |
|---|--------|---------------|--------------|
| 29 | kernel-accuracy/04_literature | Switch to 8-dim VQ codebook (QuIP#-style) | Replaces scalar LUT with 8-dim vector quantization. Fundamental representation change. |
| 30 | kernel-accuracy/04_literature | Use E8 lattice codebook (QuIP#) | Replaces scalar LUT with E8 lattice VQ. |
| 31 | kernel-accuracy/04_literature | Switch to additive VQ (AQLM 2×256) | Replaces scalar LUT with additive multi-codebook. |
| 36 | kernel-efficiency/00_overview | Switch to vector quantization (VQ) | Duplicate of #29. |
| 117 | palettes-training/00_overview | Switch to 8-dim VQ codebook | Duplicate of #29. |
| 119 | palettes-training/00_overview | Rewrite CUDA kernels for VQ | Depends on #117 (VQ). |
| 172 | literature-review/00_executive_summary | GPTVQ vector quantization (g=2) | Replaces scalar LUT with 2D VQ. |
| 189 | literature-review/03_codebook_methods | AQLM additive multi-codebook (M=2) | Duplicate of #31. |
| 190 | literature-review/03_codebook_methods | AQLM beam search index assignment | Replaces Gumbel-Softmax with beam search. |
| 24 | kernel-accuracy/03_ste_analysis | Switch to AQLM-style direct STE (no softmax) | Replaces Gumbel-Softmax with AQLM pattern. |

**Rationale:** Our approach uses scalar 2-bit LUT (4 values per group of 256). VQ/additive/lattice codebooks are fundamentally different representations. They may achieve higher cos, but they require rewriting all kernels, calibration, and training — a complete architecture change.

---

## Category 4: Architecture Rewrites (15 rejections)

These restructure the codebase into a modular package, add config systems, or adopt framework patterns.

| # | Source | Recommendation | Why Rejected |
|---|--------|---------------|--------------|
| 142 | architecture-review/05_training_loop_refactor | Refactor monolithic train_super_block into Trainer class | Modular package rewrite. Out of scope for enhancement. |
| 157 | architecture-review/05_training_loop_refactor | Config dataclass as single source of truth | Config system rewrite. Out of scope. |
| 169 | architecture-review/08_literature_comparison | Adopt HF Trainer pattern | Framework adoption. Out of scope. |
| 158 | architecture-review/06_logging_metrics | MetricsLogger with wandb/tensorboard backends | Tooling integration. Out of scope. |
| 159 | architecture-review/06_logging_metrics | Log artifacts and system metrics | Tooling integration. Out of scope. |
| 160 | architecture-review/06_logging_metrics | Rewrite sweep_qwen.py for CSV | Tooling. Out of scope. |
| 161 | architecture-review/07_testing_ci | 4-tier test scaffold | Testing infrastructure. Out of scope. |
| 162 | architecture-review/07_testing_ci | GitHub Actions CI/CD + GPU runner | CI/CD. Out of scope. |
| 163 | architecture-review/07_testing_ci | pytest-cov coverage | Testing. Out of scope. |
| 35 | kernel-accuracy/05_convergence_analysis | Per-block end-to-end joint training | Restructures training (AQLM approach). |
| 120 | palettes-training/00_overview | Per-block end-to-end training | Duplicate of #35. |
| 55-61 | kernel-efficiency/04_sm120_optimal | TMA / wgmma / tcgen05 / cluster.sync / setmaxnreg / CUTLASS | Radical kernel rewrites. Keep mma.sync.m16n8k16. |
| 67 | kernel-efficiency/05_memory_optimization | Adafactor (no second moment) | Replaces AdamW with Adafactor. Optimizer change. |
| 106 | indices-training/06_optimizer_analysis | Hybrid optimizer schedule (SGD→AdamW) | Phase-switch optimizer. Partially replaces AdamW. |
| 107 | indices-training/06_optimizer_analysis | Switch betas to (0.9, 0.999) with deterministic-ST | Depends on #82 (deterministic-ST, rejected). |

**Rationale:** These are architecture/tooling changes, not approach enhancements. They would require significant engineering effort without improving cos. The PartialWrapper→nn.Module conversion (Patch 9) IS kept because it's a speed enhancement that unlocks torch.compile.

---

## Category 5: Calibration Approach Changes (12 rejections)

These add preprocessing steps (RHT, outlier isolation, learnable clipping) that change the calibration pipeline.

| # | Source | Recommendation | Why Rejected |
|---|--------|---------------|--------------|
| 28 | kernel-accuracy/04_literature | RHT incoherence preprocessing (QuIP-style) | Adds QuIP preprocessing. Calibration change. |
| 44 | kernel-efficiency/00_overview | RHT incoherence preprocessing | Duplicate of #28. |
| 116 | palettes-training/00_overview | Add RHT incoherence preprocessing | Duplicate of #28. |
| 175 | literature-review/00_executive_summary | QuIP# Hadamard pre-rotation | Duplicate of #28. |
| 26 | kernel-accuracy/04_literature | Outlier isolation (top 0.5% in FP16) | Adds SqueezeLLM dense-sparse decomposition. Architecture change. |
| 38 | kernel-efficiency/00_overview | Outlier isolation (top 0.5% in FP16) | Duplicate of #26. |
| 115 | palettes-training/00_overview | Add outlier isolation | Duplicate of #26. |
| 174 | literature-review/00_executive_summary | SqueezeLLM dense/sparse split | Duplicate of #26. |
| 99 | indices-training/05_literature_comparison | Dense-and-sparse decomposition (SqueezeLLM) | Duplicate of #26. |
| 181 | literature-review/00_executive_summary | Learnable clipping threshold (OmniQuant) | Adds OmniQuant clipping. Calibration change. |
| 187 | literature-review/02_awq_smoothquant | OmniQuant LET (learnable s and t) | Adds OmniQuant transform. Calibration change. |
| 188 | literature-review/02_awq_smoothquant | AffineQuant lower-triangular transform | Adds AffineQuant. Calibration change. |

**Rationale:** These add preprocessing or postprocessing steps from other quantization methods (QuIP, SqueezeLLM, OmniQuant, AffineQuant). They change the calibration pipeline, which is currently 1d-kmeans. We enhance k-means (e.g., group size tuning), not replace it with multi-step pipelines.

---

## Category 6: Other Sidesteps (10 rejections)

These change the training recipe in ways that replace core components.

| # | Source | Recommendation | Why Rejected |
|---|--------|---------------|--------------|
| 102 | indices-training/05_literature_comparison | From-scratch training (BitNet pattern) | Replaces k-means init with random init. Different training paradigm. |
| 195 | literature-review/04_1bit_methods | From-scratch training (no k-means init) | Duplicate of #102. |
| 180 | literature-review/00_executive_summary | Lloyd-Max Gaussian codebook init | **Already tested as BULLSHIT.** Hessian-weighted Lloyd-Max was tested and is inferior to 1d-kmeans. |
| 85 | indices-training/00_overview | Hessian-weighted gradient (SqueezeLLM) | Adds Hessian weighting. Calibration approach change. |
| 100 | indices-training/05_literature_comparison | Activation-weighted gradient (AWQ) | Duplicate of #85. |
| 101 | indices-training/05_literature_comparison | AWQ-style scaled grid for codebook init | Replaces k-means init with AWQ scaling. |
| 103 | indices-training/05_literature_comparison | Co-train per-group scale parameter (BitNet/AWQ) | Adds scale parameter. Architecture change. |
| 193 | literature-review/04_1bit_methods | Per-tensor learnable α scale (BitNet) | Adds scale parameter. Architecture change. |
| 194 | literature-review/04_1bit_methods | Initialize one palette level at 0 (sparse) | Changes palette structure. Approach change. |
| 197 | literature-review/04_1bit_methods | SubLN (learnable gain before LayerNorm) | Adds SubLN. Architecture change. |
| 198 | literature-review/05_production_frameworks | Per-sub-group scale hierarchy (llama.cpp Q2_K) | Adds sub-group scales. Architecture change. |
| 183 | literature-review/00_executive_summary | ExLlamaV2 per-layer mixed-precision bit allocation | Mixed-precision. Architecture change. |

**Rationale:** These add learnable scale parameters, change palette structure, or replace k-means init. They are fundamental approach changes, not enhancements.

---

## Summary

| Category | Rejections | Key Pattern |
|----------|-----------|-------------|
| GPTQ alternatives | 10 | Replace 1d-kmeans with Hessian-based calibration |
| LLT/k-means alternatives | 8 | Replace Gumbel-Softmax with deterministic softmax or LUT-Q |
| VQ/codebook changes | 10 | Replace scalar LUT with vector/additive/lattice codebook |
| Architecture rewrites | 15 | Modular package, config system, framework adoption, kernel rewrite |
| Calibration changes | 12 | Add RHT/outlier/OmniQuant/AffineQuant preprocessing |
| Other sidesteps | 10 | From-scratch, scale params, palette structure changes |
| **Total** | **65** | |

**Note:** Some recommendations appear in multiple agents (duplicates). The 65 unique rejections cover all distinct sidesteps found across 568 pages of research.
