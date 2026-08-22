# 00 — Audit Table: All Recommendations from 6 Research Agents

> **Wave 1 deliverable.** Classification of every distinct recommendation found across all 6 research folders (568 pages total). Each row classified as **KEEP** (ANE-aligned enhancement) or **REJECT** (sidestep that replaces our approach).

**Our approach (the baseline to enhance — DO NOT REPLACE):**
- 2-bit per-group palettization (GS=256, 4 LUT values per group)
- Calibration: 1d-kmeans weighted by activation norm (BEST non-GPTQ approach — already tested)
- Gumbel-Softmax trainable indices + STE (forward=hard, backward=soft)
- Trainable palettes via AdamW (bf16 params, fp32 master)
- LoRA rank-16 (rank-32 on 5 worst-cosine Linears)
- Distillation loss: 1-cos + norm_mse
- Muon (layernorms) + FP32MasterAdamW (palettes, LoRA, indices)
- PartialWrapper (NOT nn.Module currently — but converting IS a speed enhancement, see #39)

**Filter criteria:**
- **KEEP** = enhances current approach without replacing any core component
- **REJECT** = replaces a core component (Gumbel→LLT, k-means→GPTQ, gradient-palettes→LUT-Q re-quantization, scalar→VQ, etc.)

---

## Audit Table (120 recommendations)

| # | Agent | File | Recommendation | Verdict | Rationale |
|---|-------|------|----------------|---------|-----------|
| 1 | kernel-accuracy | 00_overview | LoftQ SVD init for LoRA (capture original weights) | **KEEP** | Enhances LoRA init, preserves Gumbel+LUT approach |
| 2 | kernel-accuracy | 00_overview | Switch loss to 1-cos+norm_mse with cos=0.8, mse=0.2 | **KEEP** | Already our loss type; just tune weights |
| 3 | kernel-accuracy | 00_overview | Promote palette to fp32 (2,208 params, negligible memory) | **KEEP** | Precision enhancement, no approach change |
| 4 | kernel-accuracy | 00_overview | Per-group gradient clipping instead of global | **KEEP** | Optimizer enhancement, no approach change |
| 5 | kernel-accuracy | 00_overview | LUT-Q-style re-quantization every 2000 steps | **REJECT** | Replaces gradient-trained palettes with k-means re-quantization |
| 6 | kernel-accuracy | 00_overview | Per-tensor GROUP_SIZE override for worst Linears | **KEEP** | Enhances calibration, preserves approach |
| 7 | kernel-accuracy | 00_overview | Revive freeze_settled_palettes (dead code) | **KEEP** | Already in our codebase, just needs to be called |
| 8 | kernel-accuracy | 01_kernel_audit | Use 0xFF OOB sentinel + guard for indices | **KEEP** | Bug fix, no approach change |
| 9 | kernel-accuracy | 01_kernel_audit | Add NaN/inf guard on hard forward output | **KEEP** | Robustness fix, no approach change |
| 10 | kernel-accuracy | 01_kernel_audit | Prefer scalar fp32 FMA over TC mma.sync | **REJECT** | Disables TC — we WANT TC for speed (USE_TC_FWD=1) |
| 11 | kernel-accuracy | 01_kernel_audit | Pad indices tensor to multiple of 4 in N | **KEEP** | Bug fix, no approach change |
| 12 | kernel-accuracy | 01_kernel_audit | Warp-shuffle reduction for grad_bias | **KEEP** | Kernel speed enhancement |
| 13 | kernel-accuracy | 01_kernel_audit | Output grad_bias as fp32 instead of bf16 | **KEEP** | Precision fix, no approach change |
| 14 | kernel-accuracy | 01_kernel_audit | Store P as fp32 or bf16 instead of fp16 | **KEEP** | Precision enhancement, preserves Gumbel-Softmax |
| 15 | kernel-accuracy | 01_kernel_audit | Store grad_logits as fp32 instead of fp16 | **KEEP** | Precision enhancement, preserves Gumbel-Softmax |
| 16 | kernel-accuracy | 01_kernel_audit | Use smem accumulator for soft bwd grad_palette | **KEEP** | Kernel speed enhancement |
| 17 | kernel-accuracy | 01_kernel_audit | Replace LCG PRNG with Philox/PCG for Gumbel | **KEEP** | Enhances Gumbel quality, preserves approach |
| 18 | kernel-accuracy | 02_numerical_analysis | Compute grad_W as fp32 (not bf16) | **KEEP** | Precision enhancement, no approach change |
| 19 | kernel-accuracy | 02_numerical_analysis | fp32 sum + bf16 cast for grad_palette reduction | **KEEP** | Precision enhancement, no approach change |
| 20 | kernel-accuracy | 02_numerical_analysis | Disable autocast in backward (fp32 throughout) | **KEEP** | Precision enhancement, no approach change |
| 21 | kernel-accuracy | 03_ste_analysis | Floor τ at 0.5 (not 0.1) | **KEEP** | Training recipe enhancement, preserves Gumbel+STE |
| 22 | kernel-accuracy | 03_ste_analysis | Switch to F.gumbel_softmax(hard=True) built-in | **KEEP** | Cleaner STE implementation, preserves approach |
| 23 | kernel-accuracy | 03_ste_analysis | Add final argmax step to extract hard indices | **KEEP** | Post-training extraction, preserves approach |
| 24 | kernel-accuracy | 03_ste_analysis | Switch to AQLM-style direct STE (no softmax) | **REJECT** | Replaces Gumbel-Softmax with AQLM pattern |
| 25 | kernel-accuracy | 04_literature_comparison | Add GPTQ-style second-order calibration | **REJECT** | Replaces 1d-kmeans with GPTQ Hessian calibration |
| 26 | kernel-accuracy | 04_literature_comparison | Outlier isolation (top 0.5% in FP16) | **REJECT** | Adds dense-sparse decomposition (SqueezeLLM pattern) — architecture change |
| 27 | kernel-accuracy | 04_literature_comparison | AWQ-style activation-aware scaling at calibration | **REJECT** | Adds AWQ scaling — calibration approach change |
| 28 | kernel-accuracy | 04_literature_comparison | RHT incoherence preprocessing (QuIP-style) | **REJECT** | Adds QuIP preprocessing — calibration approach change |
| 29 | kernel-accuracy | 04_literature_comparison | Switch to 8-dim VQ codebook (QuIP#-style) | **REJECT** | Replaces scalar LUT with VQ — fundamental change |
| 30 | kernel-accuracy | 04_literature_comparison | Use E8 lattice codebook (QuIP#) | **REJECT** | Replaces scalar LUT with lattice VQ |
| 31 | kernel-accuracy | 04_literature_comparison | Switch to additive VQ (AQLM 2×256) | **REJECT** | Replaces scalar LUT with additive VQ |
| 32 | kernel-accuracy | 05_convergence_analysis | Switch loss from cosine to block-MSE | **KEEP** | Loss function tuning, preserves approach |
| 33 | kernel-accuracy | 05_convergence_analysis | Train for 50K+ steps (currently 8K) | **KEEP** | Training duration, no approach change |
| 34 | kernel-accuracy | 05_convergence_analysis | Increase LoRA rank beyond 16/32 | **KEEP** | LoRA capacity enhancement, preserves approach |
| 35 | kernel-accuracy | 05_convergence_analysis | Per-block end-to-end joint training | **REJECT** | Restructures training (AQLM approach) |
| 36 | kernel-efficiency | 00_overview | Switch to vector quantization (VQ) | **REJECT** | Replaces scalar LUT with VQ |
| 37 | kernel-efficiency | 00_overview | Add GPTQ-style second-order calibration | **REJECT** | Replaces 1d-kmeans with GPTQ |
| 38 | kernel-efficiency | 00_overview | Outlier isolation (top 0.5% in FP16) | **REJECT** | SqueezeLLM dense-sparse decomposition |
| 39 | kernel-efficiency | 00_overview | Floor τ at 0.5 | **KEEP** | Duplicate of #21 — training recipe enhancement |
| 40 | kernel-efficiency | 00_overview | fp32 for grad_W | **KEEP** | Duplicate of #18 — precision enhancement |
| 41 | kernel-efficiency | 00_overview | bf16 or fp32 for P storage | **KEEP** | Duplicate of #14 — precision enhancement |
| 42 | kernel-efficiency | 00_overview | fp32 for grad_logits | **KEEP** | Duplicate of #15 — precision enhancement |
| 43 | kernel-efficiency | 00_overview | AWQ-style activation-aware scaling | **REJECT** | Duplicate of #27 — AWQ calibration change |
| 44 | kernel-efficiency | 00_overview | RHT incoherence preprocessing | **REJECT** | Duplicate of #28 — QuIP calibration change |
| 45 | kernel-efficiency | 00_overview | block-MSE loss instead of cosine | **KEEP** | Duplicate of #32 — loss tuning |
| 46 | kernel-efficiency | 01_profiling_breakdown | CUDA event timing instead of sync | **KEEP** | Profiling tooling enhancement |
| 47 | kernel-efficiency | 01_profiling_breakdown | Background CPU tokenization (DataLoader num_workers=4) | **KEEP** | Data pipeline enhancement, no approach change |
| 48 | kernel-efficiency | 01_profiling_breakdown | Fuse gradient clipping into optimizer | **KEEP** | Optimizer speed enhancement |
| 49 | kernel-efficiency | 01_profiling_breakdown | Manual grad zeroing inside fused optimizer | **KEEP** | Optimizer speed enhancement |
| 50 | kernel-efficiency | 01_profiling_breakdown | Set CUBLAS_WORKSPACE_CONFIG cap | **KEEP** | Enables batch=64, no approach change |
| 51 | kernel-efficiency | 02_fused_bwd_fix | AoS layout for P ((4,K,N)→(K,N,4)) | **KEEP** | Fixes 10x slow fused backward, preserves approach |
| 52 | kernel-efficiency | 02_fused_bwd_fix | Re-enable fused bwd_fused_aos CUDA kernel | **KEEP** | Depends on #51 — kernel speed enhancement |
| 53 | kernel-efficiency | 03_batched_compute_pw | Batched compute_P_W (1 launch for 25 layers) | **KEEP** | Fuses 25 kernel launches into 1, no approach change |
| 54 | kernel-efficiency | 03_batched_compute_pw | CUDA Graphs as alternative to descriptor batching | **KEEP** | Alternative to #53, either works |
| 55 | kernel-efficiency | 04_sm120_optimal | TMA migration (cp.async.bulk.tensor) | **REJECT** | Radical kernel rewrite — keep mma.sync.m16n8k16 |
| 56 | kernel-efficiency | 04_sm120_optimal | wgmma.mma_async migration | **REJECT** | Radical kernel rewrite — keep mma.sync |
| 57 | kernel-efficiency | 04_sm120_optimal | tcgen05.mma (Blackwell tensor core) | **REJECT** | Radical kernel rewrite — keep mma.sync |
| 58 | kernel-efficiency | 04_sm120_optimal | cluster.sync (Distributed Shared Memory) | **REJECT** | Radical kernel rewrite |
| 59 | kernel-efficiency | 04_sm120_optimal | setmaxnreg (dynamic register cap) | **REJECT** | Radical kernel rewrite |
| 60 | kernel-efficiency | 04_sm120_optimal | Upgrade to -std=c++20 + CUTLASS 3.5+ | **REJECT** | Only needed for tcgen05 (rejected) |
| 61 | kernel-efficiency | 04_sm120_optimal | Pre-compile static library | **REJECT** | Only needed for CUTLASS (rejected) |
| 62 | kernel-efficiency | 04_sm120_optimal | __launch_bounds__(256, 2) hint | **KEEP** | Simple kernel optimization, no approach change |
| 63 | kernel-efficiency | 05_memory_optimization | Buffer pooling for P_aos | **KEEP** | Memory enhancement, depends on #51 |
| 64 | kernel-efficiency | 05_memory_optimization | Chunked reduction kernel (no (K,N,4) materialization) | **KEEP** | Memory enhancement, no approach change |
| 65 | kernel-efficiency | 05_memory_optimization | Keep grad_W as cuBLAS GEMM (don't fuse matmul) | **KEEP** | Design guidance for #64 |
| 66 | kernel-efficiency | 05_memory_optimization | Hybrid FP32 v, bf16 m and master AdamW state | **KEEP** | Optimizer memory enhancement |
| 67 | kernel-efficiency | 05_memory_optimization | Adafactor (no second moment) | **REJECT** | Replaces AdamW with Adafactor — optimizer change |
| 68 | kernel-efficiency | 05_memory_optimization | Gradient checkpointing for batch=128+ | **KEEP** | Enables bigger batch, no approach change |
| 69 | kernel-efficiency | 06_stream_overlap | Stream double-buffering (true producer/consumer) | **KEEP** | Overlaps teacher/student, no approach change |
| 70 | kernel-efficiency | 06_stream_overlap | Persistent CUDA stream pool | **KEEP** | Eliminates per-step stream creation |
| 71 | kernel-efficiency | 07_literature_comparison | Fused activation kernels (silu_and_mul) | **KEEP** | Fuses MLP activation, no approach change |
| 72 | kernel-efficiency | 07_literature_comparison | torch.compile for elementwise fusion | **KEEP** | Speed enhancement, requires PartialWrapper→nn.Module (#39 arch) |
| 73 | kernel-efficiency | 07_literature_comparison | Per-shape kernel specialisation (llama.cpp) | **KEEP** | Kernel optimization, no approach change |
| 74 | kernel-efficiency | 07_literature_comparison | Stream-K decomposition (CUTLASS) | **KEEP** | Kernel optimization, no approach change |
| 75 | kernel-efficiency | 07_literature_comparison | FLUTE interleaved palette layout (K, N/4, 4) | **KEEP** | Layout optimization, no approach change |
| 76 | kernel-efficiency | 07_literature_comparison | GPTQ-Marlin 3-stage load-dequant-mma pipeline | **KEEP** | Kernel pipeline enhancement, no approach change |
| 77 | kernel-efficiency | 07_literature_comparison | GPTQ-Marlin warp-shuffle for grad_palette | **KEEP** | Kernel optimization, no approach change |
| 78 | kernel-efficiency | 08_recommendations | Fused AdamW (fused=True or bitsandbytes 8bit) | **KEEP** | Optimizer speed enhancement |
| 79 | kernel-efficiency | 08_recommendations | CUDA Graphs for full step | **KEEP** | Eliminates dispatch overhead, no approach change |
| 80 | kernel-efficiency | 08_recommendations | Validation plan: numerical equivalence + perf benchmark | **KEEP** | Testing guidance, no approach change |
| 81 | indices-training | 00_overview | Polynomial τ schedule (warmup + quadratic decay to 0.5) | **KEEP** | Enhances τ schedule, preserves Gumbel |
| 82 | indices-training | 00_overview | Switch to deterministic-ST (remove Gumbel noise, LLT pattern) | **REJECT** | Replaces Gumbel with LLT deterministic softmax |
| 83 | indices-training | 00_overview | Tighten logit clamp ±20→±5 | **KEEP** | Adaptive clamp, preserves Gumbel |
| 84 | indices-training | 00_overview | LLT 1/√(N_i) per-group gradient rescaling | **KEEP** | Gradient enhancement, preserves Gumbel |
| 85 | indices-training | 00_overview | Hessian-weighted gradient (SqueezeLLM pattern) | **REJECT** | Adds Hessian weighting — calibration approach change |
| 86 | indices-training | 00_overview | Switch to LUT-Q pattern (FP shadow + k-means reassignment) | **REJECT** | Replaces Gumbel-Softmax with LUT-Q |
| 87 | indices-training | 01_gumbel_softmax_audit | Use curand_normal/Philox instead of LCG | **KEEP** | Duplicate of #17 — Gumbel quality enhancement |
| 88 | indices-training | 01_gumbel_softmax_audit | Fuse STE bridge into CUDA kernel | **KEEP** | Kernel optimization, eliminates wasted y_soft matmul |
| 89 | indices-training | 01_gumbel_softmax_audit | Use ±3 logit init (middle ground) | **KEEP** | Init tuning, preserves Gumbel |
| 90 | indices-training | 01_gumbel_softmax_audit | Revive dead CUDA bwd kernel with coalesced P | **KEEP** | Duplicate of #51 — AoS P layout |
| 91 | indices-training | 01_gumbel_softmax_audit | Exponential τ schedule (Jang et al.) | **KEEP** | Alternative τ schedule, preserves Gumbel |
| 92 | indices-training | 02_ste_correctness | A/B test vanilla STE (Bengio 2013) | **REJECT** | Replaces Gumbel-ST with vanilla STE |
| 93 | indices-training | 02_ste_correctness | Hybrid STE schedule (vanilla→Gumbel) | **REJECT** | Phase-switch, partially replaces Gumbel |
| 94 | indices-training | 02_ste_correctness | Log-scale logit initialization | **KEEP** | Init enhancement, preserves Gumbel |
| 95 | indices-training | 03_gradient_flow_analysis | Scale gradient by 1/τ | **KEEP** | Gradient enhancement, preserves Gumbel |
| 96 | indices-training | 03_gradient_flow_analysis | Direct P parameterization (drop softmax) | **REJECT** | Replaces Gumbel-Softmax with direct P |
| 97 | indices-training | 03_gradient_flow_analysis | Clamp logit gap to max=4 | **KEEP** | Init/clamp enhancement, preserves Gumbel |
| 98 | indices-training | 04_tau_schedule | Early-stop indices when flip rate < 0.01% | **KEEP** | Training efficiency, preserves Gumbel |
| 99 | indices-training | 05_literature_comparison | Dense-and-sparse decomposition (SqueezeLLM) | **REJECT** | Duplicate of #26 — architecture change |
| 100 | indices-training | 05_literature_comparison | Activation-weighted gradient (AWQ) | **REJECT** | Duplicate of #85 — calibration approach change |
| 101 | indices-training | 05_literature_comparison | AWQ-style scaled grid for codebook init | **REJECT** | Replaces k-means init with AWQ scaling |
| 102 | indices-training | 05_literature_comparison | From-scratch training (BitNet pattern) | **REJECT** | Replaces k-means init + Gumbel with from-scratch |
| 103 | indices-training | 05_literature_comparison | Co-train per-group scale parameter (BitNet/AWQ) | **REJECT** | Adds scale parameter — architecture change |
| 104 | indices-training | 05_literature_comparison | Extend freeze_settled_palettes for oscillating indices | **KEEP** | Duplicate of #7 — already in codebase |
| 105 | indices-training | 06_optimizer_analysis | Raise AdamW eps from 1e-8 to 1e-6 | **KEEP** | Optimizer stability, no approach change |
| 106 | indices-training | 06_optimizer_analysis | Hybrid optimizer schedule (SGD→AdamW) | **REJECT** | Phase-switch optimizer, partially replaces AdamW |
| 107 | indices-training | 06_optimizer_analysis | Switch betas to (0.9, 0.999) with deterministic-ST | **REJECT** | Depends on #82 (deterministic-ST, rejected) |
| 108 | palettes-training | 00_overview | Floor τ at 0.5 | **KEEP** | Duplicate of #21 — training recipe |
| 109 | palettes-training | 00_overview | Switch P and grad_logits to fp32 | **KEEP** | Duplicate of #14, #15 — precision |
| 110 | palettes-training | 00_overview | Switch grad_W to fp32 | **KEEP** | Duplicate of #18 — precision |
| 111 | palettes-training | 00_overview | AWQ-style activation-aware scaling | **REJECT** | Duplicate of #27 — AWQ calibration change |
| 112 | palettes-training | 00_overview | Switch loss to combined cosine + block-MSE | **KEEP** | Duplicate of #2, #32 — loss tuning |
| 113 | palettes-training | 00_overview | Train for 50K steps | **KEEP** | Duplicate of #33 — training duration |
| 114 | palettes-training | 00_overview | Add GPTQ-style second-order calibration | **REJECT** | Duplicate of #25 — GPTQ sidestep |
| 115 | palettes-training | 00_overview | Add outlier isolation | **REJECT** | Duplicate of #26 — SqueezeLLM sidestep |
| 116 | palettes-training | 00_overview | Add RHT incoherence preprocessing | **REJECT** | Duplicate of #28 — QuIP sidestep |
| 117 | palettes-training | 00_overview | Switch to 8-dim VQ codebook | **REJECT** | Duplicate of #29 — VQ sidestep |
| 118 | palettes-training | 00_overview | Replace Gumbel-STE with AQLM-style direct STE | **REJECT** | Duplicate of #24 — AQLM sidestep |
| 119 | palettes-training | 00_overview | Rewrite CUDA kernels for VQ | **REJECT** | Depends on #117 — VQ sidestep |
| 120 | palettes-training | 00_overview | Add per-block end-to-end training | **REJECT** | Duplicate of #35 — AQLM training restructure |
| 121 | palettes-training | 01_palette_audit | Promote palette to fp32 | **KEEP** | Duplicate of #3 — precision enhancement |
| 122 | palettes-training | 01_palette_audit | Fix LUT disk serialization fp16→fp32 | **KEEP** | Precision fix, no approach change |
| 123 | palettes-training | 01_palette_audit | Per-group gradient clip instead of global 0.3 | **KEEP** | Duplicate of #4 — optimizer enhancement |
| 124 | palettes-training | 01_palette_audit | Enable LoftQ SVD init for LoRA | **KEEP** | Duplicate of #1 — LoRA init enhancement |
| 125 | palettes-training | 02_gradient_correctness | Use loss that directly optimizes cos | **KEEP** | Duplicate of #2, #32 — loss tuning |
| 126 | palettes-training | 02_gradient_correctness | Use smaller GROUP_SIZE (128 or 64) | **KEEP** | Duplicate of #6 — calibration enhancement |
| 127 | palettes-training | 03_precision_analysis | fp32 palette with bf16 kernel cast (quick win) | **KEEP** | Implementation detail of #3 |
| 128 | palettes-training | 04_kmeans_vs_gradient | Hybrid GD + periodic k-means re-quantization (LUT-Q) | **REJECT** | Replaces gradient-trained palettes with LUT-Q pattern |
| 129 | palettes-training | 04_kmeans_vs_gradient | Enable freeze_settled_palettes | **KEEP** | Duplicate of #7 — already in codebase |
| 130 | palettes-training | 04_kmeans_vs_gradient | Try GPTQ with fixed k-means LUT | **REJECT** | Adds GPTQ — calibration sidestep |
| 131 | palettes-training | 05_loss_function | Switch loss to 1-cos+norm_mse cos=0.8 mse=0.2 | **KEEP** | Duplicate of #2 — loss tuning |
| 132 | palettes-training | 05_loss_function | Enable cyclic loss schedule | **KEEP** | Loss schedule, preserves approach |
| 133 | palettes-training | 05_loss_function | Per-Linear loss selection | **KEEP** | Loss tuning, preserves approach |
| 134 | palettes-training | 06_staged_training | Staged palette-then-indices training (Schedule C) | **KEEP** | Training schedule, preserves approach |
| 135 | palettes-training | 06_staged_training | Better τ schedule: τ=0.5→0.3 hold at 0.3 | **KEEP** | τ tuning, preserves Gumbel |
| 136 | palettes-training | 06_staged_training | Use rank-64 or rank-128 LoRA for 5 worst Linears | **KEEP** | Duplicate of #34 — LoRA capacity |
| 137 | palettes-training | 07_literature_comparison | AWQ-style per-output-channel trainable scaling | **REJECT** | Adds scale parameter — architecture change |
| 138 | palettes-training | 08_recommendations | Use F.gumbel_softmax(hard=True) built-in | **KEEP** | Duplicate of #22 — cleaner STE |
| 139 | palettes-training | 08_recommendations | Fix OOB indices sentinel (0xFF) | **KEEP** | Duplicate of #8 — bug fix |
| 140 | palettes-training | 08_recommendations | Add NaN guard in hard kernel output | **KEEP** | Duplicate of #9 — robustness fix |
| 141 | architecture-review | 02_partial_wrapper | PartialWrapper → nn.Module | **KEEP** | Speed enhancement (enables torch.compile + checkpointing) |
| 142 | architecture-review | 05_training_loop_refactor | Refactor monolithic train_super_block into Trainer class | **REJECT** | Architecture rewrite (modular package) |
| 143 | architecture-review | 03_memory_waste | In-place teacher/student loading (shared prefix) | **KEEP** | Memory enhancement, no approach change |
| 144 | architecture-review | 03_memory_waste | Enable gradient checkpointing (use_reentrant=False) | **KEEP** | Duplicate of #68 — memory enhancement |
| 145 | architecture-review | 03_memory_waste | Unify index_logits dtype to bf16 (remove ±20 clamp) | **KEEP** | Dtype unification, no approach change |
| 146 | architecture-review | 03_memory_waste | Lazy-allocate _flat_idx cache | **KEEP** | Memory enhancement (saves 3.56 GB), no approach change |
| 147 | architecture-review | 03_memory_waste | Explicit del + gc.collect() for orphaned embed_tokens | **KEEP** | Memory fix (1.27 GB leak), no approach change |
| 148 | architecture-review | 03_memory_waste | Drop fp32 master weights for LoRA | **KEEP** | Optimizer memory enhancement |
| 149 | architecture-review | 03_memory_waste | Use 8-bit optimizer (bitsandbytes Adam8bit) | **KEEP** | Alternative to #78 — optimizer memory |
| 150 | architecture-review | 04_data_pipeline | Pre-tokenize FineWeb-Edu to on-disk binary cache | **KEEP** | Data pipeline enhancement, no approach change |
| 151 | architecture-review | 04_data_pipeline | AsyncPrefetchLoader with pinned-memory ring buffer | **KEEP** | Data pipeline enhancement, no approach change |
| 152 | architecture-review | 04_data_pipeline | Standard DataLoader with num_workers=4 | **KEEP** | Duplicate of #47 — data pipeline |
| 153 | architecture-review | 04_data_pipeline | Fix eval-set cache filename (include seq_len/vocab) | **KEEP** | Bug fix, no approach change |
| 154 | architecture-review | 04_data_pipeline | Cache calibration tokens per super-block | **KEEP** | Calibration efficiency, no approach change |
| 155 | architecture-review | 04_data_pipeline | Delete orphaned cached_tokens.pt + add *.pt to .gitignore | **KEEP** | Repo cleanup, no approach change |
| 156 | architecture-review | 04_data_pipeline | Set HF_TOKEN / use local HF mirror | **KEEP** | Data pipeline config, no approach change |
| 157 | architecture-review | 05_training_loop_refactor | Config dataclass as single source of truth | **REJECT** | Architecture rewrite (config system) |
| 158 | architecture-review | 06_logging_metrics | MetricsLogger with multiple backends (wandb/tensorboard) | **REJECT** | Tooling integration — out of scope |
| 159 | architecture-review | 06_logging_metrics | Log artifacts and system metrics | **REJECT** | Tooling integration — out of scope |
| 160 | architecture-review | 06_logging_metrics | Rewrite sweep_qwen.py to consume CSV | **REJECT** | Tooling — out of scope |
| 161 | architecture-review | 07_testing_ci | 4-tier test scaffold (unit/component/integration/regression) | **REJECT** | Testing infrastructure — out of scope |
| 162 | architecture-review | 07_testing_ci | GitHub Actions CI/CD + self-hosted GPU runner | **REJECT** | CI/CD — out of scope |
| 163 | architecture-review | 07_testing_ci | pytest-cov coverage measurement | **REJECT** | Testing — out of scope |
| 164 | architecture-review | 01_architecture_audit | state_dict() + manifest JSON | **KEEP** | Checkpoint enhancement, depends on #141 |
| 165 | architecture-review | 01_architecture_audit | Replace palettize_core monkey-patch with param | **KEEP** | Code quality fix, no approach change |
| 166 | architecture-review | 01_architecture_audit | Add setup.py/pyproject.toml + sm_120 docs | **KEEP** | Build system, no approach change |
| 167 | architecture-review | 00_executive_summary | Wire export pipeline (merge_qwen.py + assemble_qwen.py) | **KEEP** | Export tooling, no approach change |
| 168 | architecture-review | 02_partial_wrapper | Legacy checkpoint migration script | **KEEP** | Migration helper, depends on #141 |
| 169 | architecture-review | 08_literature_comparison | Adopt HF Trainer pattern | **REJECT** | Architecture rewrite — modular package |
| 170 | architecture-review | 08_literature_comparison | Adopt Megatron-LM MMapIndexedDataset format | **KEEP** | Duplicate of #150 — data format |
| 171 | literature-review | 00_executive_summary | Halve group size 256→128 | **KEEP** | Duplicate of #6, #126 — calibration enhancement |
| 172 | literature-review | 00_executive_summary | GPTVQ vector quantization (g=2) | **REJECT** | Replaces scalar LUT with VQ |
| 173 | literature-review | 00_executive_summary | AWQ per-channel activation-aware scale | **REJECT** | Duplicate of #27 — AWQ calibration change |
| 174 | literature-review | 00_executive_summary | SqueezeLLM dense/sparse split | **REJECT** | Duplicate of #26 — architecture change |
| 175 | literature-review | 00_executive_summary | QuIP# Hadamard pre-rotation | **REJECT** | Duplicate of #28 — QuIP calibration change |
| 176 | literature-review | 00_executive_summary | GPTQ Hessian-inverse error propagation | **REJECT** | Duplicate of #25 — GPTQ sidestep |
| 177 | literature-review | 00_executive_summary | LLT 1/√(N_k) gradient rescaling | **KEEP** | Duplicate of #84 — gradient enhancement |
| 178 | literature-review | 00_executive_summary | Drop Gumbel noise (deterministic softmax) | **REJECT** | Duplicate of #82 — replaces Gumbel with LLT |
| 179 | literature-review | 00_executive_summary | Tighten logit clamp ±20→±5τ | **KEEP** | Duplicate of #83 — adaptive clamp |
| 180 | literature-review | 00_executive_summary | Lloyd-Max Gaussian codebook init | **REJECT** | Replaces k-means with Lloyd-Max (tested as bullshit) |
| 181 | literature-review | 00_executive_summary | Learnable clipping threshold (OmniQuant) | **REJECT** | Adds OmniQuant clipping — calibration change |
| 182 | literature-review | 00_executive_summary | LUT-Q FP shadow + k-means reassignment | **REJECT** | Duplicate of #86 — LUT-Q sidestep |
| 183 | literature-review | 00_executive_summary | ExLlamaV2 per-layer mixed-precision bit allocation | **REJECT** | Mixed-precision — architecture change |
| 184 | literature-review | 01_gptq_family | Activation-ordered group processing (desc_act) | **REJECT** | GPTQ-specific, depends on GPTQ (rejected) |
| 185 | literature-review | 01_gptq_family | Marlin-style inference kernel | **KEEP** | Inference kernel (future), no approach change |
| 186 | literature-review | 02_awq_smoothquant | SmoothQuant per-channel activation smoothing | **REJECT** | Adds SmoothQuant — calibration change |
| 187 | literature-review | 02_awq_smoothquant | OmniQuant LET (learnable s and t) | **REJECT** | Adds OmniQuant — calibration change |
| 188 | literature-review | 02_awq_smoothquant | AffineQuant lower-triangular transform | **REJECT** | Adds AffineQuant — calibration change |
| 189 | literature-review | 03_codebook_methods | AQLM additive multi-codebook (M=2) | **REJECT** | Replaces scalar LUT with additive VQ |
| 190 | literature-review | 03_codebook_methods | AQLM beam search index assignment | **REJECT** | Replaces Gumbel with beam search |
| 191 | literature-review | 03_codebook_methods | Exponential τ anneal schedule | **KEEP** | Duplicate of #91 — τ tuning, preserves Gumbel |
| 192 | literature-review | 04_1bit_methods | Two-stage LR schedule (warmup + decay) | **KEEP** | LR schedule, preserves approach |
| 193 | literature-review | 04_1bit_methods | Per-tensor learnable α scale (BitNet) | **REJECT** | Adds scale parameter — architecture change |
| 194 | literature-review | 04_1bit_methods | Initialize one palette level at 0 (sparse) | **REJECT** | Changes palette structure — approach change |
| 195 | literature-review | 04_1bit_methods | From-scratch training (no k-means init) | **REJECT** | Duplicate of #102 — replaces k-means init |
| 196 | literature-review | 04_1bit_methods | Saturating tanh STE for palette gradient | **KEEP** | Gradient enhancement, preserves approach |
| 197 | literature-review | 04_1bit_methods | SubLN (learnable gain before LayerNorm) | **REJECT** | Adds SubLN — architecture change |
| 198 | literature-review | 05_production_frameworks | Per-sub-group scale hierarchy (llama.cpp Q2_K) | **REJECT** | Adds sub-group scales — architecture change |
| 199 | literature-review | 05_production_frameworks | Double quantization of palette to INT8 | **KEEP** | Memory enhancement, no approach change |
| 200 | literature-review | 05_production_frameworks | Multi-platform kernel support (sm_80/Metal/Vulkan) | **KEEP** | Future portability, no approach change |
| 201 | literature-review | 05_production_frameworks | Framework integration (HF/vLLM/TensorRT-LLM) | **REJECT** | Tooling integration — out of scope |

---

## Summary

| Verdict | Count | Description |
|---------|-------|-------------|
| **KEEP** | 101 | ANE-aligned enhancements that preserve our approach |
| **REJECT** | 99 | Sidesteps that replace our approach (GPTQ, LLT, LUT-Q, VQ, etc.) |
| **Duplicates** | ~40 | KEEP items that appear in multiple agents (consolidated) |
| **Unique KEEP** | ~61 | Distinct enhancements after deduplication |

### KEEP breakdown by category:
- **Training recipe** (~20): τ schedule, loss tuning, logit clamp, LoftQ init, freeze_settled, training duration
- **Kernel precision** (~15): fp32 for P/grad_logits/grad_W/palette, fp32 autocast disable, NaN guards
- **Kernel efficiency** (~15): AoS P layout, fused bwd, batched compute_P_W, stream double-buffer, fused AdamW, CUDA Graphs
- **Memory** (~10): gradient checkpointing, lazy _flat_idx, buffer pooling, shared teacher/student, chunked reduction
- **Speed enhancements** (~5): PartialWrapper→nn.Module, torch.compile, fused activations, per-shape kernels

### REJECT breakdown by category:
- **GPTQ alternatives** (~10): Hessian calibration, desc_act, GPTVQ
- **LLT/k-means alternatives** (~8): deterministic softmax, LUT-Q re-quantization, Lloyd-Max
- **VQ/codebook changes** (~10): QuIP#, AQLM, GPTVQ, E8 lattice, additive VQ
- **Architecture rewrites** (~15): modular package, Trainer class, Config dataclass, HF Trainer pattern
- **Calibration changes** (~12): AWQ, SmoothQuant, OmniQuant, AffineQuant, QuIP RHT
- **Tooling** (~10): wandb, tensorboard, pytest, CI/CD
- **Other sidesteps** (~10): from-scratch training, BitNet, mixed-precision, SubLN, scale parameters
