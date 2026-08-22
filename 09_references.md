# 09 — References

> **Wave 4 deliverable #3.** Target: ≥2 pages. Consolidated bibliography
> of arXiv papers, NVIDIA documentation, GitHub repositories, and
> specification documents cited throughout `00_overview.md` through
> `08_recommendations.md`.

---

## 1. Academic papers (arXiv)

### Quantization & LUT training

1. **Jang, Gu, Poole (2017)** — "Categorical Reparameterization with Gumbel-Softmax". arXiv:1611.01144.
   The original Gumbel-Softmax estimator. Our STE implementation in `fused_lut_linear_cuda.py` lines 580–596 follows this paper's `W_hard - W_soft.detach() + W_soft` trick.

2. **Tseng, Huang, Kauvar et al. (2024)** — "FLUTE: A Simple, Efficient, and Flexible Approach for Mixed-Precision Quantized Neural Networks in PyTorch". arXiv:2407.10960.
   LUT matmul via `torch.gather` + batched einsum. The interleaved palette layout `(K, N/4, 4)` is from this paper.

3. **Stock, Arbelaez et al. (CVPR 2022)** — "LLT: A Trainable Lookup-Table Approach for Neural Network Quantization".
   Early learnable-LUT quantisation. Our `index_logits` trainable parameters follow this paradigm.

4. **Liu, Wang, Zhang et al. (2019)** — "LLSQ: Low-Bit Learnable Step Size Quantization". arXiv:1902.08153.
   Adapted in our palette-learning schedule.

5. **Zhou, Yao, Guo et al. (2018)** — "LUT-Networks: Deep Neural Networks with Tabulated Activation Functions". arXiv:1811.05355.
   Iterative LUT training; foundational to our super-block approach.

6. **Gao, Hu, Liu et al. (2024)** — "GPTQ-Marlin: Efficient and Accurate 4-bit Matrix-Multiplication". arXiv:2405.19020.
   The 3-stage load-dequant-mma pipeline inspiration for our Phase A+C design.

7. **Frantar, Alistarh et al. (2022)** — "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers". arXiv:2210.17323.
   The base quantisation algorithm used in our `palettize_core.py` `pack_idx2` function.

8. **van Baalen, Burcaleanu et al. (2024)** — "GSQ: Gumbel-Softmax Quantization". arXiv:2604.18556.
   Reference for our Gumbel-Softmax index training.

### Attention & kernel design

9. **Dao, Fu, Ermon et al. (2022)** — "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness". arXiv:2205.14135.
   Original FlashAttention; foundational tiling philosophy cited in `07_literature_comparison.md` §1.

10. **Dao (2023)** — "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning". arXiv:2307.08691.
    Backward-pass fusion technique. Referenced in `02_fused_bwd_fix.md`.

11. **Shah, Bikshandi, Zhang et al. (2024)** — "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision". arXiv:2407.08608.
    Warp-specialised producer/consumer pattern. Cited in `04_sm120_optimal.md` §4.2 and `07_literature_comparison.md` §1.

12. **Kwon, Li, Zhuang et al. (SOSP 2023)** — "Efficient Memory Management for Large Language Model Serving with PagedAttention". arXiv:2309.06180.
    vLLM's CUDA Graphs and persistent stream pool. Cited in `07_literature_comparison.md` §2.

### Optimizer design

13. **Kingma, Ba (2014)** — "Adam: A Method for Stochastic Optimization". arXiv:1412.6980.
    The base AdamW algorithm; used in our `FP32MasterAdamW` implementation.

14. **Loshchilov, Hutter (2017)** — "Decoupled Weight Decay Regularization". arXiv:1711.05101.
    AdamW (with decoupled weight decay). Used in `scripts/train_qwen.py` line 594.

15. **Jordan, Chen, Noci et al. (2024)** — "Muon: An optimizer for neural network training". (Public repo, no arXiv yet.)
    Newton-Schulz orthogonalisation; used in our `Muon` optimizer class (lines 106–150 of `train_qwen.py`).

16. **Shazeer, Stern (2018)** — "Adafactor: Adaptive Learning Rates with Sublinear Memory Cost". arXiv:1804.04235.
    Memory-efficient Adam alternative; discussed in `05_memory_optimization.md` §5.3.

---

## 2. NVIDIA documentation

17. **NVIDIA PTX ISA 8.7 Documentation** — https://docs.nvidia.com/cuda/parallel-thread-execution/
    The PTX reference. Sections referenced in `04_sm120_optimal.md`:
    - §9.7.12.4 — `cp.async.bulk.tensor` (TMA)
    - §9.7.13 — Warp-Group Matrix-Multiply-Accumulate Instructions
    - §9.7.13.5 — `tcgen05.mma` (Blackwell)
    - §9.7.14 — Thread block clusters (`cluster.sync`)
    - §9.7.15.10 — `setmaxnreg`

18. **NVIDIA CUDA C++ Programming Guide 12.8** — https://docs.nvidia.com/cuda/cuda-c-programming-guide/
    - §7.x — TMA descriptor creation (`cuTensorMapEncodeTiled`)
    - §B.x — `__grid_constant__` qualifier
    - §W.x — Blackwell sm_120 features

19. **NVIDIA cuBLAS Documentation** — https://docs.nvidia.com/cuda/cublas/
    - `cublasGemmStridedBatchedEx` — used for potential batched matmul
    - `cublasSetWorkspace` — controls the workspace allocation (Patch #7)

20. **NVIDIA Nsight Compute Documentation** — https://docs.nvidia.com/nsight-compute/
    For kernel-level profiling after each patch.

---

## 3. Reference implementations (GitHub)

21. **CUTLASS 3.x (NVIDIA)** — https://github.com/NVIDIA/cutlass (branch `v3.x`)
    - `cutlass/examples/72_hopper_warp_specialized_gemm` — warp-specialised pattern
    - `cutlass/examples/75_blackwell_sm100_tensor_op_fp8` — `tcgen05.mma` example
    - `cutlass/examples/55_hopper_mixed_dtype_gemm` — mixed bf16/fp32 (for grad_palette)
    - `cutlass/include/cute/atom/mma_atom.hpp` — CUTE atoms

22. **FlashAttention (Tri Dao)** — https://github.com/Dao-AILab/flash-attention
    - `csrc/flash_attn/flash_api.cu` — kernel entrypoints
    - `csrc/flash_attn/flash_fwd_kernel.h` — forward kernel
    - `csrc/flash_attn/flash_bwd_kernel.h` — backward kernel

23. **vLLM** — https://github.com/vllm-project/vllm
    - `vllm/worker/worker_base.py` — CUDA Graphs capture pattern
    - `vllm/model_executor/layers/quantization/gptq_marlin.py` — Marlin integration
    - `vllm/model_executor/layers/quantization/awq_marlin.py` — AWQ variant

24. **llama.cpp (Georgi Gerganov)** — https://github.com/ggerganov/llama.cpp
    - `ggml/src/ggml-cuda/kquv.cu` — k-quants kernels
    - `ggml/src/ggml-cuda/mmq.cu` — matrix-matrix quantised GEMM
    - `ggml/src/ggml-cuda/mul_mat_vec_q.cu` — GEMV kernels

25. **Marlin (IST-DASLab)** — https://github.com/IST-DASLab/marlin
    - `marlin_cuda_kernel.cu` — original 4-bit GEMM kernel
    - `marlin_moe.cpp` — MoE variant

26. **FLUTE-Quant (Han-Lin-CHW)** — https://github.com/Han-Lin-CHW/flute-quant
    - `flute/flute_kernel.cu` — LUT matmul kernel
    - `flute/flute_torch.py` — PyTorch bindings

27. **bitsandbytes** — https://github.com/bitsandbytes-foundation/bitsandbytes
    - `bitsandbytes/optim/optimizer.py` — fused AdamW implementation
    - `bitsandbytes/functional.py` — 8-bit Adam state

28. **PyTorch** — https://github.com/pytorch/pytorch
    - `torch/optim/adamw.py` — `AdamW(fused=True)` reference
    - `torch/utils/cpp_extension.py` — `load_inline` documentation
    - `torch/cuda/graphs.py` — CUDA Graphs Python API

---

## 4. Specification documents (in this repo)

29. **`SPEC.md`** — Qwen3.5-4B Palettization project specification
    Section 1: model architecture (8 super-blocks of 4 layers, GatedDeltaNet + full-attn hybrid).
    Section 4: 2-bit LUT format (`palette[G, 4] bf16`, `indices[K, N] uint8`, `GROUP_SIZE = 256`).
    Section 6: training cost (~1.0 s/step, 5000 steps/super-block, 11 hours total for 8 super-blocks).
    Section 8: reference list.

30. **`logs/train_sb0.log`** — runtime training log
    Line 23: `Loaded prefix: 1,081,957,952 params (4 layers + embed_tokens)`.
    Line 39: `AdamW groups: 106 — {'layernorms': 19, 'lora': 62, 'palettes': 25}` (confirms 25 PalettizedLinear modules per super-block).
    Lines 28–34: param counts (`indices: 1,782,579,200`, `palettes: 2,208`, etc.).

31. **`scripts/profile_training.py`** — per-component profiler
    Lines 110–199: timing structure (teacher_fwd, student_fwd, loss, backward, clip, muon_step, adamw_step, sched_step).
    Lines 226–240: summary printing with `avg_ms / first_50 / last_50 / ratio`.

32. **`scripts/profile_nosync.py`** — pipelined profiler (no per-component sync)
    Lines 89–135: single-step time measurement.
    Lines 163–190: streaming vs cached comparison.

33. **`scripts/fused_lut_kernel.cu`** — the CUDA kernels (1,632 lines)
    Lines 1–21: design notes (REV-2).
    Lines 309–518: `fused_lut_linear_fwd_tc_kernel` (TC forward).
    Lines 1498–1610: `fused_lut_linear_soft_bwd_fused_kernel` (the 3.8× slow kernel).
    Lines 1301–1352: `fused_lut_linear_soft_compute_P_W_kernel`.

34. **`scripts/fused_lut_linear_cuda.py`** — Python autograd wrapper (709 lines)
    Lines 519–684: `CUDAFusedLUTLinearSoft` class with forward + backward.
    Lines 628–639: the "strided P access" comment (verbatim, the smoking gun).
    Lines 658–673: the `(K, N, 4)` intermediate materialisation.
    Lines 552–606: the STE `W = W_hard - W_soft.detach() + W_soft` trick.

35. **`scripts/train_qwen.py`** — training loop (1,266 lines)
    Lines 1058–1080: the (broken) stream overlap pattern.
    Lines 552–611: three-optimizer stack (Muon + AdamW + FP32MasterAdamW for indices).
    Lines 1126–1141: two-tier gradient clipping (1.0 for indices, 0.3 for others).

36. **`scripts/qwen_model.py`** — `PalettizedLinear` and `QwenLoRA` modules
    Dual-mode forward (soft training, hard eval).

37. **`scripts/palettize_core.py`** — 2-bit packing (`pack_idx2`, `write_lut_scalar`).
    Constants: `BITWIDTH = 2`, `GROUP_SIZE = 256`, `PALETTE_SIZE = 4`.

---

## 5. Internal research documents (this folder)

38. **`00_overview.md`** — Executive summary of the 530 ms/step problem and the 10-patch roadmap.

39. **`01_profiling_breakdown.md`** — Detailed analysis of where the 530 ms goes (Wave 1).

40. **`02_fused_bwd_fix.md`** — Fixing the 10× slow fused backward (Wave 2).

41. **`03_batched_compute_pw.md`** — Fusing 25 `compute_P_W` launches into 1 (Wave 2).

42. **`04_sm120_optimal.md`** — TMA / `wgmma` / `tcgen05` options for Blackwell (Wave 3).

43. **`05_memory_optimization.md`** — Eliminating `(K, N, 4)` intermediates (Wave 3).

44. **`06_stream_overlap.md`** — True producer/consumer with double-buffering (Wave 3).

45. **`07_literature_comparison.md`** — Comparison with FlashAttention, vLLM, llama.cpp, CUTLASS, FLUTE, Marlin (Wave 4).

46. **`08_recommendations.md`** — Concrete code patches with expected speedup (Wave 4).

47. **`09_references.md`** — This document.

---

## 6. Closing note on versioning

All NVIDIA PTX/CUDA references in this document are based on the latest
stable as of the document date (2026-08-22):

- PTX ISA 8.7 (CUDA 12.8)
- CUTLASS 3.5+
- PyTorch 2.4+ (for `fused=True` AdamW and CUDA Graphs)
- bitsandbytes 0.43+

Earlier versions may work for the first 7 patches (no Blackwell-specific
PTX), but Patches 8-10 require NVCC 12.8+ and a Blackwell-class GPU
(sm_120+).

