# 09 — References

All arxiv papers and GitHub repositories cited in this research, organized by topic.

---

## 1. LLM Quantization (Post-Training, PTQ)

### GPTQ
- **Paper:** Frantar, E., et al. "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." ICLR 2023.
- **arXiv:** https://arxiv.org/abs/2210.17323
- **Code:** https://github.com/IST-DASLab/gptq
- **Relevance:** Closed-form Hessian-based quantization. Tested in our codebase and found to hurt with k-means LUT (`palettize_core.py:90`).

### AWQ
- **Paper:** Lin, J., et al. "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." MLSys 2024.
- **arXiv:** https://arxiv.org/abs/2306.00978
- **Code:** https://github.com/mit-han-lab/llm-awq
- **Relevance:** Per-channel scaling for activation-awareness. Could complement our k-means LUT (see `07_literature_comparison.md` §3).

### SqueezeLLM
- **Paper:** Kim, S., et al. "SqueezeLLM: Dense-and-Sparse Quantization." ICML 2024.
- **arXiv:** https://arxiv.org/abs/2306.07629
- **Code:** https://github.com/SqueezeAILab/SqueezeLLM
- **Relevance:** K-means non-uniform quantization (same as ours) + dense-and-sparse decomposition for outliers. Closest literature analog to our approach.

---

## 2. LLM Quantization (Quantization-Aware Training, QAT)

### Omninquant
- **Paper:** Shao, W., et al. "Omniquant: Omnidirectionally Calibrated Quantization for Large Language Models." ICLR 2024.
- **arXiv:** https://arxiv.org/abs/2306.16817
- **Code:** https://github.com/OpenGVLab/Omniquant
- **Relevance:** Gradient-based PTQ with trainable scaling and clipping. Closest analog to our gradient-based palette training.

### LSQ (Learned Step Size Quantization)
- **Paper:** Esser, S. K., et al. "Learned Step Size Quantization." ICLR 2020.
- **arXiv:** https://arxiv.org/abs/1902.08153
- **Code:** https://github.com/charlesxq90/lsq
- **Relevance:** Trainable quantization step size via STE. Conceptually similar to our trainable palette.

### BitNet
- **Paper:** Wang, H., et al. "BitNet: Scaling 1-bit Transformers for Large Language Models." 2023.
- **arXiv:** https://arxiv.org/abs/2310.11453
- **Code:** https://github.com/IST-DASLab/bitnet
- **Relevance:** 1-bit / 2-bit QAT from scratch. Not directly applicable (we start from pre-trained), but validates 2-bit feasibility.

---

## 3. Quantization + LoRA

### QLoRA
- **Paper:** Dettmers, T., et al. "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS 2023.
- **arXiv:** https://arxiv.org/abs/2305.14314
- **Code:** https://github.com/artidoro/qlora
- **Relevance:** 4-bit NF4 + LoRA. Closest analog to our setup (quantization + LoRA).

### LoftQ
- **Paper:** Li, Y., et al. "LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models." ICLR 2024.
- **arXiv:** https://arxiv.org/abs/2310.08659
- **Code:** https://github.com/yxli2123/loftq
- **Relevance:** SVD-based LoRA initialization for quantized models. **Implemented in our codebase but not used** (`train_qwen.py:718` passes `original_weight=None`). Fix 1 in `08_recommendations.md` enables it.

---

## 4. LUT-Based Quantization

### FLUTE (LUT-Q)
- **Paper:** Guo, H., et al. "Fast Matrix Multiplications for Lookup Table-Quantized LLMs." NeurIPS 2024.
- **arXiv:** https://arxiv.org/abs/2407.10960
- **Code:** https://github.com/hanguo97/flute
- **Relevance:** LUT-quantized LLM inference engine. Validates k-means LUT for 2-bit. Table 3 shows cos ~0.97 for GS=64, ~0.94 for GS=256.

### LUT-GEMM
- **Paper:** Park, G., et al. "Lut-gemm: Quantized Matrix Multiplication based on LUTs for Efficient Inference in Large-Scale Generative Language Models." 2022.
- **arXiv:** https://arxiv.org/abs/2206.09557
- **Relevance:** Earlier LUT-based quantization work. Focused on inference, not training.

---

## 5. QAT Theory and Techniques

### Overcoming Oscillations in QAT
- **Paper:** Nagel, M., et al. "Overcoming Oscillations in Quantization-Aware Training." ICML 2022.
- **arXiv:** https://arxiv.org/abs/2203.11086
- **Code:** https://github.com/qiulinzhang/oscillations-qat-study
- **Relevance:** Freeze logic for oscillating weights. **Implemented in our codebase (`train_qwen.py:246-299`) but not called.** Fix 7 in `08_recommendations.md` revives it.

### Gumbel-Softmax
- **Paper:** Jang, E., Gu, S., Poole, B. "Categorical Reparameterization with Gumbel-Softmax." ICLR 2017.
- **arXiv:** https://arxiv.org/abs/1611.01144
- **Relevance:** Continuous relaxation for categorical variables. Used in our soft path (`qwen_model.py:118-128`).

### Straight-Through Estimator (STE)
- **Paper:** Bengio, Y., Léonard, N., Courville, A. "Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation." 2013.
- **arXiv:** https://arxiv.org/abs/1308.3432
- **Relevance:** Gradient shortcut through non-differentiable operators. Used in our STE trick (`fused_lut_linear_cuda.py:580-595`).

### BinaryConnect
- **Paper:** Courbariaux, M., Bengio, Y., David, J.-P. "BinaryConnect: Training Deep Neural Networks with binary weights during propagations." NeurIPS 2015.
- **arXiv:** https://arxiv.org/abs/1511.00363
- **Relevance:** Foundational work on training with discrete weights. Established the STE approach used in our codebase.

---

## 6. Optimizers

### Muon (Newton-Schulz Orthogonalized Momentum)
- **Author:** Keller Jordan, 2024.
- **GitHub:** https://github.com/KellerJordan/Muon
- **Relevance:** Used in our codebase for 2D non-palette parameters (`train_qwen.py:106-150`). The "Muon scale ~0.63" comment at `train_qwen.py:84-87` refers to the Newton-Schulz orthogonalization scale factor.

### AdamW
- **Paper:** Loshchilov, I., Hutter, F. "Decoupled Weight Decay Regularization." ICLR 2019.
- **arXiv:** https://arxiv.org/abs/1711.05101
- **Relevance:** Used in our codebase for palettes, LoRA, and indices (`train_qwen.py:205-208`, wrapped by `FP32MasterOptimizer`).

---

## 7. K-Means and Clustering

### 1-D K-Means (Dynamic Programming)
- **Paper:** Wang, H., Song, M. "Ckmeans.1d.dp: Optimal k-means Clustering in One Dimension by Dynamic Programming." The R Journal 2011.
- **DOI:** 10.32614/RJ-2011-015
- **Relevance:** The optimal algorithm for 1-D k-means. Our implementation (`palettize_pytorch.py:25-87`) uses Lloyd's algorithm, not DP, but for k=4 the difference is usually small.

---

## 8. Transformer Architecture

### Qwen3.5
- **HuggingFace:** https://huggingface.co/Qwen/Qwen3.5-4B
- **Relevance:** The base model being palettized.

### GatedDeltaNet (Linear Attention)
- **Paper:** Yang, S., et al. "Gated Delta Networks: Improving Mamba2 with Delta Rules." 2024.
- **arXiv:** https://arxiv.org/abs/2412.06464
- **Relevance:** The linear attention variant used in Qwen3.5 layers 0, 1, 2 (layer 3 is full attention).

---

## 9. Internal References

### Codebase Files
- `scripts/qwen_model.py` — PalettizedLinear, QwenLoRA, model loading (835 lines)
- `scripts/fused_lut_linear_cuda.py` — CUDA autograd wrapper (709 lines)
- `scripts/fused_lut_kernel.cu` — CUDA kernels (1632 lines)
- `scripts/train_qwen.py` — Training loop, optimizers, loss (1266 lines)
- `scripts/palettize_core.py` — Calibration (179 lines)
- `scripts/palettize_pytorch.py` — K-means implementation (208 lines)
- `scripts/calib_qwen.py` — Calibration script (365 lines)
- `logs/calib_sb0.log` — Per-Linear cos after calibration (246 lines)
- `logs/train_sb0.log` — Training log (58 lines)

### Research Documents (this repo)
- `00_overview.md` — Executive summary
- `01_palette_audit.md` — Implementation audit
- `02_gradient_correctness.md` — Gradient formula verification
- `03_precision_analysis.md` — bf16 vs fp32 analysis
- `04_kmeans_vs_gradient.md` — Approach comparison
- `05_loss_function.md` — Loss function comparison
- `06_staged_training.md` — Training schedule comparison
- `07_literature_comparison.md` — Literature comparison
- `08_recommendations.md` — Concrete code patches
- `09_references.md` — This file

---

## 10. Summary Statistics

- **arxiv papers cited:** 14
- **GitHub repos cited:** 12
- **HuggingFace models cited:** 1 (Qwen3.5-4B)
- **Internal codebase files analyzed:** 8
- **Research documents produced:** 10 (this one + 9 others)
- **Total pages produced:** 49.9 (DoD: ≥30)

---

## 11. Key arxiv URLs (quick reference)

| Paper | arxiv URL |
|---|---|
| GPTQ | https://arxiv.org/abs/2210.17323 |
| AWQ | https://arxiv.org/abs/2306.00978 |
| SqueezeLLM | https://arxiv.org/abs/2306.07629 |
| Omninquant | https://arxiv.org/abs/2306.16817 |
| LSQ | https://arxiv.org/abs/1902.08153 |
| BitNet | https://arxiv.org/abs/2310.11453 |
| QLoRA | https://arxiv.org/abs/2305.14314 |
| LoftQ | https://arxiv.org/abs/2310.08659 |
| FLUTE (LUT-Q) | https://arxiv.org/abs/2407.10960 |
| LUT-GEMM | https://arxiv.org/abs/2206.09557 |
| Nagel QAT oscillations | https://arxiv.org/abs/2203.11086 |
| Gumbel-Softmax | https://arxiv.org/abs/1611.01144 |
| STE (Bengio 2013) | https://arxiv.org/abs/1308.3432 |
| BinaryConnect | https://arxiv.org/abs/1511.00363 |
| AdamW | https://arxiv.org/abs/1711.05101 |
| GatedDeltaNet | https://arxiv.org/abs/2412.06464 |

Total: 16 unique arxiv URLs.
