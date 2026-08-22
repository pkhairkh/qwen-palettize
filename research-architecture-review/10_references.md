# 10 — References: Frameworks, Tools, and Further Reading

## 1. Purpose

This document is the consolidated reference list for the architecture review. It includes every framework, tool, paper, and resource cited in documents `00`–`09`, organized by category, with URLs and a one-line description. The goal is to provide a single page that a developer can bookmark when starting the refactoring work in `09_refactoring_roadmap.md`.

---

## 2. LLM training frameworks

### 2.1 HuggingFace Transformers (`Trainer`)

- **Repo**: <https://github.com/huggingface/transformers>
- **Docs**: <https://huggingface.co/docs/transformers/main/en/main_classes/trainer>
- **API**: `transformers.Trainer`, `transformers.TrainingArguments`, `transformers.TrainerCallback`
- **License**: Apache 2.0
- **Used in this review**: `08_literature_comparison.md` §2. The pattern for the proposed `Config` dataclass, `Trainer` orchestrator, multi-backend logging, and `--gradient_checkpointing` flag.

### 2.2 PyTorch Lightning

- **Repo**: <https://github.com/Lightning-AI/pytorch-lightning>
- **Docs**: <https://lightning.ai/docs/pytorch/stable/>
- **API**: `lightning.pytorch.LightningModule`, `lightning.pytorch.Trainer`, `lightning.pytorch.callbacks.ModelCheckpoint`, `lightning.pytorch.callbacks.EarlyStopping`
- **License**: Apache 2.0
- **Used in this review**: `08_literature_comparison.md` §3. The pattern for the separation of concerns (`__init__/forward/training_step/configure_optimizers`).

### 2.3 Megatron-LM (NVIDIA)

- **Repo**: <https://github.com/NVIDIA/Megatron-LM>
- **Docs**: <https://github.com/NVIDIA/Megatron-LM/blob/main/README.md>
- **Key file**: `megatron/data/indexed_dataset.py` (the `MMapIndexedDataset` pre-tokenized binary format)
- **License**: Apache 2.0
- **Used in this review**: `08_literature_comparison.md` §4. The pattern for pre-tokenized binary data and forward-overlap-backward.

### 2.4 DeepSpeed (Microsoft)

- **Repo**: <https://github.com/microsoft/DeepSpeed>
- **Docs**: <https://www.deepspeed.ai/>
- **API**: `deepspeed.initialize`, `deepspeed.runtime.engine.Engine`, `deepspeed.runtime.config.DeepSpeedConfig`
- **License**: Apache 2.0
- **Used in this review**: `08_literature_comparison.md` §5. The reference for ZeRO optimizer state sharding and the canonical mixed-precision pattern.

### 2.5 lit-GPT (Karpathy / Lightning AI)

- **Repo**: <https://github.com/Lightning-AI/lit-GPT>
- **Docs**: <https://lightning.ai/docs/lit-gpt/stable/>
- **Key file**: `lit_gpt/pretrain.py` (the training loop), `lit_gpt/pretrain.py::pretokenize` (the binary data format)
- **License**: Apache 2.0
- **Used in this review**: `08_literature_comparison.md` §6. The closest analog to qwen-palettize in scope; the pattern for the proposed `qwen_palettize/` package.

### 2.6 nanoGPT (Karpathy)

- **Repo**: <https://github.com/karpathy/nanoGPT>
- **Docs**: <https://github.com/karpathy/nanoGPT/blob/master/README.md>
- **Key file**: `train.py` (~300 LOC)
- **License**: MIT
- **Used in this review**: `08_literature_comparison.md` §7. The minimum-viable LLM training framework; the "use the framework's primitives, don't reinvent" principle.

### 2.7 Accelerate (HuggingFace)

- **Repo**: <https://github.com/huggingface/accelerate>
- **Docs**: <https://huggingface.co/docs/accelerate>
- **API**: `accelerate.Accelerator`, `accelerate.Accelerator.prepare`, `accelerate.Accelerator.backward`
- **License**: Apache 2.0
- **Used in this review**: `02_partial_wrapper_problem.md` §4.6. The framework that calls `isinstance(model, nn.Module)` and would reject the current `PartialWrapper`.

### 2.8 bitsandbytes (8-bit optimizers)

- **Repo**: <https://github.com/TimDettmers/bitsandbytes>
- **Docs**: <https://huggingface.co/docs/bitsandbytes>
- **API**: `bitsandbytes.optim.Adam8bit`, `bitsandbytes.optim.AdamW8bit`
- **License**: MIT
- **Used in this review**: `03_memory_waste_analysis.md` §9. The 8-bit optimizer for `index_logits` that saves ~14 GB of VRAM.

---

## 3. Quantization & palettization papers

### 3.1 GPTQ

- **Paper**: "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (Frantar et al., ICLR 2023)
- **arXiv**: <https://arxiv.org/abs/2210.17323>
- **Repo**: <https://github.com/IST-DASLab/gptq>
- **License**: Apache 2.0
- **Used in this review**: `00_executive_summary.md` (referenced as the calibration method in `calib_qwen.py`).

### 3.2 LLT (Learnable Lookup Table)

- **Paper**: "LLT: Learnable Lookup Table for Efficient Image Super-Resolution" (CVPR 2022)
- **arXiv**: <https://arxiv.org/abs/2203.04350>
- **Repo**: <https://github.com/SYSU-SAIL/LLT>
- **Used in this review**: `SPEC.md` §8 (referenced).

### 3.3 FLUTE (LUT matmul kernel)

- **Paper**: "FLUTE: Pooled Vector Quantization for LLMs" (Ryu et al., 2024)
- **arXiv**: <https://arxiv.org/abs/2407.10960>
- **Repo**: <https://github.com/IST-DASLab/flute>
- **Used in this review**: `SPEC.md` §8 (referenced as the LUT matmul kernel reference).

### 3.4 GSQ (Gumbel-Softmax Quantization)

- **Paper**: "GSQ: Gumbel-Softmax Quantization for LLMs" (2024)
- **arXiv**: <https://arxiv.org/abs/2604.18556> (per SPEC §8; URL may not be live)
- **Repo**: <https://github.com/IST-DASLab/GSQ>
- **Used in this review**: `SPEC.md` §8. The reference for trainable indices via Gumbel-Softmax.

### 3.5 Gumbel-Softmax (Jang et al.)

- **Paper**: "Categorical Reparameterization with Gumbel-Softmax" (Jang, Gu, Poole, ICLR 2017)
- **arXiv**: <https://arxiv.org/abs/1611.01144>
- **Used in this review**: `SPEC.md` §8. The foundational paper for the trainable indices mechanism.

### 3.6 LSQ (Learned Step Size Quantization)

- **Paper**: "Learned Step Size Quantization" (Esser, McKinstry, Bhalgat, Modha, ICLR 2020)
- **arXiv**: <https://arxiv.org/abs/1902.08153>
- **Used in this review**: `SPEC.md` §8. The reference for learnable quantization parameters.

### 3.7 Nagel et al. (QAT oscillation freezing)

- **Paper**: "Overcoming Oscillations in Quantization-Aware Training" (Nagel, Fournarakis, Hegde, Markovic, Khudia, ICLR 2022)
- **arXiv**: <https://arxiv.org/abs/2110.04412>
- **Used in this review**: `train_qwen.py:246` (the `freeze_settled_palettes` function cites this paper in its docstring).

### 3.8 LoftQ (LoRA initialization for quantized models)

- **Paper**: "LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models" (Li et al., ICLR 2024)
- **arXiv**: <https://arxiv.org/abs/2310.08615>
- **Repo**: <https://github.com/yxuansu/LoRA_FineTuning_Quantized_LLMs>
- **Used in this review**: `qwen_model.py:209` (the `init="loftq"` argument to `QwenLoRA`).

---

## 4. PyTorch subsystems

### 4.1 `torch.utils.checkpoint`

- **Docs**: <https://pytorch.org/docs/stable/checkpoint.html>
- **API**: `torch.utils.checkpoint.checkpoint(function, *args, use_reentrant=True)`
- **Used in this review**: `02_partial_wrapper_problem.md` §4.2, `03_memory_waste_analysis.md` §7. The gradient checkpointing API; requires `use_reentrant=False` for compatibility with the soft kernel's `autograd.Function`.

### 4.2 `torch.autograd.Function`

- **Docs**: <https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function>
- **API**: `class MyFunction(torch.autograd.Function): forward(...) / backward(...)`
- **Used in this review**: `fused_lut_linear_cuda.py:709` (the `CUDAFusedLUTLinear` and `CUDAFusedLUTLinearSoft` classes).

### 4.3 `torch.compile`

- **Docs**: <https://pytorch.org/docs/stable/generated/torch.compile.html>
- **API**: `torch.compile(model, mode="reduce-overhead")`
- **Used in this review**: `02_partial_wrapper_problem.md` §4.1, `05_training_loop_refactor.md` §4.1. The 1.5–2× free speedup, blocked by the current `PartialWrapper`.

### 4.4 `torch.utils.data.DataLoader`

- **Docs**: <https://pytorch.org/docs/stable/data.html>
- **API**: `torch.utils.data.DataLoader(dataset, batch_size=, shuffle=, num_workers=, pin_memory=, prefetch_factor=)`
- **Used in this review**: `04_data_pipeline.md` §3.3 (Stage C). The standard async data loader with worker processes.

### 4.5 `torch.distributed.fsdp.FullyShardedDataParallel`

- **Docs**: <https://pytorch.org/docs/stable/fsdp.html>
- **API**: `torch.distributed.fsdp.FullyShardedDataParallel(model, ...)`
- **Used in this review**: `02_partial_wrapper_problem.md` §4.5. The FSDP wrapper, blocked by the current `PartialWrapper`.

### 4.6 `torch.utils.tensorboard.SummaryWriter`

- **Docs**: <https://pytorch.org/docs/stable/tensorboard.html>
- **API**: `SummaryWriter(log_dir=)`, `add_scalar(name, value, step)`
- **Used in this review**: `06_logging_metrics.md` §6. The TensorBoard backend for the proposed `MetricsLogger`.

### 4.7 `torch.amp.autocast` and `torch.amp.GradScaler`

- **Docs**: <https://pytorch.org/docs/stable/amp.html>
- **API**: `torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)`, `torch.amp.GradScaler`
- **Used in this review**: `train_qwen.py:1085` (the current autocast usage), `05_training_loop_refactor.md` §4.1 (the proposed usage in the `Trainer` class).

---

## 5. Data and tokenization

### 5.1 FineWeb-Edu

- **Dataset**: <https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu>
- **Subset**: `sample-10BT` (10 billion tokens)
- **Used in this review**: `train_qwen.py:898` (the streaming data source), `04_data_pipeline.md` throughout.

### 5.2 HuggingFace `datasets`

- **Repo**: <https://github.com/huggingface/datasets>
- **Docs**: <https://huggingface.co/docs/datasets>
- **API**: `datasets.load_dataset(name, split=, streaming=)`, `datasets.IterableDataset`
- **License**: Apache 2.0
- **Used in this review**: `train_qwen.py:898`. The streaming data loader.

### 5.3 HuggingFace Tokenizers

- **Repo**: <https://github.com/huggingface/tokenizers>
- **Docs**: <https://huggingface.co/docs/tokenizers>
- **API**: `tokenizers.Tokenizer`, `transformers.AutoTokenizer.from_pretrained`
- **License**: Apache 2.0
- **Used in this review**: `train_qwen.py:905` (the per-step tokenization call).

---

## 6. Logging and observability

### 6.1 Weights & Biases (wandb)

- **Website**: <https://wandb.ai>
- **Docs**: <https://docs.wandb.ai>
- **API**: `wandb.init(project=, name=, config=)`, `wandb.log(metrics, step=)`, `wandb.Artifact(name, type=, metadata=)`
- **License**: Open source (with paid tiers for teams)
- **Used in this review**: `06_logging_metrics.md` §5. The primary logging backend for the proposed `MetricsLogger`.

### 6.2 TensorBoard

- **Repo**: <https://github.com/tensorflow/tensorboard>
- **Docs**: <https://www.tensorflow.org/tensorboard>
- **API**: `torch.utils.tensorboard.SummaryWriter`, `tensorboard --logdir=`
- **License**: Apache 2.0
- **Used in this review**: `06_logging_metrics.md` §6. The fallback logging backend (no account required).

### 6.3 MLflow

- **Repo**: <https://github.com/mlflow/mlflow>
- **Docs**: <https://mlflow.org>
- **API**: `mlflow.log_metric(key, value, step=)`, `mlflow.log_artifact(path)`
- **License**: Apache 2.0
- **Used in this review**: Not used in this review (mentioned as a third option in `06_logging_metrics.md` §1.1).

---

## 7. Configuration and experiment management

### 7.1 Hydra (Facebook Research)

- **Repo**: <https://github.com/facebookresearch/hydra>
- **Docs**: <https://hydra.cc>
- **API**: `@hydra.main(config_path=, config_name=)`, `hydra.utils.instantiate`
- **License**: Apache 2.0
- **Used in this review**: `00_executive_summary.md` §9 (issue #9, proposed as a fix for the three-sources-of-truth configuration chaos).

### 7.2 OmegaConf

- **Repo**: <https://github.com/omry/omegaconf>
- **Docs**: <https://omegaconf.readthedocs.io>
- **API**: `OmegaConf.create(yaml)`, `OmegaConf.merge(a, b)`
- **License**: BSD 3-Clause
- **Used in this review**: Implicit in the Hydra recommendation; OmegaConf is the underlying config library.

### 7.3 DVC (Data Version Control)

- **Repo**: <https://github.com/iterative/dvc>
- **Docs**: <https://dvc.org>
- **API**: `dvc init`, `dvc add data/`, `dvc.yaml` for pipeline definition
- **License**: Apache 2.0
- **Used in this review**: Not used directly, but mentioned in `01_architecture_audit.md` §11 as the alternative to a manual pipeline orchestration.

---

## 8. Testing and CI/CD

### 8.1 pytest

- **Repo**: <https://github.com/pytest-dev/pytest>
- **Docs**: <https://docs.pytest.org>
- **API**: `def test_*():`, `@pytest.fixture`, `@pytest.mark.regression`, `pytest.ini`
- **License**: MIT
- **Used in this review**: `07_testing_ci.md` throughout. The test runner for the proposed 4-tier test scaffold.

### 8.2 pytest-cov

- **Repo**: <https://github.com/pytest-dev/pytest-cov>
- **Docs**: <https://pytest-cov.readthedocs.io>
- **API**: `pytest --cov=qwen_palettize --cov-report=term-missing`
- **License**: MIT
- **Used in this review**: `07_testing_ci.md` §8. Coverage measurement.

### 8.3 GitHub Actions

- **Docs**: <https://docs.github.com/en/actions>
- **API**: `.github/workflows/*.yml`, `runs-on:`, `steps:`, `schedule: cron`
- **License**: Free for public repositories
- **Used in this review**: `07_testing_ci.md` §7.1. The CI/CD pipeline for the proposed test scaffold.

### 8.4 pre-commit

- **Repo**: <https://github.com/pre-commit/pre-commit>
- **Docs**: <https://pre-commit.com>
- **API**: `.pre-commit-config.yaml`, `pre-commit install`
- **License**: MIT
- **Used in this review**: `07_testing_ci.md` §7.3. The pre-commit hook for Tier 1 tests.

---

## 9. Optimizers

### 9.1 Muon (Newton-Schulz orthogonalized momentum)

- **Reference**: Karpathy's `llm.c` project, the "Muon" optimizer variant
- **Repo**: <https://github.com/KellerJordan/Muon>
- **License**: MIT
- **Used in this review**: `train_qwen.py:106` (the `Muon` class). The optimizer for 2D non-palette weights (LoRA B matrices).

### 9.2 AdamW (PyTorch built-in)

- **Docs**: <https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html>
- **API**: `torch.optim.AdamW(params, lr=, betas=, eps=, weight_decay=)`
- **Used in this review**: `train_qwen.py:208` (the `FP32MasterAdamW` wraps this). The optimizer for palettes, LoRA A matrices, index_logits, and 1D params.

### 9.3 bitsandbytes 8-bit AdamW

- **Docs**: <https://huggingface.co/docs/bitsandbytes/main/en/optimizers>
- **API**: `bitsandbytes.optim.AdamW8bit(params, lr=, ...)`
- **Used in this review**: `03_memory_waste_analysis.md` §9. The 8-bit alternative that saves ~14 GB for `index_logits`.

---

## 10. CUDA and GPU tools

### 10.1 PyTorch C++ Extensions (`torch.utils.cpp_extension`)

- **Docs**: <https://pytorch.org/docs/stable/cpp_extension.html>
- **API**: `torch.utils.cpp_extension.load_inline(name, cpp_sources, cuda_sources, ...)`
- **Used in this review**: `fused_lut_linear_cuda.py:30` (the kernel compilation entry point).

### 10.2 NVIDIA Nsight Systems

- **Docs**: <https://docs.nvidia.com/nsight-systems/>
- **License**: Free
- **Used in this review**: Mentioned in `01_architecture_audit.md` §11 as the recommended profiler for CUDA kernel optimization.

### 10.3 pynvml (NVIDIA Management Library Python bindings)

- **Repo**: <https://github.com/gpuopenanalytics/pynvml>
- **Docs**: <https://pynvml.readthedocs.io>
- **API**: `pynvml.nvmlInit()`, `pynvml.nvmlDeviceGetUtilizationRates(handle)`
- **License**: BSD
- **Used in this review**: Implicit in the `torch.cuda.utilization()` call at `train_qwen.py:1182`.

---

## 11. Internal references (within the qwen-palettize repo)

### 11.1 Source files

| File | LOC | Role |
|---|---|---|
| `scripts/train_qwen.py` | 1,266 | The monolithic training loop (target of Phase 3 refactor) |
| `scripts/qwen_model.py` | 834 | `PalettizedLinear`, `QwenLoRA`, `PartialWrapper` (target of Phase 1 fix) |
| `scripts/fused_lut_linear_cuda.py` | 709 | CUDA kernel wrappers |
| `scripts/fused_lut_kernel.cu` | — | The CUDA kernels themselves |
| `scripts/palettize_core.py` | 178 | 2-bit packing and metadata |
| `scripts/palettize_pytorch.py` | 207 | Reference kmeans and packing |
| `scripts/calib_qwen.py` | 365 | Stage 1 calibration |
| `scripts/calib_stage2.py` | 362 | (Legacy) Stage 2 palettization |
| `scripts/convert_trained_to_packed.py` | 120 | One-off migration script |
| `scripts/cache_tokens.py` | 48 | Tiny utility for eval token caching |
| `scripts/verify_palettize_core.py` | 167 | Manual smoke test |
| `scripts/verify_2bit_format.py` | 198 | Manual smoke test |
| `scripts/test_palettized_v2.py` | 213 | Manual smoke test (not a real pytest test) |
| `scripts/bench_palettized_v2.py` | 113 | Microbenchmark |
| `scripts/profile_training.py` | 255 | Profiling script |
| `scripts/profile_nosync.py` | 196 | Profiling without sync |
| `scripts/sweep_qwen.py` | 313 | Hyperparameter sweep (parses `print()` output) |
| `scripts/dry_run_counts.py` | 60 | Param count utility |

### 11.2 Spec and logs

- `SPEC.md` — the project specification (referenced throughout).
- `logs/train_sb0.log` — the training log (referenced in `00_executive_summary.md`, `01_architecture_audit.md`, `03_memory_waste_analysis.md`).
- `logs/calib_sb0.log` — the calibration log (not referenced in detail in this review, but mentioned in `01_architecture_audit.md` §11).

### 11.3 Binary artifacts

- `cached_tokens.pt` (516 KB) — orphaned, should be deleted (see `04_data_pipeline.md` §4.3).
- `eval_tokens.pt` (1.1 MB) — the eval set cache.
- `trained/superblock_0_best/` (232 MB) — 140+ `.pt` files, the saved checkpoint.
- `trained/superblock_0_safe_backup/` — backup checkpoint.
- `palettized/superblock_0/` (107 MB) — the Stage 1 calibration output.

---

## 12. Review documents (this series)

| Doc | Title | Words |
|---|---|---|
| `00_executive_summary.md` | Top 10 systemic issues, severity-ranked | ~2,400 |
| `01_architecture_audit.md` | Current system diagram + component analysis | ~4,900 |
| `02_partial_wrapper_problem.md` | Why `PartialWrapper` not being `nn.Module` breaks everything | ~3,700 |
| `03_memory_waste_analysis.md` | Teacher/student duplication, no checkpointing, GB budget | ~4,300 |
| `04_data_pipeline.md` | Streaming vs prefetch vs cached | ~4,000 |
| `05_training_loop_refactor.md` | Monolithic → modular proposal | ~4,000 |
| `06_logging_metrics.md` | wandb/tensorboard/csv/structured output | ~3,200 |
| `07_testing_ci.md` | pytest, regression tests, CI/CD | ~3,200 |
| `08_literature_comparison.md` | HF Trainer, Lightning, Megatron, DeepSpeed, lit-GPT, nanoGPT | ~3,700 |
| `09_refactoring_roadmap.md` | Phased plan with time estimates | ~3,500 |
| `10_references.md` | This document | ~1,500 |

**Total**: ~39,400 words, ~112 pages (at 350 words/page).

---

## 13. Closing

This reference list is intentionally curated, not exhaustive. Every link is one that the architecture review found directly relevant to the qwen-palettize codebase or to the refactoring proposals in documents `05`–`09`. A developer starting the refactoring work should bookmark this page, read the linked docs for the patterns they are adopting (HF Trainer for `Config`, Lightning for `Trainer`, lit-GPT for `data.py`, etc.), and proceed with the phased plan in `09_refactoring_roadmap.md`.

For papers (§3), the recommended reading order is:

1. Jang et al. (Gumbel-Softmax) — foundational; understand the temperature annealing.
2. Nagel et al. (QAT oscillation freezing) — understand the `freeze_settled_palettes` function.
3. LoftQ (Li et al.) — understand the `init="loftq"` argument to `QwenLoRA`.
4. GPTQ (Frantar et al.) — understand the calibration phase.
5. FLUTE (Ryu et al.) — understand the LUT matmul kernel design.

For frameworks (§2), the recommended reading order is:

1. lit-GPT `pretrain.py` — the closest analog; read the full file (~500 LOC).
2. HF `Trainer` docs — the pattern reference.
3. Lightning `LightningModule` docs — the separation-of-concerns pattern.
4. Megatron `MMapIndexedDataset` — the data format reference.
5. DeepSpeed ZeRO docs — the memory-optimization reference (for future, not current, work).

The architecture review is complete. The roadmap in `09_refactoring_roadmap.md` is ready to execute.
