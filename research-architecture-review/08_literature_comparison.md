# 08 — Literature & Framework Comparison

## 1. Purpose

This document compares the qwen-palettize architecture to six mainstream LLM-training frameworks. The goal is not to crown a winner — the comparison is between frameworks designed for general LLM training and a codebase specialized for 2-bit LUT palettization. The goal is to extract the architectural patterns each framework uses to solve problems the qwen-palettize codebase currently handles poorly (or not at all): modularity, distributed training, mixed precision, data loading, logging, configuration, checkpointing, and testing.

The six frameworks are:

1. **HuggingFace `transformers.Trainer`** — the most popular LLM training framework, used by 50,000+ models on the HF Hub.
2. **PyTorch Lightning** — a lightweight framework that emphasizes separation of concerns.
3. **Megatron-LM (NVIDIA)** — the production-grade framework for training 100B+ parameter models.
4. **DeepSpeed (Microsoft)** — the framework that powers the ZeRO optimizers and 3D parallelism.
5. **lit-GPT (Karpathy)** — a minimalist framework designed for education and small-team research.
6. **nanoGPT (Karpathy)** — the even-smaller predecessor, ~300 LOC of training loop.

For each framework, we describe: (a) the architectural pattern, (b) what qwen-palettize could adopt, (c) what qwen-palettize should NOT adopt (because it would be over-engineering for a 4B-parameter single-GPU research project).

---

## 2. HuggingFace `transformers.Trainer`

**Repository**: <https://github.com/huggingface/transformers>
**Docs**: <https://huggingface.co/docs/transformers/main/en/main_classes/trainer>
**LOC**: ~3,500 (the `Trainer` class itself, in `src/transformers/trainer.py`)
**Users**: 50,000+ models, ~1M weekly downloads

### 2.1 Architectural pattern

`Trainer` is a single class that holds the model, optimizer, scheduler, data loader, and a `TrainingArguments` config object. The training loop is a method called `training_step`, which the user can override in a subclass. The framework provides:

- **`TrainingArguments`** — a 200-field dataclass with every hyperparameter the framework supports, validated by `transformers.HfArgumentParser`.
- **`Trainer`** — the orchestrator class. The user subclasses it (or uses it as-is) and overrides `compute_loss`, `training_step`, `evaluate`, `save_model`, etc.
- **`TrainingCallback`** — a callback system for early stopping, logging, checkpointing, custom behavior.
- **Integrations** — automatic W&B, TensorBoard, MLflow, AzureML, ClearML, CometML logging via the `report_to` argument. The user sets `report_to="wandb,tensorboard"` and gets both backends for free.
- **Distributed training** — built-in DDP, FSDP, DeepSpeed integration. The user sets `--ddp_find_unused_parameters` or `--fsdp_config` and gets distributed training without writing any code.
- **Mixed precision** — `--bf16` or `--fp16` flags; the framework handles `autocast`, `GradScaler`, and the dtype conversion.
- **Gradient accumulation** — `--gradient_accumulation_steps N`; the framework accumulates gradients across N micro-batches before stepping the optimizer.
- **Gradient checkpointing** — `--gradient_checkpointing`; the framework calls `model.gradient_checkpointing_enable()` and handles the recomputation.
- **Checkpointing** — `--save_strategy steps --save_steps N`; the framework saves a checkpoint every N steps, with a `save_total_limit` for rotation.
- **Resume** — `--resume_from_checkpoint path`; the framework loads the checkpoint, the optimizer state, the scheduler state, and the RNG state.

### 2.2 What qwen-palettize could adopt

| Pattern | Source in HF Trainer | Analogous qwen-palettize fix |
|---|---|---|
| `TrainingArguments` dataclass | `TrainingArguments` class | `qwen_palettize.config.Config` (proposed in `05_training_loop_refactor.md` §3) |
| `Trainer` orchestrator class | `Trainer.training_step` method | `qwen_palettize.train.Trainer` (proposed in `05_training_loop_refactor.md` §4) |
| `TrainingCallback` system | `TrainerCallback` class | A future `TrainerCallback` for qwen-palettize (early stopping, custom logging) |
| `report_to="wandb,tensorboard"` | `integrations` package | `MetricsLogger` multi-backend (proposed in `06_logging_metrics.md` §2) |
| `--gradient_checkpointing` | `model.gradient_checkpointing_enable()` | `use_gradient_checkpointing` flag (proposed in `05` §4.1) |
| `--save_strategy steps` | `Trainer._save_checkpoint` | `save_state` with `save_every` (already implemented, but could be a callback) |
| `--resume_from_checkpoint` | `Trainer._load_from_checkpoint` | `load_state` (already implemented) |

The HF `Trainer` is the **single best reference** for the qwen-palettize refactor. The proposed `Trainer` class in `05_training_loop_refactor.md` is a stripped-down version of the HF `Trainer`, omitting the distributed-training, gradient-accumulation, and mixed-precision features that qwen-palettize does not need (yet).

### 2.3 What qwen-palettize should NOT adopt

- **The 200-field `TrainingArguments` dataclass.** qwen-palettize has ~25 hyperparameters. The full `TrainingArguments` is overkill and would obscure the configuration.
- **The DeepSpeed / FSDP integration.** qwen-palettize is single-GPU. Adopting DeepSpeed's config schema would be premature.
- **The `Trainer.evaluate()` method with its `compute_metrics` callback.** qwen-palettize's evaluation is a simple cosine similarity, not a multi-metric NLP eval.

### 2.4 Verdict

Adopt the **pattern** (dataclass config, orchestrator class, callback system, multi-backend logging), not the **scale**. The proposed refactor in `05_training_loop_refactor.md` does exactly this.

---

## 3. PyTorch Lightning

**Repository**: <https://github.com/Lightning-AI/pytorch-lightning>
**Docs**: <https://lightning.ai/docs/pytorch/stable/>
**LOC**: ~50,000 (the `lightning` package)
**Users**: ~10K research projects

### 3.1 Architectural pattern

Lightning's central abstraction is the `LightningModule`, which separates concerns into:

- `__init__` — model definition (the `nn.Module`s).
- `forward` — inference-only forward pass (used for production).
- `training_step` — one training step (forward + loss + backward, but NOT the optimizer step).
- `validation_step` — one validation step.
- `configure_optimizers` — returns the optimizer(s) and scheduler(s).

The `Trainer` (a separate class) handles:

- The training loop (calling `training_step`, the optimizer step, the scheduler step, the zero_grad).
- Distributed training (DDP, FSDP, DeepSpeed, model parallel).
- Mixed precision (`--precision bf16-true` or `--precision 16-mixed`).
- Gradient accumulation (`--accumulate_grad_batches N`).
- Gradient clipping (`--gradient_clip_val 0.3`).
- Checkpointing (via `ModelCheckpoint` callback).
- Early stopping (via `EarlyStopping` callback).
- Logging (via `log_every_n_steps`, with W&B, TensorBoard, CSV, MLflow backends).

The key insight: the **model** defines what to compute; the **trainer** defines how to compute it. The two are separate classes, allowing the same model to be trained with different strategies (single-GPU, DDP, FSDP) without changing the model code.

### 3.2 What qwen-palettize could adopt

| Pattern | Source in Lightning | Analogous qwen-palettize fix |
|---|---|---|
| `LightningModule` separation of concerns | `LightningModule` class | `Trainer.training_step` (proposed in `05` §4.1) — the proposed `Trainer` already separates `_train_step` from `_clip_and_step` |
| `configure_optimizers` method | `LightningModule.configure_optimizers` | `qwen_palettize.optim.build_optimizers` (proposed in `05` §6.2) |
| `ModelCheckpoint` callback | `lightning.pytorch.callbacks.ModelCheckpoint` | A future `CheckpointCallback` for qwen-palettize |
| `EarlyStopping` callback | `lightning.pytorch.callbacks.EarlyStopping` | A future `EarlyStoppingCallback` (e.g., stop if cos does not improve for 1000 steps) |
| `log_every_n_steps` arg | `Trainer(log_every_n_steps=50)` | `cfg.log_every` (already in the proposed `Config`) |

### 3.3 What qwen-palettize should NOT adopt

- **The full `LightningModule` abstraction.** qwen-palettize's `Trainer` is small enough that the Lightning's `__init__/forward/training_step/validation_step/configure_optimizers` split is over-engineered.
- **The `Trainer` class's 200+ arguments.** Same reasoning as HF Trainer.
- **The `Strategy` / `Accelerator` / `Precision` plugin system.** qwen-palettize is single-GPU, single-precision.

### 3.4 Verdict

Adopt the **separation of concerns** (model definition vs. training loop), not the full Lightning abstraction. The proposed `Trainer` class in `05` already implements this separation.

---

## 4. Megatron-LM (NVIDIA)

**Repository**: <https://github.com/NVIDIA/Megatron-LM>
**Docs**: <https://github.com/NVIDIA/Megatron-LM/blob/main/README.md>
**LOC**: ~50,000
**Users**: NVIDIA internal, + large research labs (Meta, Microsoft, Google)

### 4.1 Architectural pattern

Megatron-LM is the **production** framework for training 100B+ parameter models. Its key architectural innovations:

- **Tensor parallelism** — splitting each Linear's matmul across N GPUs, so a 100B model fits on 4 GPUs.
- **Pipeline parallelism** — splitting the layers across M GPUs, so a 100-layer model is split into 4 stages of 25 layers each.
- **Data parallelism** — replicating the model across K GPU groups, with gradient all-reduce.
- **3D parallelism** — combining tensor + pipeline + data parallelism.
- **Overlap of compute and communication** — the backward pass of layer N overlaps with the gradient all-reduce of layer N+1.
- **Fused kernels** — custom CUDA kernels for LayerNorm, GeLU, Softmax, etc., that fuse multiple operations into one kernel.
- **Transformer-specific optimizations** — flash attention, fused MLP, etc.

The training loop is in `megatron/training/training.py`, ~1,000 LOC, and is **monolithic** — much like the current qwen-palettize `train_super_block`. The modularity is in the **model** layer (each transformer block is a separate `nn.Module` with `tensor_model_parallel_*` constructors), not in the training loop.

### 4.2 What qwen-palettize could adopt

| Pattern | Source in Megatron-LM | Analogous qwen-palettize fix |
|---|---|---|
| Fused CUDA kernels for LayerNorm/GeLU/Softmax | `megatron/core/fused_kernels/` | Already done — the `fused_lut_kernel.cu` is a custom fused LUT matmul kernel |
| Pre-tokenized binary data format (`MMapIndexedDataset`) | `megatron/data/indexed_dataset.py` | The proposed `data_cache_dir` in `04_data_pipeline.md` §3.1 |
| Per-layer activation checkpointing | `megatron.models.transformer.ParallelTransformer` with `checkpoint_layers` | The proposed `use_gradient_checkpointing` flag in `05` §4.1 |
| Forward-overlap-backward via CUDA streams | `megatron.training.training.forward_backward_step` | Already done — the `stream_t` at `train_qwen.py:1060` overlaps teacher forward with student backward |

### 4.3 What qwen-palettize should NOT adopt

- **Tensor/pipeline/data parallelism.** qwen-palettize is single-GPU, and the palettization is per-super-block (not pipeline-friendly).
- **Fused kernels for LayerNorm/GeLU/Softmax.** These are HuggingFace transformer layers, not Megatron layers. HuggingFace already provides optimized implementations via `flash_attn` and `torch.nn.functional.scaled_dot_product_attention`.
- **The monolithic training loop.** qwen-palettize is moving away from this, toward the modular `Trainer` class.

### 4.4 Verdict

Adopt the **data format** (pre-tokenized binary) and the **forward-overlap-backward** pattern (already done). Skip the parallelism and the monolithic loop.

---

## 5. DeepSpeed (Microsoft)

**Repository**: <https://github.com/microsoft/DeepSpeed>
**Docs**: <https://www.deepspeed.ai/>
**LOC**: ~150,000
**Users**: Microsoft internal, + many research labs

### 5.1 Architectural pattern

DeepSpeed is the framework that powers the ZeRO (Zero Redundancy Optimizer) memory optimizations:

- **ZeRO Stage 1** — shard the optimizer state across N GPUs.
- **ZeRO Stage 2** — also shard the gradients.
- **ZeRO Stage 3** — also shard the model parameters.
- **ZeRO-Infinity** — offload to NVMe and CPU RAM, enabling 1T-parameter training on a single GPU.

DeepSpeed also provides:

- **3D parallelism** — combining ZeRO with tensor + pipeline parallelism.
- **MoE (Mixture of Experts)** support — for sparse expert models.
- **Long context support** — DeepSpeed-UltraConflong, 1M-token context.
- **Mixed precision** — `fp16`, `bf16`, and `fp8` (on H100).

The DeepSpeed `Engine` class wraps the user's model and intercepts the forward/backward calls to inject the ZeRO optimizations.

### 5.2 What qwen-palettize could adopt

| Pattern | Source in DeepSpeed | Analogous qwen-palettize fix |
|---|---|---|
| ZeRO Stage 1 (optimizer state sharding) | `deepspeed.runtime.engine.Engine` | Not applicable to single-GPU, but the principle (shard the optimizer state) could be applied to `index_logits`'s 21 GB of state via 8-bit Adam |
| 8-bit optimizers (`bnb.optim.Adam8bit`) | bitsandbytes library | The proposed fix in `03_memory_waste_analysis.md` §9 (saves ~14 GB) |
| Mixed precision with `bf16` params + `fp16` logits + `fp32` optimizer | `deepspeed.runtime.config.DeepSpeedConfig` | The current dtype chaos in qwen-palettize (issue #5) — DeepSpeed shows the canonical way to do this |

### 5.3 What qwen-palettize should NOT adopt

- **The full ZeRO Stage 3 / ZeRO-Infinity.** qwen-palettize is single-GPU and the super-block model fits in 41 GB. ZeRO would be overkill.
- **The MoE support.** qwen-palettize does not use MoE.
- **The DeepSpeed `Engine` class.** It wraps the user's model, intercepting forward/backward. This is incompatible with the `Trainer` pattern proposed in `05`. If qwen-palettize ever needs DeepSpeed, the `Trainer` class would need to delegate to DeepSpeed's `Engine`, but this is a future concern.

### 5.4 Verdict

Adopt the **8-bit optimizer** (via `bitsandbytes`) and the **canonical mixed-precision pattern** (bf16 params, fp16 logits, fp32 optimizer — but with a single master dtype, not the current chaos). Skip the ZeRO and the Engine class.

---

## 6. lit-GPT (Karpathy)

**Repository**: <https://github.com/Lightning-AI/lit-GPT>
**Docs**: <https://lightning.ai/docs/lit-gpt/stable/>
**LOC**: ~5,000
**Users**: ~5K research projects, educational use

### 6.1 Architectural pattern

lit-GPT is Andrej Karpathy's minimalist LLM-training framework, built on top of PyTorch Lightning. Its philosophy is:

- **Read the source code, don't read the docs.** Every file is < 500 LOC and is self-documenting.
- **Use the simplest possible abstraction.** No `TrainingArguments` dataclass (the CLI args are passed directly to a `setup` function). No callback system (the training loop is one function). No `Trainer` orchestrator (the training loop is in `lit_gpt.pretrain.train`, a single function).
- **Pre-tokenize to a binary file.** `lit_gpt.pretrain.pretokenize` writes a `train.bin` and `val.bin` of int32 tokens, mmap'd at training time.
- **Use PyTorch's `DataLoader` directly.** No custom data loader, no async prefetch — the standard `DataLoader` with `num_workers=4, pin_memory=True` is sufficient.
- **Use `torch.utils.checkpoint.checkpoint` directly.** No `gradient_checkpointing_enable()` abstraction — the user adds `torch.utils.checkpoint.checkpoint(layer, x)` in the forward pass.
- **Use W&B directly.** `wandb.init` in the training function, `wandb.log` in the loop.

### 6.2 What qwen-palettize could adopt

| Pattern | Source in lit-GPT | Analogous qwen-palettize fix |
|---|---|---|
| Pre-tokenize to a binary file | `lit_gpt.pretrain.pretokenize` | The proposed `cache_tokens_v2.py` in `04_data_pipeline.md` §6.4 |
| Standard `DataLoader` with `num_workers=4` | `lit_gpt.pretrain.train` | Stage C of the data pipeline (`04` §3.3) |
| Direct `torch.utils.checkpoint` | `lit_gpt.model.GPT.forward` | The proposed `use_gradient_checkpointing` flag in `05` §4.1 |
| Direct `wandb.init` | `lit_gpt.pretrain.train` | The `WandbBackend` in `06_logging_metrics.md` §5 |
| Single-file simplicity | the entire `lit_gpt/` package | The proposed `qwen_palettize/` package follows the same "every file < 500 LOC" principle |

### 6.3 What qwen-palettize should NOT adopt

- **The PyTorch Lightning dependency.** lit-GPT depends on Lightning, which adds ~50 MB of dependencies. qwen-palettize can achieve the same separation of concerns with plain PyTorch.
- **The lack of a `Config` dataclass.** lit-GPT passes CLI args directly, which is fragile. qwen-palettize should use a `Config` dataclass (proposed in `05` §3).

### 6.4 Verdict

lit-GPT is the **closest analog** to qwen-palettize in scope. The proposed refactor in `05` is essentially "make qwen-palettize look like lit-GPT, but with a `Config` dataclass and a multi-backend `MetricsLogger`."

---

## 7. nanoGPT (Karpathy)

**Repository**: <https://github.com/karpathy/nanoGPT>
**Docs**: <https://github.com/karpathy/nanoGPT/blob/master/README.md>
**LOC**: ~300 (the `train.py` file)
**Users**: ~30K stars on GitHub, educational use

### 7.1 Architectural pattern

nanoGPT is the **smallest possible** GPT training framework, written for educational purposes. The entire training loop is in `train.py`, a single ~300-line file. The architecture:

- One file, one function, one loop.
- No abstractions: no `Trainer`, no `Config`, no `MetricsLogger`.
- Direct `torch.optim.AdamW`, direct `torch.amp.autocast`, direct `wandb.init`.
- Pre-tokenized data via `nanoGPT/data/openwebtext/prepare.py3`, which writes `train.bin` and `val.bin`.
- DDP support via `torch.distributed.init_process_group` directly (no Lightning, no DeepSpeed).
- Gradient clipping via `torch.nn.utils.clip_grad_norm_` directly.

### 7.2 What qwen-palettize could adopt

| Pattern | Source in nanoGPT | Analogous qwen-palettize fix |
|---|---|---|
| Single-file simplicity (the *goal*) | `train.py` | The proposed `qwen_palettize/train.py` is the spiritual successor — but with the `Trainer` class to support subclassing |
| Pre-tokenized binary data | `prepare.py3` | The proposed `cache_tokens_v2.py` in `04_data_pipeline.md` §6.4 |
| Direct DDP without a framework | `train.py` lines 50-60 | Not applicable to qwen-palettize (single-GPU), but if multi-GPU is needed, this is the reference |
| No abstractions, just direct PyTorch | the entire file | The proposed refactor moves in the opposite direction (more abstractions), but the principle of "use the framework's primitives, don't reinvent" is sound |

### 7.3 What qwen-palettize should NOT adopt

- **The single-file structure.** qwen-palettize is too complex for one file — the 1,266-line `train_qwen.py` is the warning, not the model. The proposed `qwen_palettize/` package with 10 modules is the right size.
- **The lack of tests.** nanoGPT has no tests. qwen-palettize should not repeat this mistake (see `07_testing_ci.md`).

### 7.4 Verdict

nanoGPT is the **minimum viable** LLM training framework. qwen-palettize has already outgrown this minimum (the 1,266-line monolith is the evidence). The proposed refactor moves qwen-palettize from "nanoGPT scale but messy" to "lit-GPT scale and clean."

---

## 8. Cross-framework comparison matrix

The table below summarizes the architectural patterns of all six frameworks, with qwen-palettize as the reference:

| Feature | HF Trainer | Lightning | Megatron-LM | DeepSpeed | lit-GPT | nanoGPT | qwen-palettize (current) | qwen-palettize (proposed) |
|---|---|---|---|---|---|---|---|---|
| Config dataclass | ✓ (200 fields) | ✓ (~50 fields) | ✗ (CLI args) | ✓ (JSON) | ✗ (CLI args) | ✗ (CLI args) | 3 sources (CLI + JSON + dict) | ✓ (`Config`, ~25 fields) |
| Trainer orchestrator class | ✓ | ✓ | ✗ (function) | ✓ (Engine) | ✗ (function) | ✗ (function) | ✗ (1,266-line function) | ✓ (`Trainer`) |
| Modular separation | ✓ | ✓ | partial | ✓ | ✓ | ✗ | ✗ | ✓ |
| Distributed training | ✓ (DDP/FSDP/DeepSpeed) | ✓ (DDP/FSDP/DeepSpeed) | ✓ (3D parallel) | ✓ (ZeRO/3D) | ✓ (DDP) | ✓ (DDP) | ✗ | not yet (single-GPU) |
| Mixed precision | ✓ (`--bf16`) | ✓ (`--precision`) | ✓ | ✓ (fp16/bf16/fp8) | ✓ (`--precision`) | ✓ (`--dtype`) | chaotic (bf16+fp16+fp32) | to be unified |
| Gradient checkpointing | ✓ (`--gradient_checkpointing`) | ✓ (callback) | ✓ (`checkpoint_layers`) | ✓ | ✓ (direct) | ✗ | ✗ (blocked by PartialWrapper) | ✓ (`use_gradient_checkpointing`) |
| Pre-tokenized data | ✓ (HF datasets cache) | ✓ (use HF datasets) | ✓ (`MMapIndexedDataset`) | ✓ (HF datasets) | ✓ (`train.bin`) | ✓ (`train.bin`) | ✗ (streaming + tokenize per step) | ✓ (`data_cache_dir`) |
| Multi-backend logging | ✓ (wandb/tb/csv/mlflow) | ✓ (wandb/tb/csv) | ✓ (custom) | ✓ (wandb/tb) | ✓ (wandb) | ✓ (wandb) | ✗ (print only) | ✓ (`MetricsLogger`) |
| Test scaffold | ✓ (`transformers/tests/`) | ✓ (`tests/`) | ✓ (`tests/`) | ✓ (`tests/`) | ✓ (`tests/`) | ✗ | ✗ | ✓ (4-tier) |
| CI/CD | ✓ (GitHub Actions) | ✓ (GitHub Actions) | ✓ (internal CI) | ✓ (GitHub Actions) | ✓ (GitHub Actions) | ✗ | ✗ | ✓ (GitHub Actions) |
| `torch.compile` | ✓ (via `--torch_compile`) | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ (blocked by PartialWrapper) | ✓ (via `use_torch_compile`) |
| Single-file structure | ✗ | ✗ | ✗ | ✗ | partial | ✓ | ✓ (messy) | ✗ (10 modules) |
| Single-GPU scope | ✗ | partial | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ |

### Key takeaways

1. **The proposed qwen-palettize refactor matches lit-GPT in scope, with HF Trainer's modularity.** This is the right size for a 4B-parameter single-GPU research project.
2. **The current qwen-palettize codebase is closer to nanoGPT in structure** (single function, no abstractions) but at 3× the size — it has outgrown the nanoGPT pattern without adopting the lit-GPT pattern.
3. **No framework matches qwen-palettize's specific need** (LUT palettization with Gumbel-Softmax trainable indices). The closest analog is lit-GPT, but lit-GPT does not do quantization-aware training. The proposed refactor is therefore novel in combining lit-GPT's simplicity with HF Trainer's modularity, plus a custom `Trainer` subclass for the palettization-specific concerns (tau annealing, palette freezing, two-tier gradient clipping).

---

## 9. The "minimal viable refactor" benchmark

The literature comparison suggests a "minimal viable refactor" (MVR) — the smallest set of changes that brings qwen-palettize in line with industry-standard patterns:

1. **Adopt lit-GPT's pre-tokenized binary data format.** (1 day, `04_data_pipeline.md` Stage A)
2. **Adopt HF Trainer's `Config` dataclass pattern.** (1 day, `05_training_loop_refactor.md` §3)
3. **Adopt Lightning's `Trainer` orchestrator class pattern.** (3 days, `05_training_loop_refactor.md` §4)
4. **Adopt HF Trainer's multi-backend logging pattern.** (1 day, `06_logging_metrics.md`)
5. **Adopt HF Trainer's `--gradient_checkpointing` flag pattern.** (0.5 day, after the `PartialWrapper` fix)
6. **Adopt HF Trainer's test scaffold pattern.** (2 days, `07_testing_ci.md`)
7. **Adopt HF Trainer's CI/CD pattern.** (1 day, `07_testing_ci.md` §7)
8. **Adopt the `nn.Module` contract** (fix `PartialWrapper`). (1 day, `02_partial_wrapper_problem.md`)

Total: ~10.5 days. This is the minimum required to bring qwen-palettize to "industry-standard" status. The full refactoring roadmap in `09_refactoring_roadmap.md` allocates 14.5 days, leaving 4 days of slack for the unanticipated complications that always arise in a refactor of this size.

---

## 10. Conclusion

The qwen-palettize codebase is not using any pattern from any of the six frameworks. It is, in the language of software architecture, a "big ball of mud" — a single 1,266-line function with no separation of concerns, no testability, no observability, and no composability. The fix is not to adopt any single framework wholesale, but to extract the patterns that each framework has validated through years of production use:

- From HF Trainer: the `Config` dataclass, the `Trainer` orchestrator, the multi-backend logging.
- From Lightning: the separation of `__init__/forward/training_step/configure_optimizers`.
- From Megatron-LM: the pre-tokenized binary data format, the forward-overlap-backward pattern.
- From DeepSpeed: the canonical mixed-precision pattern (bf16 params, fp32 optimizer, single master dtype).
- From lit-GPT: the single-file simplicity, the standard `DataLoader` usage, the direct `torch.utils.checkpoint`.
- From nanoGPT: the "use the framework's primitives, don't reinvent" principle.

The proposed refactor in `05_training_loop_refactor.md` is the synthesis of these patterns, adapted for the specific needs of LUT palettization training. The roadmap in `09_refactoring_roadmap.md` is the phased plan to implement it.
