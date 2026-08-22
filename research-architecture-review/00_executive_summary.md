# 00 — Executive Summary: Top 10 Systemic Architecture Deficiencies

**Repository under review**: `pkhairkh/qwen-palettize` (branch `main`, HEAD `b82a6be`).
**Reviewer mandate**: Holistic architecture & workflow review of the 2-bit LUT quantization training system for Qwen3.5-4B. The review deliberately crosses component boundaries — issues that no single kernel/indices/palettes reviewer would surface.
**Scope**: The 14 Python files in `scripts/` (5,904 LOC total, dominated by `train_qwen.py` at 1,265 LOC and `qwen_model.py` at 834 LOC), the project `SPEC.md`, the `.gitignore`, the two `logs/*.log` files, and the cached binary artifacts in `cached_tokens.pt` / `eval_tokens.pt` / `trained/` / `palettized/`.

The training pipeline achieves cos=0.953 against the fp16 teacher on super-block 0, but the SPEC's stated target is cos>0.999, and the run has plateaued across four successive restarts (see git log: `5446edf → 0a46612 → 3a5b750 → b82a6be`). The plateau is not a kernel-accuracy or indices-quality problem alone. It is the surface symptom of a constellation of systemic architecture problems, listed below in severity-ranked order. Each problem is cross-component — fixing it requires touching three or more of the existing code files simultaneously.

---

## Severity ranking

The ranking below uses three axes:

| Code | Meaning |
|---|---|
| **C** | **Cost**: GPU-hours wasted per super-block, or VRAM wasted per step. |
| **R** | **Risk of correctness regression**: probability that the next refactor introduces silent NaN/grad-error. |
| **B** | **Blocking**: blocks the SPEC's cos>0.999 goal directly. |

`Severity = C + R + B` with each axis scored 1–3 (3 = worst). Maximum = 9.

---

## #1 — `PartialWrapper` / `PartialModel` are not `nn.Module` (Severity: 9)

The single most damaging design choice in the codebase. `qwen_model.py` lines 476–571 define `PartialModel` and `PartialWrapper` as plain Python classes with hand-rolled `parameters()`, `named_parameters()`, `named_modules()`, `to()`, `eval()`, `train()`, and `get_submodule()` reimplementations. They do **not** inherit from `torch.nn.Module`. As a direct consequence:

- **`torch.compile(model)`** raises `TypeError` on first tracing pass — the Dynamo frontend walks `_modules` and `_parameters` dicts, which don't exist on a plain class. The 1.5–2× free speedup documented in the PyTorch 2.x release notes is unreachable.
- **Gradient checkpointing** (`torch.utils.checkpoint.checkpoint`) cannot wrap any layer inside `PartialWrapper`. The checkpoint API requires the wrapped callable to participate in autograd via `nn.Module.forward`; the wrapper bypasses this contract.
- **FSDP / DDP / model parallel** wrappers all expect `nn.Module`. `FullyShardedDataParallel(wrapper)` raises immediately.
- **`activation_checkpointing`** in the HuggingFace Transformers sense (calling `layer.gradient_checkpointing_enable()`) silently no-ops because the layers list is a plain Python `list` stored as `self.layers`, not as `nn.ModuleList`.
- **State dict save/load** is broken: `torch.save(wrapper.state_dict())` raises because `state_dict()` is not implemented. The codebase works around this with the manual per-tensor `save_state()` in `train_qwen.py:742`, which itself produces 140+ `.pt` files for one super-block (visible in `trained/superblock_0_best/`).
- **HuggingFace `from_pretrained`** is invoked twice per super-block (once for student, once for teacher), because `load_qwen_super_block_only` does not return an `nn.Module` the HF cache can manage. See issue #3.

The full breakdown of breakage is in `02_partial_wrapper_problem.md`.

---

## #2 — Monolithic 1,265-line `train_super_block` function (Severity: 8)

`train_qwen.py:922–1224` is a single Python function spanning 302 lines of body and incorporating, in execution order: log rewiring, hyperparameter loading, model building, parameter counting, optimizer construction, LR scheduler construction, teacher loading, teacher freezing, embed_tokens sharing, training loop, NaN detection, gradient clipping, optimizer stepping, log clamping, scheduler stepping, gradient zeroing, eval trigger, eval execution, save trigger, save execution, and shutdown hook. The `main()` function at line 1227 is a 32-line `argparse` shim that does nothing else.

This monolith blocks every form of iteration:

- You cannot unit-test the loss without instantiating the full teacher.
- You cannot A/B test two optimizers without copy-pasting the entire function.
- You cannot insert a profiler step without editing the live function.
- Hyperparameter sweeps (`sweep_qwen.py`) are forced to call this function as a black box and parse `print()` output to recover metrics.
- The `live JSON control` mechanism (`/tmp/hyperparams_qwen.json`, reloaded every 10 steps at line 1043) is a workaround for the absence of a proper config object — it is the third source of truth after CLI args and `DEFAULT_HYPERPARAMS`.

The refactor target is detailed in `05_training_loop_refactor.md`.

---

## #3 — Teacher and student are loaded separately (Severity: 7)

`train_qwen.py:942` calls `build_student_super_block` → `load_qwen_super_block_only` → `from_pretrained(Qwen/Qwen3.5-4B)`. Then `train_qwen.py:987` calls `load_qwen_super_block_only` **again** for the teacher. The HF cache makes the second download cheap, but the second model object holds a full second copy of all 1,081,957,952 prefix params (per train log line 23: "Loaded prefix: 1,081,957,952 params").

A naive tally: 2 × 1.08B params × 2 bytes (bf16) = **4.32 GB of VRAM consumed by the prefix alone**, of which at least 2.16 GB is redundant — both teacher and student hold the same `embed_tokens` (635.7M params × 2 bytes = 1.27 GB) and the same frozen `layers[0..sb_start-1]` when `sb_idx > 0`. The sharing line `student.model.embed_tokens = teacher.model.embed_tokens` (line 994) only re-points the student's attribute; the original student `embed_tokens` tensor is not freed, it is just orphaned (Python refcount still holds it via `model.model.embed_tokens`'s previous binding). `gc.collect()` and `torch.cuda.empty_cache()` are never called between the two loads.

The proposed architecture: load the teacher **once**, share `embed_tokens` + frozen prefix layers, build the student **in place** by replacing the active super-block's Linears with `PalettizedLinear`. This is the "in-place palettization" pattern used by GPTQ, AWQ, and SpinQuant. Detailed memory math in `03_memory_waste_analysis.md`.

---

## #4 — No gradient checkpointing anywhere (Severity: 7)

A `grep` for `checkpoint`, `checkpoint_sequential`, `use_checkpoint`, `gradient_checkpointing` across all 14 scripts returns **zero** matches in `scripts/train_qwen.py` and `scripts/qwen_model.py`. The only matches are in `profile_training.py` and `profile_nosync.py` — and those are uses of `torch.utils.checkpoint` as a *baseline comparison*, not as the production path.

For super-block 0 at `seq_len=512`, `batch_size=32` (the configuration in `train_sb0.log` line 6), the activation memory for 4 Qwen3.5 layers can be estimated as roughly:

- Per-layer attention activations: `batch × heads × seq × seq × 2 bytes ≈ 32 × 40 × 512 × 512 × 2 ≈ 670 MB` per layer.
- Per-layer MLP activations: `batch × seq × intermediate × 2 bytes ≈ 32 × 512 × 9216 × 2 ≈ 590 MB` per layer.
- Four layers × ~1.26 GB = **~5 GB** of activation memory per step.

With checkpointing at the layer granularity, this collapses to **~1.3 GB** (one layer live at a time), enabling `batch_size=128` at the same `seq_len` — a 4× throughput improvement for free. The reason this isn't enabled is directly issue #1: the layers are stored in a plain Python list, not an `nn.ModuleList`, so HuggingFace's `gradient_checkpointing_enable()` cannot find them.

---

## #5 — Mixed-precision chaos (Severity: 6)

The codebase uses four different dtypes simultaneously with no coherent master/slave relationship:

| Component | Dtype | Where defined |
|---|---|---|
| Frozen `embed_tokens` | bf16 | `qwen_model.py:994` via `from_pretrained(torch_dtype=bfloat16)` |
| Student palette params | bf16 | `qwen_model.py:82` |
| `index_logits` params | **fp16** | `qwen_model.py:124` |
| LoRA `lora_A`, `lora_B` | bf16 | `qwen_model.py:236–237` |
| Optimizer master weights | fp32 | `train_qwen.py:170` (`FP32MasterOptimizer`) |
| `grad_palette` accumulator | fp32 (cast to bf16) | `fused_lut_linear_cuda.py:158–168` |
| Loss math | fp32 (via `.float()` casts inside `compute_loss`) | `train_qwen.py:224–225` |

This violates PyTorch's autocast convention (which expects a single `dtype` per region) and forces the manual `FP32MasterOptimizer` wrapper (154 LOC, lines 153–214) which itself violates `torch.optim.Optimizer`'s contract — it is **not** an `Optimizer` subclass, so any framework that calls `isinstance(opt, torch.optim.Optimizer)` (e.g., accelerate, transformers' `Trainer`, lightning's `LiteOptimizer`) will reject it. The clamp at `train_qwen.py:1149–1153` (`par.data.clamp_(-20.0, 20.0)` on every step) is a band-aid for `fp16` overflow that would not exist if `index_logits` were `bf16` (which has the same exponent range as fp32).

---

## #6 — No data pipeline; FineWeb-Edu streamed synchronously (Severity: 6)

`stream_training_data` at `train_qwen.py:889–918` calls `datasets.load_dataset("HuggingFaceFW/fineweb-edu", streaming=True)` and yields batches one at a time. Each `next(iter)` triggers a network round-trip (HTTP request to HF Hub → CDN download → parquet decode → JSON parse → tokenization on the calling thread). The tokenization in particular is single-threaded Python and runs **on the same thread as the GPU step**, blocking the GPU between steps.

A 1.73 tps baseline (from `train_sb0.log` and the `5446edf` commit message: "1.45 → 1.73 tps, 30% faster backward") at `seq_len=512, batch=32` means each step takes ~580 ms. Of that, a few hundred ms is the network+tokenization window when the GPU should be doing forward+backward. With proper prefetch (async `IterableDataset` + pinned-memory ring buffer), this drops to near zero. The fix is detailed in `04_data_pipeline.md`.

The codebase also has `cached_tokens.pt` (516 KB) and `eval_tokens.pt` (1.1 MB) — tokenized eval data is cached, but training data is not. There is no on-disk tokenization cache, no `map(num_proc=8)` pre-pass, no `pin_memory=True`, no `prefetch_factor` argument.

---

## #7 — No structured logging or metrics (Severity: 5)

Every metric is emitted via `print(..., flush=True)` to stdout, which is tee'd into `logs/train_sbN.log` by the `_DualStream` shim at `train_qwen.py:50–73`. The output format is:

```
  step={global_step:5d} loss={comps['loss']:.4f} cos={cos_val:.4f} tps={tps:.1f} {tau_str}{gpu_str} gn=[{gn_str}] [{lrs_str}]
```

There is no machine-readable emitter (no `wandb.log`, no `SummaryWriter.add_scalar`, no CSV sink). Hyperparameter sweep results can only be compared by grepping log files. The `sweep_qwen.py` script (213 LOC) has to parse these free-text strings to recover numbers — fragile, error-prone, and brittle to any cosmetic format change. The `eval_every=250` cadence is the only structured metric, and even that is interleaved with training-step prints in the same file. Without time-series DB support, you cannot easily plot `cos vs tau`, `grad_norm[indices] vs step`, or `gpu_util vs batch_size` — three correlations the SPEC explicitly asks for (§5, Temperature Annealing).

---

## #8 — No automated testing (Severity: 5)

The repo has zero `pytest` tests, zero `unittest.TestCase` subclasses, and no `tests/` directory. The only thing named `test_*` is `test_palettized_v2.py` (213 LOC), which is a **manual smoke script** — it prints assertions to stdout and exits, with no test runner, no fixture, no parametrization, no coverage. The `verify_palettize_core.py` (167 LOC) and `verify_2bit_format.py` (198 LOC) are similarly manual.

The user-visible cost: every refactor requires running 8,000 training steps (~1.5 hours) to detect regression. The user message confirms this: "Every change is tested by running 8000 steps." There is no CI, no `.github/workflows/`, no pre-commit hook, no minimum-viable smoke test that runs in under 60 seconds. The CUDA kernel is verified by 41 micro-tests in a flat script — but these run only when manually invoked, and the kernel ↔ autograd ↔ training-loop integration has zero coverage.

The fix is detailed in `07_testing_ci.md`.

---

## #9 — Three sources of truth for configuration (Severity: 4)

Configuration is split across three layers with no documented precedence:

1. `DEFAULT_HYPERPARAMS` dict at `train_qwen.py:75–102` — Python literal, requires code edit to change.
2. `argparse` CLI args at `train_qwen.py:1227–1249` — only top-level knobs (sb_idx, max_steps, lora_rank, seq_len, batch_size, tau_*, shutdown_on_done).
3. `/tmp/hyperparams_qwen.json` — re-read every 10 steps at `train_qwen.py:1043`, can change LRs / loss_type / loss_weights / freeze groups live.

The interaction is non-obvious: the JSON file overrides the `DEFAULT_HYPERPARAMS` dict, but only for the keys it contains. If you delete the JSON, the `DEFAULT_HYPERPARAMS` wins. If you set `groups.palettes = false` in the JSON, it takes effect on the next 10-step boundary. But if you also pass `--lora_rank 32` on the CLI, that overrides the **default** but not the **live JSON**. There is no schema, no validation, no typed `Config` object. A typo like `"palattes": 3e-4` in the JSON silently creates a new key that the loader never reads, and the original `palettes` LR stays at its default.

The fix is Hydra or OmegaConf with a single YAML schema, merged from CLI overrides. See `05_training_loop_refactor.md`.

---

## #10 — Export pipeline detached from training (Severity: 3)

`convert_trained_to_packed.py` is a 120-LOC one-off script that walks `trained/superblock_{N}_best/*.pt` files and writes `.idx2` + `.lut_scalar` + `metadata.json` to a sibling `_best_packed/` directory. It is a **migration script** for an old format, not part of the production export flow. The SPEC §3.1 Stage 3 calls for `merge_qwen.py` that "extract hard indices + merge LoRA into palette" — this file does not exist in the repo.

The result: a saved checkpoint in `trained/superblock_0_best/` contains 140+ `.pt` files (~232 MB on disk) holding intermediate optimizer-visible tensors, not deployable artifacts. To go from "training finished" to "2-bit deployable model" requires running three more scripts in sequence (none of which are wired together): `convert_trained_to_packed.py` → a future `merge_qwen.py` → a future `assemble_qwen.py`. None of these are integrated. A single training crash at step 8000 leaves a half-finished checkpoint that no downstream tool can consume without manual intervention.

---

## Cross-cutting root cause

These ten issues are not independent. They share a single root cause: **the codebase was built as a research script, not as a system**. `train_qwen.py` was written top-down by one author who needed to see `cos=0.95` by a deadline. Every infrastructure concern (modularity, testability, reproducibility, logging) was deferred in favor of forward progress on the cos metric. The four parallel research agents investigating kernel/indices/palettes correctness are optimizing local concerns; none of them will surface these architectural problems, because the architecture is not their assignment.

The remainder of this review (documents 01 through 10) provides the diagnostic depth, the refactoring proposals, the literature comparison, and the phased roadmap to address these systemic issues without halting the research program.

---

## Recommended triage order (impact × cost)

| Order | Issue | Est. effort | Expected ROI |
|---|---|---|---|
| 1 | `PartialWrapper` → `nn.Module` (#1) | 1 day | Unlocks #4, #6, #11 |
| 2 | In-place teacher/student loading (#3) | 1 day | Saves 2.16 GB VRAM; enables batch=64 |
| 3 | Gradient checkpointing (#4) | 0.5 day | 4× batch size, 3× throughput |
| 4 | `train_super_block` refactor (#2) | 3 days | Unlocks #5, #7, #9 |
| 5 | Mixed-precision unification (#5) | 1 day | Removes `FP32MasterOptimizer` kludge |
| 6 | Async data prefetch (#6) | 2 days | Recovers ~30% step time |
| 7 | Structured logging (#7) | 1 day | Enables sweeps, plots, dashboards |
| 8 | Test scaffold (#8) | 2 days | 60s smoke test replaces 1.5h regression |
| 9 | Hydra config (#9) | 1 day | One source of truth |
| 10 | Export integration (#10) | 2 days | One-step training → deployable |

Total: ~14.5 engineering days for the full Wave 1–3 program. See `09_refactoring_roadmap.md` for the phased plan.
