# 01 — Architecture Audit: Current System Diagram + Component Analysis

## 1. Purpose and scope

This document is the canonical reference for **what the qwen-palettize codebase is today**, as of commit `b82a6be` on `main`. It is the substrate against which all subsequent review documents (`02_partial_wrapper_problem.md` through `09_refactoring_roadmap.md`) are written. The audit covers every file under `scripts/`, the `SPEC.md`, the `logs/` directory, and the binary artifacts in `trained/`, `palettized/`, `cached_tokens.pt`, and `eval_tokens.pt`.

The audit is deliberately exhaustive on the **structure** of the system and conservative on the **correctness** of individual kernels — that is the kernel-accuracy agent's job. Our concern here is the **shape** of the system: how the components connect, where the contracts are violated, where the data flows synchronously that should flow asynchronously, and where the abstractions leak.

---

## 2. System architecture diagram (ASCII)

The diagram below is the ground truth of the system as it actually runs. It is not the diagram from `SPEC.md` §2.1 (which is the *intended* architecture) — it is the *implemented* architecture, reconstructed from `train_qwen.py:train_super_block` and `qwen_model.py:load_qwen_super_block_only`.

```
                ┌──────────────────────────────────────────────────────────────┐
                │  scripts/train_qwen.py  (1,265 LOC)                          │
                │                                                              │
                │  ┌────────────────────────────────────────────────────────┐  │
                │  │ main() L1227  →  argparse (CLI source of truth #1)     │  │
                │  └──────────────────────┬─────────────────────────────────┘  │
                │                         │                                     │
                │                         ▼                                     │
                │  ┌────────────────────────────────────────────────────────┐  │
                │  │ train_super_block() L922  (302-line monolith)           │  │
                │  │                                                        │  │
                │  │  1. write_default_hyperparams() L938  ─────┐            │  │
                │  │  2. load_hyperparams() L939  ─────────────┤ JSON       │  │
                │  │                                           │ source #3 │  │
                │  │  3. build_student_super_block() L942 ──────┼──→ qwen_   │  │
                │  │  4. apply_groups() L947                   │    model.py│  │
                │  │  5. load_state() L959 (if --resume)       │            │  │
                │  │  6. build_optimizers() L963  ─────────────┤            │  │
                │  │     ├─ FP32MasterMuon (154 LOC)            │            │  │
                │  │     └─ FP32MasterAdamW                     │            │  │
                │  │  7. LambdaLR schedulers L973-975           │            │  │
                │  │  8. load_qwen_super_block_only() L987 ─────┘            │  │
                │  │     (TEACHER — second HF from_pretrained call!)         │  │
                │  │  9. teacher.model.embed_tokens ← student.model.embed_tokens
                │  │                                                        │  │
                │  │  ┌──────────────────────────────────────────────────┐  │  │
                │  │  │ for batch_ids in stream_training_data():         │  │  │
                │  │  │   • tau anneal      L1035                        │  │  │
                │  │  │   • JSON reload      L1043  (every 10 steps)     │  │  │
                │  │  │   • teacher fwd on   L1060  (stream_t)           │  │  │
                │  │  │     stream_t                                       │  │  │
                │  │  │   • student fwd     L1082  (default stream)      │  │  │
                │  │  │   • compute_loss    L1103                        │  │  │
                │  │  │   • NaN check       L1105                        │  │  │
                │  │  │   • loss.backward() L1123                        │  │  │
                │  │  │   • clip_grad_norm_ L1137  (indices separate)    │  │  │
                │  │  │   • opt.step()      L1142-4                      │  │  │
                │  │  │   • clamp logits    L1149  ±20                   │  │  │
                │  │  │   • sched.step()    L1154-6                      │  │  │
                │  │  │   • zero_grad        L1157-9                      │  │  │
                │  │  │   • print()          L1163  (every 50 steps)    │  │  │
                │  │  │   • evaluate()       L1194  (every 250 steps)    │  │  │
                │  │  │   • save_state()     L1205  (every 2000 + best)  │  │  │
                │  │  └──────────────────────────────────────────────────┘  │  │
                │  └────────────────────────────────────────────────────────┘  │
                └──────────────────────────────────────────────────────────────┘

                       │                              │
                       ▼                              ▼
   ┌───────────────────────────────────┐    ┌──────────────────────────────────┐
   │ scripts/qwen_model.py  (834 LOC)  │    │ scripts/palettize_core.py (178)  │
   │                                   │    │                                  │
   │  • PalettizedLinear (nn.Module)   │    │  • pack_idx2()                   │
   │    ├ indices (int8 buffer)         │    │  • pack_indices_transposed_2bit()│
   │    ├ indices_int8 (int8 buffer)   │    │  • palettize_tensor_2bit()       │
   │    ├ palette (bf16 Parameter)     │    │  • load_indices() / load_lut()   │
   │    ├ index_logits (fp16 Param)   │    │  • sanitize_name()               │
   │    ├ bias (bf16 buffer)           │    │  • write_lut_scalar()            │
   │    ├ _flat_idx (int64 buffer)     │    │  • write_metadata_json()         │
   │    ├ _hard_kernel (function ref) │    └──────────────────────────────────┘
   │    └ _soft_kernel (function ref) │
   │    forward(x):                    │
   │      if train and soft:           │
   │        y = _soft_kernel(...)      │   ┌──────────────────────────────────┐
   │      else:                        │   │ scripts/fused_lut_linear_cuda.py │
   │        y = _hard_kernel(...)      │   │              (709 LOC)            │
   │      + LoRA correction            │   │                                  │
   │                                   │   │  • CUDAFusedLUTLinear            │
   │  • QwenLoRA (nn.Module)           │   │    (torch.autograd.Function)     │
   │    ├ lora_A (bf16 Parameter)      │   │  • CUDAFusedLUTLinearSoft        │
   │    └ lora_B (bf16 Parameter)      │   │    (torch.autograd.Function)     │
   │                                   │   │  • load_inline(.cu source)        │
   │  • PartialModel (plain class!)    │   │                                  │
   │  • PartialWrapper (plain class!)  │   │  Wraps: fused_lut_kernel.cu      │
   │  • load_qwen_super_block_only()  │   │  (Tensor Core fwd + 3 bwd        │
   │  • load_qwen_model()              │   │   kernels)                       │
   │  • palettize_linear()             │   └──────────────────────────────────┘
   │  • capture_original_weights()     │
   │  • attach_lora_to_layer()         │   ┌──────────────────────────────────┐
   │  • should_palettize()             │   │ scripts/calib_qwen.py (365 LOC)   │
   │  • isolate_super_block()          │   │  • Stage 1 calibration: GPTQ +   │
   │  • insert_correction_layers()     │   │    kmeans1d → .idx2 + .lut_scalar│
   │                                   │   │  • Output → palettized/superblk/ │
   └───────────────────────────────────┘   └──────────────────────────────────┘

                       │                              │
                       ▼                              ▼
   ┌───────────────────────────────────┐    ┌──────────────────────────────────┐
   │ DATA PATH                         │    │ ARTIFACTS                         │
   │                                   │    │                                  │
   │  FineWeb-Edu (HF streaming)       │    │  palettized/superblock_0/        │
   │    │                              │    │    47 files: .idx2 + .lut_scalar │
   │    ▼                              │    │    metadata.json                 │
   │  stream_training_data() L889     │    │                                  │
   │  • single-threaded tokenization  │    │  trained/superblock_0_best/      │
   │  • synchronous network I/O        │    │    140+ .pt files (intermediate)  │
   │  • no prefetch, no pin_memory     │    │    _resume.json                  │
   │  • no tokenization cache          │    │                                  │
   │                                   │    │  trained/superblock_0_safe/     │
   │  cached_tokens.pt (516 KB)        │    │    140+ .pt files (backup)       │
   │  eval_tokens.pt    (1.1 MB)       │    │                                  │
   └───────────────────────────────────┘    └──────────────────────────────────┘

                       │
                       ▼
   ┌───────────────────────────────────┐
   │ LOGGING / OBSERVABILITY            │
   │                                   │
   │  stdout  → _DualStream L50         │
   │    │                              │
   │    ▼                              │
   │  logs/train_sbN.log (overwrite)   │
   │                                   │
   │  No wandb / tensorboard / CSV     │
   │  No structured metrics emitter    │
   └───────────────────────────────────┘
```

Three observations from the diagram alone:

1. **The data path is the smallest box.** FineWeb-Edu enters through a 30-line streaming generator and exits through a `print()` statement. There is no caching, no prefetch, no async — the data path is treated as a side effect, not a first-class pipeline component.
2. **The artifact directory is the largest box.** 140+ `.pt` files per checkpoint is a direct consequence of issue #1 (`PartialWrapper` is not `nn.Module`, so `state_dict()` is unavailable, so the save logic enumerates parameters by hand and emits one file per tensor).
3. **The teacher's `from_pretrained` call appears inside the training function.** This is the duplication issue (#3) — the teacher is loaded as a peer of the student, not as a parent.

---

## 3. Component inventory

The 14 Python files in `scripts/` total 5,904 lines of code. They break down as follows:

| File | LOC | Role | Layer |
|---|---|---|---|
| `train_qwen.py` | 1,266 | Training loop, optimizers, loss, eval, save/load | Application |
| `qwen_model.py` | 834 | `PalettizedLinear`, `QwenLoRA`, `PartialWrapper`, model loading | Model |
| `fused_lut_linear_cuda.py` | 708 | `torch.autograd.Function` wrappers for the CUDA kernels | Kernel |
| `fused_lut_kernel.cu` | — | Tensor Core forward + three backward kernels | Kernel (CUDA) |
| `palettize_core.py` | 178 | 2-bit packing, kmeans hooks, metadata writer | Compression |
| `palettize_pytorch.py` | 207 | Reference kmeans1d, `palettize_groups`, `reconstruct_Wq` | Compression |
| `calib_qwen.py` | 365 | Stage 1 calibration: activations → GPTQ → kmeans | Pipeline |
| `calib_stage2.py` | 362 | (Legacy) Stage 2 palettization — to be removed | Pipeline (legacy) |
| `convert_trained_to_packed.py` | 120 | One-off migration: `.pt` → `.idx2 + .lut_scalar` | Pipeline |
| `cache_tokens.py` | 48 | Tiny utility to pre-cache eval tokens | Tooling |
| `verify_palettize_core.py` | 167 | Manual smoke test of compression round-trip | Tooling |
| `verify_2bit_format.py` | 198 | Manual smoke test of 2-bit packing | Tooling |
| `test_palettized_v2.py` | 213 | Manual smoke test of `PalettizedLinear` | Tooling |
| `bench_palettized_v2.py` | 113 | Microbenchmark: palettized vs dense matmul | Profiling |
| `profile_training.py` | 255 | Profile the training step (with/without checkpoint) | Profiling |
| `profile_nosync.py` | 196 | Profile without `torch.cuda.synchronize` | Profiling |
| `sweep_qwen.py` | 313 | Hyperparameter sweep (parses `print()` output) | Tooling |
| `dry_run_counts.py` | 60 | Print param counts without training | Tooling |

### Layered analysis

**Application layer (1,266 LOC in a single file).** `train_qwen.py` accounts for 21% of the codebase. It is the entry point and contains *all* of: optimizer definitions, FP32 master weight wrappers, the loss function, the Gumbel-Softmax temperature scheduler, the iterative palette freezing logic, the gradient noise injection, the cyclic loss schedule, the eval set preparation, the streaming data generator, the model builder, the parameter classifier, the optimizer builder, the LR updater, the state save/load, and the training loop. This concentration is the central fact of the architecture: **the application layer has no sub-modules**. The file is its own module.

**Model layer (834 LOC).** `qwen_model.py` is half the size of `train_qwen.py` but contains more architectural decisions per line. The two flagship classes are `PalettizedLinear` (lines 65–180, an `nn.Module` that holds int8 indices, a bf16 palette, an fp16 `index_logits`, and a dual-mode forward that dispatches to either the hard or soft CUDA kernel) and `QwenLoRA` (lines 184–265, an `nn.Module` that wraps either a `PalettizedLinear` or an `nn.Linear` and adds a rank-r LoRA correction). Both of these are *correct* `nn.Module` subclasses — they would work with `torch.compile`, FSDP, and checkpointing in isolation.

The architectural defect lives in the **prefix wrapper**, not in the linears. Lines 476–571 define `PartialModel` and `PartialWrapper` as plain Python classes, ostensibly to avoid the overhead of `nn.Module`'s `_modules` dict. The cost of this shortcut is documented in `02_partial_wrapper_problem.md`. The intent (visible in the docstring at line 17) was to share `embed_tokens` between teacher and student to save VRAM, but the actual implementation does not achieve that — `student.model.embed_tokens = teacher.model.embed_tokens` (line 994) re-points the attribute but does not free the student's previous tensor.

**Kernel layer (708 + 0 LOC, plus the CUDA file).** `fused_lut_linear_cuda.py` is a careful, well-structured wrapper around the `.cu` kernel. It exposes two `torch.autograd.Function` subclasses (`CUDAFusedLUTLinear` for hard indices, `CUDAFusedLUTLinearSoft` for Gumbel-Softmax relaxation), each with proper `forward` and `backward` static methods that call into the C++ host wrappers via `load_inline`. The dtype contract is explicit and enforced via `TORCH_CHECK` macros: bf16 in, bf16 out, fp32 accumulator for `grad_palette`. The kernel layer is the most well-engineered part of the codebase. The only architectural complaint against it is that `torch.autograd.Function` does not play well with `torch.compile` mode `"reduce-overhead"` (CUDA Graphs) without manual intervention — but this is a known PyTorch limitation, not a codebase defect.

**Compression layer (178 + 207 LOC).** `palettize_core.py` is a thin shim around `palettize_pytorch.py` that overrides the `pack_indices_transposed` function to emit 2-bit packing instead of 4-bit. The override at lines 50–57 (`palettize_pytorch.pack_indices_transposed = pack_indices_transposed_with_2bit`) is a **module-level monkey-patch** — it mutates a different module's namespace at import time. This is a fragile pattern: if `palettize_pytorch` is ever imported before `palettize_core`, the patch is silently no-op. The pattern should be replaced with a function parameter or a subclass.

**Pipeline layer (365 + 362 + 120 LOC).** The calibration scripts (`calib_qwen.py`, `calib_stage2.py`) and the export script (`convert_trained_to_packed.py`) are independent programs that share no code with `train_qwen.py` other than the model-loading function. They are not chained together — there is no `Makefile`, no `dvc.yaml`, no `snakemake` workflow. Running a super-block from end to end requires the operator to remember the order: `calib_qwen.py` → `train_qwen.py` → `convert_trained_to_packed.py` → (future) `merge_qwen.py` → (future) `assemble_qwen.py`. The `SPEC.md` documents this as a 3-stage pipeline (§3.1) but the code only implements stages 1 and 2 plus a partial migration script.

**Tooling layer (~1,000 LOC across 6 files).** `verify_*` and `test_*` scripts are manual smoke tests. `profile_*` scripts are one-off benchmarks. `sweep_qwen.py` is the most ambitious tool — it runs `train_super_block` in subprocesses with different hyperparameters and parses the log files to compare results. It works, but the parsing is brittle: a single `.4f` → `.5f` format change in the print statement at `train_qwen.py:1188` would silently break the sweep.

---

## 4. Data flow analysis

This section traces the journey of a single training batch from the FineWeb-Edu dataset to the gradient update on `index_logits`. The trace exposes four serialization points where the GPU idles waiting for Python.

### Stage A: Network fetch (synchronous, ~50–200 ms)

`stream_training_data` at `train_qwen.py:889` calls `datasets.load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True, name="sample-10BT")` once. The returned `IterableDataset` is iterated with a plain `for ex in ds:` loop (line 902). Each `next(iter(ds))` triggers a network request to the HF Hub CDN, downloading a parquet shard, decoding it, and yielding a Python dict with `{"text": ..., "score": ...}`. There is no prefetch, no background thread, no ring buffer. The GPU sits idle for the duration of each fetch.

### Stage B: Tokenization (synchronous, ~30–80 ms per batch)

Line 905: `ids = tokenizer(text, add_special_tokens=True, truncation=True, max_length=seq_len, return_tensors="pt")["input_ids"].squeeze(0)`. The tokenizer is the HuggingFace fast Rust tokenizer, but it is invoked single-threaded from Python. For a batch of 32 sequences at seq_len=512, this is ~30 ms on an L4 CPU — and it happens on the training thread, blocking the GPU.

### Stage C: Teacher forward (CUDA stream `stream_t`, ~80 ms)

Lines 1060–1078 launch the teacher forward on a dedicated CUDA stream. This is the **one** place the codebase uses stream parallelism, and it is used correctly: the teacher forward overlaps with the student forward on the default stream. However, the synchronization is implicit — the teacher's output `h_out` is `.detach()`'d at line 1078 but the stream is not explicitly synchronized before `compute_loss` uses `h_out` at line 1103. PyTorch's default stream synchronization makes this correct, but the cross-stream dependency is invisible to `torch.compile`'s CUDA Graph capture.

### Stage D: Student forward + backward (default stream, ~200 ms)

Lines 1082–1099 run the student forward, layer by layer, in a Python `for` loop. Each layer's forward is a Python call into `nn.Module.forward`, which dispatches to the CUDA kernel via the `PalettizedLinear.forward` method. The Python dispatch overhead is small per-call (~10 µs) but accumulates: 4 layers × ~25 sub-modules per layer × 10 µs = 1 ms of Python overhead per step, on top of the 200 ms of actual CUDA work.

The backward at line 1123 (`loss.backward()`) is the dominant cost. The soft kernel's backward computes `grad_W = x.T @ grad_y` in Python (via cuBLAS), then dispatches two CUDA kernels for `grad_logits` and `grad_palette`. This Python-in-the-loop backward is the reason `torch.compile` cannot be applied to the soft path — the matmul is outside the `autograd.Function`.

### Stage E: Optimizer step (CPU + GPU, ~50 ms)

The `FP32MasterOptimizer.step()` at `train_qwen.py:186–199` does three things: (1) copy bf16 model grads → fp32 master grads (line 191), (2) call the inner optimizer's `.step()` on the fp32 masters (line 194), (3) copy fp32 masters back → bf16 model params (lines 196–199). The three steps are sequential, no streams, no overlap. For 17 M trainable params (per SPEC §2.4), this is ~50 ms of HBM bandwidth.

### Serialization summary

| Stage | GPU activity | Wall time (L4, batch=32, seq=512) | Idle % |
|---|---|---|---|
| A. Network fetch | None | 50–200 ms | 100% idle |
| B. Tokenization | None | 30–80 ms | 100% idle |
| C. Teacher fwd (stream_t) | Yes | ~80 ms | 0% (overlaps D) |
| D. Student fwd+bwd | Yes | ~200 ms | 0% |
| E. Optimizer step | Partial | ~50 ms | ~80% idle |
| **Total per step** | | **~410–530 ms** | **~50% GPU idle** |

A 50% GPU idle rate is consistent with the observed `tps=1.73` (580 ms/step) in `train_sb0.log`. The theoretical minimum, with all stages overlapped and no Python dispatch overhead, is ~250 ms/step, or `tps≈4.0` — a 2.3× speedup that requires no kernel changes, only architecture changes.

---

## 5. Configuration and state management

The system has three configuration sources (see `00_executive_summary.md` issue #9). Their interaction at runtime is:

```
  CLI args (L1227)
       │
       ▼
  train_super_block(sb_idx, max_steps, lora_rank, ...)
       │
       ├── write_default_hyperparams() L938  →  /tmp/hyperparams_qwen.json (initial)
       ├── load_hyperparams()          L939  ←  /tmp/hyperparams_qwen.json (read back)
       │
       │   hp = {
       │     "groups":   {"palettes": True, "lora": True, ...},
       │     "lrs":      {"palettes": 3e-3, "lora": 1e-3, ...},
       │     "loss_type": "norm_mse",
       │     ...
       │   }
       │
       ├── build_optimizers(model, hp, sb_idx) L963  ←  hp["lrs"], hp["groups"]
       │
       └── [training loop]
              │
              ├── every 10 steps: load_hyperparams() L1044  ←  JSON may have changed!
              │   if changed:
              │     apply_groups(model, new_hp, sb_idx)    L1050
              │     update_lrs(opt, new_hp, sb_idx, sched) L1052
              │
              └── every 50 steps: print(... hp["log_every"] ...)
```

The JSON file is the **dominant** source of truth at runtime — CLI args are read once at startup and never consulted again. But the JSON file is also the **least validated** source: there is no schema check, no type check, no key check. A typo in `palattes` (sic) creates a new key that the loader never reads. The `DEFAULT_HYPERPARAMS` dict at line 75 is the **fallback** source, used only when the JSON file is missing or does not contain a key.

The `--resume_from` CLI argument is the only one that is consulted at runtime (other than the initial parse). It is passed to `load_state` at line 959, which loads 140 `.pt` files from the resume directory. The resume mechanism is **per-tensor**: each `.pt` file is loaded individually with `torch.load(...)`, then assigned to the corresponding parameter via `model.get_submodule(name).data.copy_(t)`. This is correct but slow — 140 file opens + 140 GPU copies takes ~30 seconds at the start of every resumed run, visible in `train_sb0.log` line 35: "Resumed 140 params from ... (step=8000, cos=0.946436)".

---

## 6. State save and load

`save_state` at `train_qwen.py:742` writes one `.pt` file per trainable tensor. The directory structure is:

```
trained/superblock_0_best/
├── _resume.json                                          # {step, cos, loss}
├── layers_0_input_layernorm_weight.pt                    # layernorm
├── layers_0_linear_attn_in_proj_a_base_weight.pt         # SSM (frozen?)
├── layers_0_linear_attn_in_proj_qkv_base.idx2            # hard indices (already packed)
├── layers_0_linear_attn_in_proj_qkv_base.lut_scalar      # palette
├── layers_0_linear_attn_in_proj_qkv_lora_A.pt            # LoRA A
├── layers_0_linear_attn_in_proj_qkv_lora_B.pt            # LoRA B
├── layers_0_linear_attn_in_proj_z_base.idx2
├── layers_0_linear_attn_in_proj_z_base.lut_scalar
├── layers_0_linear_attn_in_proj_z_lora_A.pt
├── layers_0_linear_attn_in_proj_z_lora_B.pt
├── ... (140+ files total)
└── layers_3_self_attn_v_proj_base.idx2
```

Two observations:

1. **Mixed file formats in one directory.** Some tensors are saved as `.pt` (raw torch tensors), some as `.idx2` (2-bit packed binary), some as `.lut_scalar` (palette binary). The format depends on the tensor type and is determined by string matching on the tensor name (`"palette"`, `"index_logits"`, `"lora_A"`, `"lora_B"`) inside `save_state`. There is no manifest that says which format each tensor uses — the loader has to guess from the file extension.
2. **No version field.** The `_resume.json` contains `{step, cos, loss, sb_idx}` but no schema version. If the save format ever changes, old checkpoints will load with silent corruption. The `convert_trained_to_packed.py` script exists because the format *did* change (from `.pt`-only to `.idx2`+`.lut_scalar`), and the migration is manual.

The natural fix is to use `nn.Module.state_dict()` (which serializes all parameters into a single OrderedDict of typed tensors) plus a manifest JSON with schema version. This is blocked by issue #1 — `PartialWrapper` is not `nn.Module`, so `state_dict()` is unavailable.

---

## 7. Logging and observability

The logging system is a single class: `_DualStream` at `train_qwen.py:50–73`. It is a `sys.stdout` replacement that tees writes to both the original stdout and a log file. The log file is overwritten on every run (`open(log_path, "w", buffering=1)` at line 54). There is no log rotation, no per-run directory, no timestamp in the filename.

The actual metric emission is a single `print()` at line 1188:

```python
print(f"  step={global_step:5d} loss={comps['loss']:.4f} cos={cos_val:.4f} tps={tps:.1f} "
      f"{tau_str}{gpu_str} gn=[{gn_str}] [{lrs_str}]", flush=True)
```

This single line is the **entire** metric output. There is no:

- `wandb.log({...})` — the project has no W&B integration.
- `SummaryWriter.add_scalar(...)` — no TensorBoard.
- CSV sink — no `csv.writer` writing to a file.
- Structured JSON emitter — no `json.dumps({"step": ..., "loss": ...})` to a `.jsonl` file.

The only structured output is the `[EVAL] step=N cos=X loss=Y` line at 1196, emitted every 250 steps. Even this is interleaved with the per-step prints in the same file, so extracting it requires a regex.

The `sweep_qwen.py` script (213 LOC) is the only consumer of these logs. It runs `train_super_block` in subprocesses, captures stdout, parses the `[EVAL]` lines with a hardcoded regex, and tabulates results. The parsing is so fragile that a single space change in the print statement would silently break the sweep.

The proposed observability stack is detailed in `06_logging_metrics.md`.

---

## 8. Error handling

The training loop has exactly one error-handling mechanism: the NaN check at line 1105:

```python
if not torch.isfinite(loss):
    param_nan = {}
    for name, par in student.named_parameters():
        ...
    print(f"  [step {global_step}] NaN loss — skipping ...", flush=True)
    n_nan_skip += 1
    opt.zero_grad(...)
    del batch_ids, h_out, student_out, loss, comps
    global_step += 1
    continue
```

This is a **skip-and-continue** strategy. When the loss goes to NaN, the step is abandoned, the optimizers are zeroed, and training proceeds to the next batch. There is no:

- **Automatic rollback** to the last known-good checkpoint. The model continues with the NaN-producing parameters, which can poison subsequent steps.
- **Parameter sanity check** after the optimizer step. The clamp at line 1149 (`par.data.clamp_(-20.0, 20.0)`) is the only post-step safety, and it only applies to `index_logits`, not to palettes or LoRA.
- **CUDA error detection.** A CUDA OOM is caught by Python's default exception handler, which prints a traceback and exits. There is no `try/except` around `loss.backward()` or `opt.step()`.
- **Gradient explosion detection** before clipping. The clip at line 1137 caps the gradient norm at 1.0 for indices and 0.3 for others, but if a single parameter has `grad=inf`, the clip silently passes the inf through (PyTorch's `clip_grad_norm_` returns the norm but does not sanitize the individual gradients).

The clamp at line 1149 deserves special attention. The comment explains: "Gumbel-Softmax grad at low tau can push fp32 master to ±1e6, which overflows fp16 (max 65504) → inf → NaN on next forward." This is a **workaround for a known numerical instability** of fp16 logits at low temperature. The correct fix is to store `index_logits` in bf16 (which has the same exponent range as fp32, max ~3.4e38) — but the soft CUDA kernel requires fp16 (`TORCH_CHECK(logits.dtype() == torch::kHalf)` at `fused_lut_linear_cuda.py:235`), so the dtype is locked in by the kernel contract.

---

## 9. Cross-component contract violations

The codebase has several implicit contracts between components that are not enforced by the type system. Each is a latent bug waiting for a refactor to trigger.

### 9.1 `PalettizedLinear` ↔ `fused_lut_linear_cuda` dtype contract

`PalettizedLinear.__init__` at `qwen_model.py:82` creates `self.palette = nn.Parameter(..., dtype=torch.bfloat16)`. The hard CUDA kernel at `fused_lut_linear_cuda.py:81` asserts `palette.dtype() == torch::kBFloat16`. So far so good. But at line 124, `self.index_logits = nn.Parameter(..., dtype=torch.float16)`. The soft kernel at line 235 asserts `logits.dtype() == torch::kHalf` (fp16). The two dtypes are different, and there is no comment explaining why. The `train_qwen.py:1149` clamp is the symptom; the cause is that fp16 cannot hold the unbounded logits that Gumbel-Softmax produces at low temperature.

### 9.2 `QwenLoRA` ↔ `PalettizedLinear` shape contract

`QwenLoRA.__init__` at `qwen_model.py:194` extracts `in_dim` and `out_dim` from `base_module.in_features` and `base_module.out_features`. Both `nn.Linear` and `PalettizedLinear` set these attributes in their `__init__`. But `PalettizedLinear` also stores `indices` with shape `(in_dim, out_dim)` and `palette` with shape `(n_groups, 4)`. If someone constructs a `QwenLoRA` around a `PalettizedLinear` whose `indices` was loaded transposed (the `pre_transposed=True` case), the LoRA matmul will silently use the wrong dimensions. The `pre_transposed` flag is stored on the `PalettizedLinear` but not consulted by `QwenLoRA` — the LoRA assumes the base is always in the natural orientation.

### 9.3 `PartialWrapper.named_modules()` ↔ HuggingFace layer naming

`PartialWrapper.named_modules()` at `qwen_model.py:551–553` yields names like `layers.0`, `layers.0.linear_attn`, `embed_tokens`. The HuggingFace `Qwen3_5ForConditionalGeneration` model uses names like `model.layers.0`, `model.language_model.layers.0.linear_attn`, etc. The mismatch means any code that loads a HF state dict into a `PartialWrapper` (e.g., to resume from a HF checkpoint rather than the custom `.pt` format) will silently fail to match keys. The `load_state` function at `train_qwen.py:796` works around this by manually mapping `layers_N_*` filename patterns back to module paths via `get_submodule` — but this is fragile and only handles the specific naming convention used by `save_state`.

### 9.4 Optimizer ↔ `nn.optim.Optimizer` contract

`FP32MasterOptimizer` at `train_qwen.py:154` is **not** a subclass of `torch.optim.Optimizer`. It exposes `.param_groups`, `.state`, `.step()`, `.zero_grad()` — the four methods that callers actually use — but it does not inherit. Any framework that calls `isinstance(opt, torch.optim.Optimizer)` will return False. This includes:

- `torch.optim.lr_scheduler.LambdaLR(opt)` — works, because it only requires `.param_groups` and `.step()`. But it returns a `LambdaLR` instance that calls `opt.step()` directly, bypassing the fp32 master sync. The codebase handles this by calling `sched_muon.step(epoch=...)` (note the deprecated `epoch` arg) *after* `opt.step()`, but the order matters and is not enforced.
- `accelerate.Accelerator.prepare(opt)` — would wrap the optimizer in an `AcceleratedOptimizer`, which delegates to the inner optimizer. But the inner optimizer is `self.opt` (a real `AdamW`), not `self` — so the accelerate wrapper would call the wrong `.step()` and skip the fp32 master sync.
- HuggingFace `Trainer(opt)` — would reject the optimizer entirely.

The fix is to make `FP32MasterOptimizer` a real `Optimizer` subclass, or to use `torch.optim.Optimizer`'s built-in fp32 master support (which exists since PyTorch 2.1 via the `fused` and `foreach` options).

---

## 10. Dependency graph

The import graph of the codebase is:

```
train_qwen.py
  ├── palettize_core  (BITWIDTH, GROUP_SIZE, PALETTE_SIZE, load_indices, load_lut)
  └── qwen_model     (SUPER_BLOCKS, PalettizedLinear, QwenLoRA,
                      load_qwen_super_block_only, load_qwen_model,
                      insert_correction_layers, attach_lora_to_layer,
                      should_palettize, palettize_linear,
                      capture_original_weights)

qwen_model.py
  ├── palettize_core  (load_indices, load_lut, sanitize_name,
                      BITWIDTH, GROUP_SIZE, PALETTE_SIZE)
  └── [lazy] fused_lut_linear_cuda  (fused_lut_linear, fused_lut_linear_soft)

palettize_core.py
  └── palettize_pytorch  (kmeans1d_weighted, palettize_groups, reconstruct_Wq,
                          sanitize_name, write_lut_scalar,
                          write_metadata_json, pack_indices_transposed)

fused_lut_linear_cuda.py
  └── torch.utils.cpp_extension.load_inline  (compiles fused_lut_kernel.cu)

calib_qwen.py
  ├── palettize_core
  ├── palettize_pytorch
  └── qwen_model  (load_qwen_model, isolate_super_block, ...)

convert_trained_to_packed.py
  ├── palettize_core
  └── qwen_model  (PalettizedLinear, QwenLoRA, SUPER_BLOCKS)
```

Three observations:

1. **`palettize_core` is a shim over `palettize_pytorch`** with a monkey-patch at the module level (lines 50–57). This is the only place in the codebase where a module is mutated at import time. The pattern is fragile and undocumented.
2. **`qwen_model` imports `fused_lut_linear_cuda` lazily** (inside `PalettizedLinear.__init__`, line 110). This is correct — it allows the model to be loaded without a CUDA GPU (for testing on CPU). But the lazy import means that a typo in the kernel module name (`fused_lut_linear_cuda` vs `fused_lut_linear_Cuda`) would only be caught at model-construction time, not at module-load time.
3. **No script imports `train_qwen`.** This is good — the training loop is a leaf, not a library. But it also means the training loop's helper functions (`compute_loss`, `apply_groups`, `build_optimizers`, etc.) cannot be reused by other scripts. `sweep_qwen.py` works around this by `subprocess.Popen(["python", "train_qwen.py", ...])` — a clear violation of the "library first, script second" principle.

---

## 11. Build and deploy

There is no `setup.py`, no `pyproject.toml`, no `requirements.txt`, no `environment.yml`. The dependencies are implicit and discoverable only by reading the imports:

- `torch` (>= 2.1, for `torch.amp.autocast` and `torch.utils.checkpoint`)
- `transformers` (for `AutoModelForCausalLM.from_pretrained`)
- `datasets` (for `load_dataset` streaming)
- `numpy` (used everywhere)
- `tqdm` (transitive via datasets)
- A CUDA toolkit (>= 11.8, for `load_inline` compilation of `fused_lut_kernel.cu`)

The CUDA kernel is compiled **at first import** of `fused_lut_linear_cuda`, via `torch.utils.cpp_extension.load_inline`. The compiled `.so` is cached under `~/.cache/torch_extensions/`. There is no pre-build step, no `Makefile`, no way to compile the kernel without running Python. On a fresh machine, the first `import fused_lut_linear_cuda` takes ~90 seconds (compile time); subsequent imports are sub-second (cache hit).

The targeted GPU architectures are documented in `fused_lut_linear_cuda.py:22`: `sm_89` (L4 / Ada Lovelace), `sm_80` (A100), `sm_90` (H100). The actual training log shows the run is on a Blackwell-class GPU (commit `5446edf`: "Blackwell sm_120 working version"), which requires `sm_120` — not in the default flag list. The codebase has been patched to add `sm_120` somewhere (visible in the working state), but the documentation in `fused_lut_linear_cuda.py` was not updated. This is a documentation defect, not a code defect, but it would confuse a new contributor trying to reproduce the build.

---

## 12. Summary of audit findings

The system **works** — it produces cos=0.953 on super-block 0, which is a meaningful research result. But it works **despite** its architecture, not **because of** it. The architecture has three load-bearing defects, in order of severity:

1. **`PartialWrapper` is not `nn.Module`** (issue #1). This blocks `torch.compile`, gradient checkpointing, FSDP, state dict save/load, and HuggingFace framework integration. It is the root cause of issues #3 (teacher/student duplication cannot be cleanly resolved), #4 (no gradient checkpointing), #6 (no `torch.compile` speedup), and #10 (export pipeline is detached because `state_dict()` is unavailable).

2. **The training loop is a monolith** (issue #2). This blocks unit testing, hyperparameter sweeps, A/B comparisons, and the integration of any external framework (`accelerate`, `transformers.Trainer`, `lightning`). It is the root cause of issues #7 (no structured logging — the log statements are interleaved with the loop body and cannot be extracted), #8 (no tests — the loop cannot be instantiated without a GPU and a teacher), and #9 (three config sources — the loop reads from all three at different points, and there is no single object that represents "the configuration").

3. **The dtype contract is incoherent** (issue #5). The codebase mixes bf16, fp16, and fp32 across components with no master/slave relationship, forcing the `FP32MasterOptimizer` workaround (which itself violates the `Optimizer` contract). This is the root cause of the NaN-clamp workaround at line 1149 and the inability to use `torch.compile` mode `"reduce-overhead"` (which requires a single dtype per region).

The remaining issues (#6 data pipeline, #7 logging, #8 tests, #9 config, #10 export) are consequences of the above three, not independent defects. Fixing the top three unlocks the rest.
