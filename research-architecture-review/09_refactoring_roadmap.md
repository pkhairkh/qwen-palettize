# 09 — Refactoring Roadmap: Phased Plan to Fix the Architecture

## 1. Overview

This document is the **execution plan** for the architecture review. It converts the diagnoses in documents `00`–`08` into a sequence of concrete refactoring tasks, with time estimates, dependencies, verification criteria, and rollback plans.

The roadmap has **four phases**, each phase delivering a self-contained improvement that can be merged independently. The total effort is **~14.5 engineering days** for a single developer, or ~7 days for two developers working in parallel (with the dependencies respected).

### 1.1 Phase summary

| Phase | Duration | Deliverables | Risk |
|---|---|---|---|
| Phase 1: Foundation | 3 days | `PartialWrapper` → `nn.Module` fix; `Config` dataclass; `MetricsLogger` multi-backend | Low (no behavior change) |
| Phase 2: Throughput | 4 days | Gradient checkpointing; in-place teacher; lazy `_flat_idx`; tokenization cache | Medium (behavior changes, but recoverable) |
| Phase 3: Modularization | 4 days | Extract `qwen_palettize/` package; `Trainer` class; thin CLI shim | Medium (large diff, but mechanical) |
| Phase 4: Quality | 3.5 days | Test scaffold (4 tiers); CI/CD pipeline; mixed-precision unification | Low (additive, no behavior change) |
| **Total** | **14.5 days** | | |

Each phase is independently mergeable. Each phase preserves the cos=0.953 baseline (verified by a regression test that runs at the end of each phase). The phases are ordered by dependency: Phase 2's gradient checkpointing requires Phase 1's `PartialWrapper` fix; Phase 3's modularization benefits from Phase 1's `Config` and `MetricsLogger`; Phase 4's tests verify Phase 3's modularization.

### 1.2 Slack allocation

The 14.5-day estimate includes 1.5 days of slack for unanticipated complications (CUDA kernel recompilation issues, dtype-mismatch bugs surfaced by the refactor, regression in cos due to a subtle behavior change). The slack is allocated to Phase 2 (1 day) and Phase 3 (0.5 day), which are the highest-risk phases.

### 1.3 Prerequisites

The roadmap assumes:

- One developer with PyTorch 2.x experience.
- One Blackwell-class GPU (or A100/H100) for verification runs.
- `git` access to the `qwen-palettize` repo.
- `pytest` and `wandb` accounts set up.
- No concurrent feature work on the `train_qwen.py` or `qwen_model.py` files during the refactor.

---

## 2. Phase 1: Foundation (3 days)

### 2.1 Goal

Establish the foundation for all subsequent work: fix the `PartialWrapper` (so `nn.Module`-dependent features work), introduce the `Config` dataclass (single source of truth), and introduce the `MetricsLogger` (multi-backend logging). After Phase 1, the cos=0.953 baseline is preserved bit-for-bit, but the codebase is ready for `torch.compile`, FSDP, and the modularization in Phase 3.

### 2.2 Tasks

#### Task 1.1 — Fix `PartialWrapper` and `PartialModel` (1 day)

**Files**: `scripts/qwen_model.py` (lines 476–571).

**What to do**:

1. Change `class PartialModel:` to `class PartialModel(nn.Module):`.
2. Change `class PartialWrapper:` to `class PartialWrapper(nn.Module):`.
3. In `PartialModel.__init__`, call `super().__init__()` first.
4. Change `self.layers = layers` to `self.layers = nn.ModuleList(layers)`.
5. Add a `forward(self, input_ids, position_ids=None)` method to `PartialModel` that runs the standard layer loop.
6. In `PartialWrapper.__init__`, call `super().__init__()` first.
7. Add a `forward(self, input_ids, position_ids=None)` method to `PartialWrapper` that delegates to `self.model(...)`.
8. Delete the hand-rolled `parameters()`, `named_parameters()`, `named_modules()`, `to()`, `eval()`, `train()`, `get_submodule()` methods (they are now inherited from `nn.Module`).

**Verification**:

- Run `python scripts/train_qwen.py --sb_idx 0 --max_steps 100 --seq_len 512 --batch_size 32` and verify cos at step 100 matches the pre-refactor baseline (within 1e-4 tolerance, accounting for nondeterminism).
- Verify `torch.compile(student)` no longer raises (smoke test).
- Verify `student.state_dict()` no longer raises (returns a non-empty `OrderedDict`).

**Rollback**: `git revert` the commit. The change is mechanical and localized to two classes, so a revert is safe.

**Dependencies**: None.

#### Task 1.2 — Introduce `Config` dataclass (1 day)

**Files**: new file `scripts/qwen_palettize/__init__.py`, new file `scripts/qwen_palettize/config.py`.

**What to do**:

1. Create the `qwen_palettize` package directory.
2. Implement the `Config`, `GroupsConfig`, `LrsConfig`, `LossWeightsConfig` dataclasses as in `05_training_loop_refactor.md` §3.1.
3. Add `Config.from_cli(args)` and `Config.merge_from_json(path)` methods.
4. Add `Config.to_json()` and `Config.signature()` methods.
5. Modify `scripts/train_qwen.py:main()` to construct a `Config` from CLI args, but otherwise use the same training loop. The loop still reads `cfg.hyperparams_file` via `merge_from_json` every 10 steps (preserving the live hot-reload behavior).

**Verification**:

- Run `python scripts/train_qwen.py --sb_idx 0 --max_steps 100 ...` and verify cos matches the baseline.
- Run `pytest scripts/qwen_palettize/tests/unit/test_config.py` (the unit tests from `07_testing_ci.md` §3.1).

**Rollback**: `git revert` the commit. The `Config` is used only inside `train_qwen.py`, so a revert restores the original `DEFAULT_HYPERPARAMS` + CLI args + JSON behavior.

**Dependencies**: None.

#### Task 1.3 — Introduce `MetricsLogger` (1 day)

**Files**: new file `scripts/qwen_palettize/log.py`.

**What to do**:

1. Implement the `MetricsLogger` class with `StdoutBackend`, `WandbBackend`, `TensorboardBackend`, `CsvBackend` as in `06_logging_metrics.md` §2–3.
2. Modify `scripts/train_qwen.py:train_super_block()` to construct a `MetricsLogger` and route all `print()` statements through it.
3. Add `--log_backend` CLI flag (default `"stdout"`).
4. Verify the `StdoutBackend` output is byte-identical to the current `print()` output (modulo the timestamp, which the current code does not include).

**Verification**:

- Run with `--log_backend stdout` and `diff` the output against a pre-refactor run. They should be identical.
- Run with `--log_backend csv` and verify a `metrics.csv` file is written with the expected columns.
- Run with `--log_backend wandb` (with `WANDB_API_KEY` set) and verify a W&B run is created with the expected metrics.

**Rollback**: `git revert` the commit. The `MetricsLogger` is a thin wrapper; reverting restores the direct `print()` calls.

**Dependencies**: Task 1.2 (the `MetricsLogger` reads `cfg.log_backend`).

### 2.3 Phase 1 DoD

- [ ] `PartialWrapper` is an `nn.Module` subclass.
- [ ] `torch.compile(student)` does not raise.
- [ ] `student.state_dict()` does not raise.
- [ ] `Config` dataclass is the single source of truth for hyperparameters.
- [ ] `MetricsLogger` supports stdout, wandb, tensorboard, csv backends.
- [ ] cos=0.953 baseline preserved (regression test passes).
- [ ] Unit tests for `Config` pass.

---

## 3. Phase 2: Throughput (4 days)

### 3.1 Goal

Unlock the throughput improvements that Phase 1 made possible: gradient checkpointing (4× batch size), in-place teacher (2 GB VRAM saved), lazy `_flat_idx` (3.56 GB saved), and tokenization cache (2× step-time speedup). After Phase 2, the training step is ~2.5× faster and uses ~50% less VRAM.

### 3.2 Tasks

#### Task 2.1 — Lazy-allocate `_flat_idx` (0.5 day)

**Files**: `scripts/qwen_model.py:PalettizedLinear.__init__` (lines 97–102).

**What to do**:

1. Move the `_flat_idx` allocation from `__init__` to a `_get_flat_idx()` method that allocates on first call.
2. In the fallback path of `forward` (line 156), call `self._get_flat_idx()` instead of accessing `self._flat_idx` directly.
3. The CUDA path does not use `_flat_idx`, so the lazy allocation means the cache is never created when the CUDA kernel is available.

**Verification**:

- Run training with `USE_TC_FWD=1` (CUDA kernel enabled) and verify `_flat_idx` is never allocated (check via `torch.cuda.memory_allocated()` before and after model construction).
- Verify the fallback path (set `USE_TC_FWD=0`) still works correctly.

**Rollback**: `git revert`. The change is localized to `PalettizedLinear`.

**Dependencies**: None.

#### Task 2.2 — In-place teacher (1 day)

**Files**: `scripts/qwen_model.py:load_qwen_super_block_only` (lines 451–581), `scripts/train_qwen.py:train_super_block` (lines 987–994).

**What to do**:

1. Add a new function `load_teacher_and_student(sb_idx, ...)` that:
   - Loads the prefix once via `from_pretrained`.
   - Freezes it (the teacher).
   - Builds the student by `deepcopy`-ing the teacher (or by sharing `embed_tokens`, `rotary_emb`, `norm` and copying only the active layers).
   - Palettizes the student's active layers in-place.
2. Replace the two separate `load_qwen_super_block_only` calls in `train_super_block` with a single `load_teacher_and_student` call.
3. Delete the `student.model.embed_tokens = teacher.model.embed_tokens` line (the sharing is now done at construction time).
4. Add explicit `gc.collect()` and `torch.cuda.empty_cache()` after the in-place palettization.

**Verification**:

- Run training and verify `torch.cuda.memory_allocated()` after model construction is ~2.16 GB lower than the pre-refactor baseline.
- Verify the cos at step 100 matches the baseline.

**Rollback**: `git revert`. The change is localized to the model-loading path.

**Dependencies**: Task 1.1 (the in-place teacher requires `nn.Module` semantics for the layer copying).

#### Task 2.3 — Enable gradient checkpointing (1 day, + 0.5 day slack)

**Files**: `scripts/train_qwen.py:train_super_block` (lines 1091–1098), new `--use_gradient_checkpointing` CLI flag.

**What to do**:

1. Add `use_gradient_checkpointing` field to the `Config` dataclass.
2. Add `--use_gradient_checkpointing` CLI flag (default 0).
3. In the student forward loop, replace:
   ```python
   out = layer(s_h, position_embeddings=s_pos_emb)
   ```
   with:
   ```python
   if self.cfg.use_gradient_checkpointing and self.student.training:
       out = torch.utils.checkpoint.checkpoint(layer, s_h, s_pos_emb, use_reentrant=False)
   else:
       out = layer(s_h, position_embeddings=s_pos_emb)
   ```
4. Run training with `--use_gradient_checkpointing 1 --batch_size 128` and verify it does not OOM.
5. Verify the cos at step 100 matches the baseline (within tolerance — gradient checkpointing can introduce minor numerical differences due to recomputation).

**Verification**:

- Run training with `--batch_size 128` and verify it fits in VRAM (the pre-refactor baseline would OOM at batch=64).
- Verify cos at step 100 is within 1% of the baseline.

**Rollback**: `git revert`. The change is gated by a flag; reverting restores the non-checkpointed forward.

**Dependencies**: Task 1.1 (the `PartialWrapper` fix is required for `torch.utils.checkpoint` to work with the model).

**Risk**: This is the highest-risk task in Phase 2. The `use_reentrant=False` flag is critical for compatibility with the soft kernel's custom autograd Function. If the soft kernel's backward does not work with `use_reentrant=False`, the gradient checkpointing must be disabled for the soft path (and only enabled for the hard path, which is used during eval). The slack day in Phase 2 is for debugging this interaction.

#### Task 2.4 — Tokenization cache (1 day)

**Files**: new file `scripts/cache_tokens_v2.py`, `scripts/train_qwen.py:stream_training_data` (lines 889–918).

**What to do**:

1. Implement `scripts/cache_tokens_v2.py` as in `04_data_pipeline.md` §6.4.
2. Modify `stream_training_data` to check for a `cache_dir` argument; if provided and the manifest exists, stream from the cache. Otherwise, fall back to the current HF streaming + per-step tokenization.
3. Add `--data_cache_dir` CLI flag (default None).
4. Pre-tokenize a 1 BT subset (~8 GB on disk) using `cache_tokens_v2.py`.

**Verification**:

- Run training with `--data_cache_dir /data/fineweb_edu_seq512_bs32` and verify the per-step time drops from ~580 ms to ~280 ms.
- Verify cos at step 100 matches the baseline (the data is the same, just pre-tokenized).

**Rollback**: `git revert`. The change is gated by a flag; reverting restores the streaming path.

**Dependencies**: None.

### 3.3 Phase 2 DoD

- [ ] `_flat_idx` is lazy-allocated; saves ~3.56 GB VRAM.
- [ ] Teacher and student are loaded once (in-place); saves ~2.16 GB VRAM.
- [ ] `--use_gradient_checkpointing 1` enables gradient checkpointing; batch=128 is feasible.
- [ ] `--data_cache_dir` enables pre-tokenized cache; step time is ~280 ms.
- [ ] cos=0.953 baseline preserved (regression test passes, with 1% tolerance for gradient-checkpointing numerical differences).
- [ ] Per-step time is ~280 ms (vs ~580 ms pre-refactor).

---

## 4. Phase 3: Modularization (4 days)

### 4.1 Goal

Extract the monolithic `train_super_block` function into the `qwen_palettize/` package with 10 modules, as proposed in `05_training_loop_refactor.md` §2. After Phase 3, the codebase is unit-testable, A/B-testable, and composable.

### 4.2 Tasks

#### Task 3.1 — Extract `model.py`, `optim.py`, `loss.py`, `anneal.py` (2 days)

**Files**: new files in `scripts/qwen_palettize/`.

**What to do**:

1. Move `PalettizedLinear`, `QwenLoRA`, `PartialModel`, `PartialWrapper` from `qwen_model.py` to `qwen_palettize/model.py`. These are the (already-fixed, post-Phase-1) `nn.Module` subclasses.
2. Move `Muon`, `FP32MasterOptimizer`, `FP32MasterAdamW`, `FP32MasterMuon` from `train_qwen.py` to `qwen_palettize/optim.py`.
3. Move `compute_loss`, `normalize_weights` from `train_qwen.py` to `qwen_palettize/loss.py`.
4. Move `update_tau`, `freeze_settled_palettes`, `snapshot_palette_indices`, `inject_gradient_noise`, `get_loss_type_for_step` from `train_qwen.py` to `qwen_palettize/anneal.py`.
5. Keep `qwen_model.py` and `train_qwen.py` as thin re-export shims (for backward compatibility with `calib_qwen.py` and other scripts).

**Verification**:

- Run training and verify cos matches the baseline.
- Run `pytest scripts/qwen_palettize/tests/component/` (the component tests from `07_testing_ci.md` §4).

**Rollback**: `git revert`. The shims in `qwen_model.py` and `train_qwen.py` ensure backward compatibility, so a revert restores the original monolithic structure.

**Dependencies**: Task 1.1 (the model extraction requires the `nn.Module` fix).

#### Task 3.2 — Extract `data.py`, `checkpoint.py`, `evaluate.py` (1 day)

**Files**: new files in `scripts/qwen_palettize/`.

**What to do**:

1. Move `stream_training_data` to `qwen_palettize/data.py`. Add the `cache_dir` argument.
2. Move `save_state`, `load_state` to `qwen_palettize/checkpoint.py`. Add the legacy-checkpoint migration function.
3. Move `prepare_eval_set`, `evaluate` to `qwen_palettize/evaluate.py`.

**Verification**:

- Run training and verify cos matches the baseline.
- Run `pytest scripts/qwen_palettize/tests/component/test_checkpoint.py` (the round-trip test from `07_testing_ci.md` §4.3).

**Rollback**: `git revert`.

**Dependencies**: Task 3.1 (the `data.py` and `checkpoint.py` modules depend on `model.py`).

#### Task 3.3 — Extract `Trainer` class (1 day)

**Files**: new file `scripts/qwen_palettize/train.py`.

**What to do**:

1. Implement the `Trainer` class as in `05_training_loop_refactor.md` §4.1.
2. Modify `scripts/train_qwen.py:main()` to construct a `Trainer` and call `trainer.run()`.
3. `train_qwen.py` shrinks from 1,266 LOC to ~50 LOC.

**Verification**:

- Run training and verify cos matches the baseline.
- Run `pytest scripts/qwen_palettize/tests/integration/test_train_one_step.py` (the integration test from `07_testing_ci.md` §5.1).

**Rollback**: `git revert`. The `Trainer` class is a new file; the old `train_super_block` function is restored.

**Dependencies**: Tasks 3.1, 3.2, 1.2 (the `Trainer` uses `Config`), 1.3 (the `Trainer` uses `MetricsLogger`).

### 4.3 Phase 3 DoD

- [ ] `scripts/qwen_palettize/` package exists with 10 modules.
- [ ] `scripts/train_qwen.py` is < 100 LOC (was 1,266).
- [ ] Each module has a clear interface and a unit/component test.
- [ ] `Trainer` class is subclassable for research variants.
- [ ] cos=0.953 baseline preserved (regression test passes).
- [ ] Integration tests pass (Tier 3).

---

## 5. Phase 4: Quality (3.5 days)

### 5.1 Goal

Add the test scaffold, CI/CD pipeline, and mixed-precision unification that complete the architecture refactor. After Phase 4, the codebase is "industry-standard" by the criteria of `08_literature_comparison.md`.

### 5.2 Tasks

#### Task 4.1 — Test scaffold (2 days)

**Files**: new files in `scripts/qwen_palettize/tests/`.

**What to do**:

1. Implement the 4-tier test scaffold as in `07_testing_ci.md` §2:
   - Tier 1 (unit): `test_config.py`, `test_loss.py`, `test_anneal.py`, `test_palettize_core.py`, `test_packing.py`.
   - Tier 2 (component): `test_model.py`, `test_optim.py`, `test_checkpoint.py`, `test_log.py`.
   - Tier 3 (integration): `test_train_one_step.py`, `test_checkpoint_resume.py`, `test_eval.py`.
   - Tier 4 (regression): `test_full_run.py`.
2. Implement `conftest.py` with shared fixtures (tiny model, tiny teacher, temporary directories).
3. Add `pytest.ini` at the repo root with markers: `@pytest.mark.regression` for Tier 4, `@pytest.mark.cuda` for tests requiring CUDA.

**Verification**:

- Run `pytest scripts/qwen_palettize/tests/unit/ -v` — all tests pass in < 5 seconds.
- Run `pytest scripts/qwen_palettize/tests/component/ -v -k "not cuda"` — all CPU-only tests pass in < 30 seconds.
- Run `pytest scripts/qwen_palettize/tests/integration/ -v` (with GPU) — all tests pass in < 60 seconds.
- Run `pytest scripts/qwen_palettize/tests/regression/ -v -m regression` (nightly) — passes within 78 minutes.

**Rollback**: `git revert`. The tests are additive; reverting removes them without affecting production code.

**Dependencies**: Tasks 3.1, 3.2, 3.3 (the tests target the modularized code).

#### Task 4.2 — CI/CD pipeline (1 day)

**Files**: new file `.github/workflows/test.yml`.

**What to do**:

1. Implement the GitHub Actions workflow as in `07_testing_ci.md` §7.1.
2. Set up the self-hosted GPU runner (one-time setup).
3. Configure the nightly regression test schedule.
4. Add a pre-commit hook that runs Tier 1 tests.

**Verification**:

- Push a commit and verify the CI runs Tier 1 + Tier 2 (CPU) tests.
- Open a PR and verify the CI runs Tier 3 (GPU) tests.
- Wait for the nightly schedule and verify the CI runs Tier 4 (regression).

**Rollback**: `git revert` the workflow file. The pre-commit hook is removed by reverting the `.pre-commit-config.yaml`.

**Dependencies**: Task 4.1 (the CI runs the tests).

#### Task 4.3 — Mixed-precision unification (0.5 day)

**Files**: `scripts/qwen_palettize/model.py:PalettizedLinear.__init__` (line 124), `scripts/fused_lut_linear_cuda.py` (line 235).

**What to do**:

1. Change `index_logits` from fp16 to bf16 in `PalettizedLinear.__init__`.
2. Update the soft CUDA kernel's `TORCH_CHECK(logits.dtype() == torch::kHalf)` to also accept `torch::kBFloat16`. (This requires recompiling the kernel.)
3. Remove the `±20` clamp at `train_qwen.py:1149` (no longer needed — bf16 has the same exponent range as fp32).
4. Update the `FP32MasterOptimizer` to skip the fp32 master for LoRA (LoRA's `lora_A` and `lora_B` are small enough to use bf16 directly).

**Verification**:

- Run training and verify cos matches the baseline.
- Verify no `NaN` losses occur in the first 1000 steps (the fp16 overflow that the clamp was working around should not occur in bf16).

**Rollback**: `git revert`. The change is localized to the dtype and the clamp.

**Dependencies**: None (can be done independently, but is conceptually part of Phase 4).

### 5.3 Phase 4 DoD

- [ ] 4-tier test scaffold exists, with all tests passing.
- [ ] CI/CD pipeline runs Tier 1+2 on every push, Tier 3 on every PR, Tier 4 nightly.
- [ ] `index_logits` is bf16 (not fp16).
- [ ] `±20` clamp is removed.
- [ ] LoRA's fp32 master is removed.
- [ ] cos=0.953 baseline preserved (regression test passes).

---

## 6. Timeline and milestones

```
Day 1   Phase 1.1: PartialWrapper → nn.Module           [3 days of Phase 1]
Day 2   Phase 1.2: Config dataclass
Day 3   Phase 1.3: MetricsLogger                          ★ Phase 1 complete
Day 4   Phase 2.1: Lazy _flat_idx                        [4 days of Phase 2]
Day 5   Phase 2.2: In-place teacher
Day 6   Phase 2.3: Gradient checkpointing                 ★ batch=128 feasible
Day 7   Phase 2.4: Tokenization cache                    ★ Phase 2 complete
Day 8   Phase 3.1: Extract model/optim/loss/anneal       [4 days of Phase 3]
Day 9   (cont.)
Day 10  Phase 3.2: Extract data/checkpoint/evaluate
Day 11  Phase 3.3: Extract Trainer class                 ★ Phase 3 complete
Day 12  Phase 4.1: Test scaffold                         [3.5 days of Phase 4]
Day 13  (cont.)
Day 14  Phase 4.2: CI/CD pipeline
Day 14.5 Phase 4.3: Mixed-precision unification          ★ Phase 4 complete
```

### 6.1 Milestones

- **Day 3**: cos=0.953 reproduces with `nn.Module`-based `PartialWrapper`. `torch.compile` no longer raises.
- **Day 6**: batch=128 fits in VRAM with gradient checkpointing. Step time is ~280 ms with tokenization cache.
- **Day 11**: `train_qwen.py` is < 100 LOC. `qwen_palettize/` package has 10 modules. `Trainer` class is subclassable.
- **Day 14.5**: 4-tier test scaffold passes. CI/CD pipeline runs nightly regression. Mixed precision is unified.

### 6.2 Critical path

The critical path is:

```
  Phase 1.1 (PartialWrapper)  →  Phase 2.3 (checkpointing)  →  batch=128
```

If Phase 1.1 is delayed by a day, Phase 2.3 is delayed by a day, and the batch=128 milestone shifts by a day. The other tasks have slack — Phase 1.2, 1.3, 2.1, 2.2, 2.4 can be reordered within their phases.

---

## 7. Verification strategy

Each phase ends with a verification step:

1. **Phase 1**: Run `python scripts/train_qwen.py --sb_idx 0 --max_steps 100 ...`. Verify cos at step 100 matches the pre-refactor baseline (within 1e-4 tolerance).
2. **Phase 2**: Same as Phase 1, but with `--use_gradient_checkpointing 1 --batch_size 128 --data_cache_dir /data/...`. Verify cos matches the baseline (within 1% tolerance, accounting for gradient-checkpointing numerical differences).
3. **Phase 3**: Run `pytest scripts/qwen_palettize/tests/`. All tests pass.
4. **Phase 4**: Verify the nightly regression test passes (cos >= 0.94 at step 8000).

### 7.1 The "regression baseline" file

A file `tests/regression/baseline.json` records the cos=0.953 baseline:

```json
{
  "version": 1,
  "git_commit": "b82a6be",
  "config": {
    "sb_idx": 0,
    "max_steps": 8000,
    "seq_len": 512,
    "batch_size": 32,
    "lora_rank": 16,
    "use_soft_indices": true,
    "tau_init": 2.0,
    "tau_final": 0.1,
    "tau_anneal_steps": 4000
  },
  "expected_cos": 0.9530,
  "tolerance": 0.013
}
```

The regression test reads this file and verifies `observed_cos >= expected_cos - tolerance`. If the refactor is correct, the test passes. If the refactor introduces a behavior change, the test fails — and the developer must either fix the regression or update the baseline (with a justification).

### 7.2 Continuous verification

The regression test runs nightly on the GPU runner. If it fails, the team is alerted via the CI/CD pipeline's notification mechanism (e.g., a Slack webhook, a GitHub issue auto-created).

---

## 8. Risk register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| `torch.utils.checkpoint` incompatible with the soft kernel's `autograd.Function` | Medium | High (blocks Phase 2.3) | Slack day allocated; if it fails, disable checkpointing for the soft path (only enable for hard path during eval) |
| `index_logits` in bf16 produces NaN (the original reason for fp16 + clamp) | Low | Medium (blocks Phase 4.3) | Verify on a 1000-step run before removing the clamp; if NaN occurs, keep the clamp but switch to bf16 anyway |
| `state_dict()` key mismatch with legacy `.pt` checkpoints | Medium | Medium (breaks resume) | Write a migration script that converts legacy checkpoints to the new `state_dict` format; run it once per checkpoint |
| `Trainer` class refactor introduces a subtle behavior change (e.g., optimizer step order) | Medium | High (cos regression) | Phase 3 verification is a 1000-step run; if cos drifts, bisect the commits within Phase 3 |
| Self-hosted GPU runner unavailable | Medium | Medium (blocks Phase 4 CI) | Use a cloud GPU (Lambda Labs, RunPod, AWS p4d) as a fallback |
| Developer unavailable (sick, vacation) | Low | Low (delays by 1 week) | Roadmap is single-developer; a second developer can pick up by reading the docs |

---

## 9. Post-refactor outlook

After Phase 4, the qwen-palettize codebase is "industry-standard" by the criteria of `08_literature_comparison.md`. The team can then focus on the research goal (cos=0.999) without being blocked by architecture issues:

- **Larger batches** (Phase 2.3) → more stable gradients → potentially higher cos.
- **Faster steps** (Phase 2.4) → more steps per hour → faster iteration on hyperparameters.
- **Modular code** (Phase 3) → A/B test variants → faster experimentation.
- **Test scaffold** (Phase 4) → 60-second regression check → fearless refactoring.
- **CI/CD** (Phase 4) → automated quality gate → no silent regressions.

The cos=0.999 target in `SPEC.md` §6.2 is **unreachable** without these improvements, because:

- The current batch=32 produces noisy gradients that limit the cos achievable in 8000 steps.
- The current 1.5-hour-per-regression-check iteration speed makes hyperparameter search impractical.
- The current monolithic code makes it impossible to try architecture variants (e.g., different LoRA ranks, different temperature schedules) without copy-pasting the 1,266-line training function.

The 14.5-day refactor investment is the **prerequisite** for the cos=0.999 target. Without it, the team is iterating on a codebase that cannot scale to the research ambition.

---

## 10. Summary

The roadmap is:

- **Phase 1 (3 days)**: Fix `PartialWrapper`, introduce `Config`, introduce `MetricsLogger`. Foundation for everything else.
- **Phase 2 (4 days)**: Lazy `_flat_idx`, in-place teacher, gradient checkpointing, tokenization cache. 2.5× speedup, 50% VRAM reduction.
- **Phase 3 (4 days)**: Extract `qwen_palettize/` package with 10 modules. Modular, testable, composable.
- **Phase 4 (3.5 days)**: 4-tier test scaffold, CI/CD pipeline, mixed-precision unification. Industry-standard quality.

Each phase is independently mergeable. Each phase preserves the cos=0.953 baseline. The total is 14.5 days, with 1.5 days of slack for unanticipated complications. The critical path is Phase 1.1 → Phase 2.3 (gradient checkpointing requires the `nn.Module` fix), and the batch=128 milestone is on Day 6.

The roadmap is the **execution plan** for the architecture review. It is ready to begin on Day 1.
