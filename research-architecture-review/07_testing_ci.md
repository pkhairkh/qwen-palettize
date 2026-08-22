# 07 — Testing and CI: pytest, Regression Tests, CI/CD

## 1. The current state

The qwen-palettize repository has **zero automated tests**. There is no `pytest`, no `unittest.TestCase`, no `tests/` directory, no `.github/workflows/`. The closest things to tests are:

- `scripts/test_palettized_v2.py` (213 LOC) — a manual smoke script that prints assertions to stdout and exits. It does not use a test runner, has no fixtures, has no parametrization, and has no coverage report. The file name starts with `test_`, which would make `pytest` discover it — but `pytest` is not installed and the file is not structured as a `pytest` test (it has top-level `print` statements, not `def test_*` functions).
- `scripts/verify_palettize_core.py` (167 LOC) — similar manual smoke test for the compression round-trip.
- `scripts/verify_2bit_format.py` (198 LOC) — similar manual smoke test for the 2-bit packing.
- `scripts/bench_palettized_v2.py` (113 LOC) — a microbenchmark, not a test.
- `scripts/profile_training.py` (255 LOC) — a profiling script that runs the training loop with `torch.profiler`, not a test.

The user message confirms the operational cost: "Every change is tested by running 8000 steps." At the current ~580 ms/step (per `train_sb0.log` and commit `b82a6be`), this is **~78 minutes per regression check**. There is no faster feedback loop. There is no way to verify that a refactor preserves the cos=0.953 baseline without running a full training run.

This document proposes a test scaffold with four tiers, each progressively slower but more comprehensive:

| Tier | What it tests | Time | When to run |
|---|---|---|---|
| Tier 1: Unit | Pure functions, no GPU | < 5 seconds | Every commit (pre-commit hook + CI) |
| Tier 2: Component | Single module with mocked dependencies | < 30 seconds | Every commit (CI) |
| Tier 3: Integration | Real model build + 1 training step on GPU | < 60 seconds | Every PR (CI, GPU runner) |
| Tier 4: Regression | Full training run, baseline cos | ~78 minutes | Nightly (CI, scheduled) |

The full test suite (Tiers 1–3) runs in under 2 minutes. Tier 4 is the safety net for refactors.

---

## 2. The test directory structure

```
scripts/
└── qwen_palettize/
    └── tests/                           ← new
        ├── __init__.py
        ├── conftest.py                   ← shared fixtures
        ├── unit/
        │   ├── test_config.py
        │   ├── test_loss.py
        │   ├── test_anneal.py
        │   ├── test_palettize_core.py
        │   └── test_packing.py
        ├── component/
        │   ├── test_model.py            ← PalettizedLinear, QwenLoRA on tiny tensors
        │   ├── test_optim.py             ← Muon, FP32MasterAdamW on tiny tensors
        │   ├── test_checkpoint.py        ← save_state, load_state round-trip
        │   └── test_log.py               ← MetricsLogger backends
        └── integration/
            ├── test_train_one_step.py    ← Full Trainer, 1 step, tiny model
            ├── test_eval.py              ← evaluate() on tiny model
            └── test_checkpoint_resume.py ← save → load → 1 step, verify cos matches
```

The `tests/` directory is inside the `qwen_palettize/` package (not at the repo root) so that `from qwen_palettize.tests.conftest import ...` works as a relative import. The tests are discovered by `pytest scripts/qwen_palettize/tests/` from the repo root.

---

## 3. Tier 1: Unit tests (pure functions)

Unit tests cover pure functions — no GPU, no model, no I/O. They run in milliseconds.

### 3.1 `test_config.py`

```python
# scripts/qwen_palettize/tests/unit/test_config.py
import json, os, tempfile
import pytest
from qwen_palettize.config import Config, GroupsConfig, LrsConfig

def test_default_config():
    cfg = Config()
    assert cfg.sb_idx == 0
    assert cfg.max_steps == 5000
    assert cfg.lora_rank == 16
    assert cfg.use_soft_indices is True
    assert cfg.tau_init == 2.0
    assert cfg.tau_final == 0.1

def test_config_from_cli_args():
    """Config.from_cli should populate from argparse.Namespace."""
    import argparse
    args = argparse.Namespace(
        sb_idx=3, max_steps=10000, lora_rank=32, lora_alpha=64,
        seq_len=1024, batch_size=64, resume_from="/path/to/ckpt",
        use_soft_indices=1, tau_init=1.5, tau_final=0.05,
        tau_anneal_steps=2000, shutdown_on_done=0,
        use_gradient_checkpointing=1, use_torch_compile=0,
        use_async_data_loader=0, data_cache_dir="/data/cache",
        log_backend="wandb", wandb_project="qwen", wandb_run_name="test",
    )
    cfg = Config.from_cli(args)
    assert cfg.sb_idx == 3
    assert cfg.max_steps == 10000
    assert cfg.use_gradient_checkpointing is True
    assert cfg.log_backend == "wandb"

def test_config_merge_from_json_typo():
    """A typo in the JSON (e.g., 'palattes' instead of 'palettes') should raise."""
    cfg = Config()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"lrs": {"palattes": 3e-3, "lora": 1e-3}}, f)
        path = f.name
    try:
        with pytest.raises(TypeError):
            cfg.merge_from_json(path)
    finally:
        os.unlink(path)

def test_config_signature_stability():
    """The same config should produce the same signature."""
    cfg1 = Config(sb_idx=0, max_steps=1000, lora_rank=16)
    cfg2 = Config(sb_idx=0, max_steps=1000, lora_rank=16)
    assert cfg1.signature() == cfg2.signature()

def test_config_signature_changes_on_diff():
    """Different configs should produce different signatures."""
    cfg1 = Config(sb_idx=0)
    cfg2 = Config(sb_idx=1)
    assert cfg1.signature() != cfg2.signature()
```

### 3.2 `test_loss.py`

```python
# scripts/qwen_palettize/tests/unit/test_loss.py
import torch
import pytest
from qwen_palettize.loss import compute_loss, normalize_weights
from qwen_palettize.config import Config

def test_compute_loss_cos_only():
    cfg = Config(loss_type="1-cos", loss_weights={"cos": 1.0, "mse": 0.0})
    s = torch.randn(4, 8, 16)
    t = torch.randn(4, 8, 16)
    loss, comps = compute_loss(s, t, cfg)
    assert loss.item() >= 0
    assert "cos" in comps
    assert "loss" in comps

def test_compute_loss_perfect_match():
    """If student == teacher, loss should be 0."""
    cfg = Config(loss_type="norm_mse", loss_weights={"cos": 0.0, "mse": 1.0})
    t = torch.randn(4, 8, 16)
    loss, comps = compute_loss(t.clone(), t, cfg)
    assert loss.item() < 1e-6

def test_compute_loss_zero_norm_student():
    """Zero-norm student output should not raise (eps=1e-4)."""
    cfg = Config(loss_type="1-cos+norm_mse", loss_weights={"cos": 0.5, "mse": 0.5})
    s = torch.zeros(4, 8, 16)
    t = torch.randn(4, 8, 16)
    loss, comps = compute_loss(s, t, cfg)
    assert torch.isfinite(loss)

def test_normalize_weights():
    assert normalize_weights({"a": 0.3, "b": 0.7}) == {"a": 0.3, "b": 0.7}
    assert normalize_weights({"a": 1.0, "b": 1.0}) == {"a": 0.5, "b": 0.5}
    assert normalize_weights({"a": 0.0, "b": 0.0}) == {"a": 0.0, "b": 0.0}
```

### 3.3 `test_palettize_core.py` and `test_packing.py`

```python
# scripts/qwen_palettize/tests/unit/test_packing.py
import numpy as np
import torch
from palettize_core import pack_idx2, pack_indices_transposed_2bit, BITWIDTH, GROUP_SIZE

def test_pack_idx2_roundtrip():
    """Pack and unpack should round-trip."""
    indices = torch.randint(0, 4, (100, 200), dtype=torch.uint8)
    packed = pack_idx2(indices)
    # Unpack: each byte holds 4 indices, 2 bits each, little-endian within byte
    unpacked = np.frombuffer(packed, dtype=np.uint8)
    result = np.zeros(len(unpacked) * 4, dtype=np.uint8)
    for i, b in enumerate(unpacked):
        for j in range(4):
            result[i * 4 + j] = (b >> (2 * j)) & 0x03
    result = result[:indices.numel()].reshape(indices.shape)
    assert np.array_equal(result, indices.numpy())

def test_pack_idx2_all_zeros():
    indices = torch.zeros(10, 10, dtype=torch.uint8)
    packed = pack_idx2(indices)
    assert all(b == 0 for b in packed)

def test_pack_idx2_all_threes():
    indices = torch.full((10, 10), 3, dtype=torch.uint8)
    packed = pack_idx2(indices)
    # Each byte should be 0b11_11_11_11 = 0xFF = 255
    assert all(b == 255 for b in packed)

def test_bitwidth_is_2():
    assert BITWIDTH == 2

def test_group_size_is_256():
    assert GROUP_SIZE == 256
```

These tests run in ~1 second total. They verify the pure-Python functions that have no GPU dependency.

---

## 4. Tier 2: Component tests (single module, mocked deps)

Component tests cover a single module (e.g., `model.py`) with mocked dependencies. They may use a GPU (for `PalettizedLinear`'s CUDA kernel), but the test is structured so the GPU is optional — if CUDA is unavailable, the test falls back to the PyTorch path and skips the CUDA-specific assertions.

### 4.1 `test_model.py`

```python
# scripts/qwen_palettize/tests/component/test_model.py
import torch
import pytest
from qwen_palettize.model import PalettizedLinear, QwenLoRA

@pytest.fixture
def tiny_linear():
    """A 4×8 nn.Linear with random weights."""
    lin = torch.nn.Linear(4, 8, bias=False)
    torch.manual_seed(42)
    torch.nn.init.normal_(lin.weight, std=0.1)
    return lin

@pytest.fixture
def tiny_pal_lin(tiny_linear):
    """A PalettizedLinear wrapping tiny_linear."""
    indices = torch.randint(0, 4, (4, 8), dtype=torch.long)
    n_groups = 2  # 8 / 4 (group_size=4 for test)
    palette = torch.randn(n_groups, 4, dtype=torch.bfloat16)
    pal = PalettizedLinear(
        original_linear=tiny_linear,
        indices=indices, n_groups=n_groups, palette_size=4, group_size=4,
        pre_transposed=False, initial_palette=palette, use_soft_indices=False,
    )
    return pal

def test_palettized_linear_forward_shape(tiny_pal_lin):
    """Forward should produce the right output shape."""
    x = torch.randn(2, 3, 4, dtype=torch.bfloat16)
    y = tiny_pal_lin(x)
    assert y.shape == (2, 3, 8)

def test_palettized_linear_forward_no_bias(tiny_pal_lin):
    """If no bias, output should match x @ gathered."""
    x = torch.randn(2, 3, 4, dtype=torch.bfloat16)
    y = tiny_pal_lin(x)
    # Compute expected
    flat_palette = tiny_pal_lin.palette.reshape(-1)
    gathered = flat_palette[tiny_pal_lin._flat_idx]
    expected = x @ gathered
    assert torch.allclose(y, expected, atol=1e-2)

def test_palettized_linear_extract_hard_indices(tiny_pal_lin):
    """extract_hard_indices should produce argmax of logits."""
    pal = tiny_pal_lin
    # Manually set up index_logits
    pal.use_soft_indices = True
    pal.index_logits = torch.nn.Parameter(
        torch.tensor([[[10, -10, -10, -10, 10, -10, -10, -10,  # plane 0
                         -10, 10, -10, -10, 10, -10, -10, -10,]],
                      [[-10, 10, -10, -10, -10, 10, -10, -10,  # plane 1
                        -10, -10, 10, -10, -10, -10, 10, -10,]],
                      ], dtype=torch.float16)
    )
    # Note: this is a contrived example; the real test would use a proper fixture
    pal.extract_hard_indices()
    assert pal.index_logits is None
    assert pal.use_soft_indices is False

def test_qwen_lora_forward_shape():
    """LoRA forward should add a correction to the base."""
    base = torch.nn.Linear(4, 8, bias=False)
    lora = QwenLoRA(base, rank=2, alpha=4, init="random")
    x = torch.randn(2, 3, 4, dtype=torch.bfloat16)
    y = lora(x)
    assert y.shape == (2, 3, 8)
```

### 4.2 `test_optim.py`

```python
# scripts/qwen_palettize/tests/component/test_optim.py
import torch
import pytest
from qwen_palettize.optim import Muon, FP32MasterAdamW, FP32MasterMuon

def test_muon_step_reduces_loss_on_quadratic():
    """Muon should reduce loss on a simple quadratic."""
    x = torch.randn(10, 5, requires_grad=True)
    opt = Muon([x], lr=0.1, momentum=0.95, ns_steps=5)
    for _ in range(50):
        opt.zero_grad()
        loss = (x ** 2).sum()
        loss.backward()
        opt.step()
    final_loss = (x ** 2).sum().item()
    assert final_loss < 0.1  # should be near zero

def test_fp32_master_adamw_step():
    """FP32MasterAdamW should copy bf16 grads to fp32, step, copy back."""
    p = torch.randn(10, 5, dtype=torch.bfloat16, requires_grad=True)
    master_opt = FP32MasterAdamW([{"params": [p], "lr": 1e-3}], betas=(0.9, 0.95))
    # Simulate a gradient
    p.grad = torch.randn_like(p, dtype=torch.float32)
    initial = p.data.clone()
    master_opt.step()
    # The param should have changed
    assert not torch.allclose(initial, p.data)

def test_fp32_master_muon_step():
    p = torch.randn(10, 5, dtype=torch.bfloat16, requires_grad=True)
    master_opt = FP32MasterMuon([{"params": [p], "lr": 0.1, "momentum": 0.95, "ns_steps": 5}])
    p.grad = torch.randn_like(p, dtype=torch.float32)
    initial = p.data.clone()
    master_opt.step()
    assert not torch.allclose(initial, p.data)
```

### 4.3 `test_checkpoint.py`

```python
# scripts/qwen_palettize/tests/component/test_checkpoint.py
import torch, os, tempfile
import pytest
from qwen_palettize.checkpoint import save_state, load_state
from qwen_palettize.model import PalettizedLinear

@pytest.fixture
def tiny_pal_lin():
    lin = torch.nn.Linear(4, 8, bias=False)
    indices = torch.randint(0, 4, (4, 8), dtype=torch.long)
    pal = PalettizedLinear(lin, indices=indices, n_groups=2, palette_size=4,
                            group_size=4, pre_transposed=False, use_soft_indices=True)
    return pal

def test_save_load_roundtrip(tiny_pal_lin):
    """Save and load should round-trip the parameters."""
    # Wrap in a tiny nn.Module
    class TinyModel(torch.nn.Module):
        def __init__(self, pal):
            super().__init__()
            self.pal = pal
    model = TinyModel(tiny_pal_lin)
    
    with tempfile.TemporaryDirectory() as tmp:
        save_state(model, sb_idx=0, step=100, cos=0.95, loss=0.05, out_dir=tmp)
        assert os.path.exists(os.path.join(tmp, "_resume.json"))
        
        # Load into a fresh model
        model2 = TinyModel(PalettizedLinear(
            torch.nn.Linear(4, 8, bias=False),
            indices=torch.randint(0, 4, (4, 8), dtype=torch.long),
            n_groups=2, palette_size=4, group_size=4, pre_transposed=False,
            use_soft_indices=True,
        ))
        step, cos, loss = load_state(model2, tmp)
        assert step == 100
        assert abs(cos - 0.95) < 1e-6
        assert abs(loss - 0.05) < 1e-6
        # Verify palette round-tripped
        assert torch.allclose(model.pal.palette, model2.pal.palette)
```

### 4.4 `test_log.py`

```python
# scripts/qwen_palettize/tests/component/test_log.py
import os, tempfile, csv
import pytest
from qwen_palettize.log import MetricsLogger, StdoutBackend, CsvBackend
from qwen_palettize.config import Config

def test_stdout_backend_smoke(capsys):
    cfg = Config(log_backend="stdout")
    logger = MetricsLogger(cfg)
    logger.log_step(step=10, loss=0.05, cos=0.95, tps=1.7,
                    tau=0.5, grad_norms={"indices": 0.4, "other": 0.2},
                    lrs={"palettes": 3e-3}, gpu_mem=(2.0, 4.0), gpu_util=80)
    captured = capsys.readouterr()
    assert "step=10" in captured.out
    assert "loss=0.0500" in captured.out

def test_csv_backend_writes_file():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(log_backend="csv", logs_dir=tmp)
        logger = MetricsLogger(cfg)
        logger.log_step(step=10, loss=0.05, cos=0.95, tps=1.7,
                        tau=0.5, grad_norms={"indices": 0.4, "other": 0.2},
                        lrs={"palettes": 3e-3, "lora": 1e-3,
                             "indices": 1e-3, "layernorms": 3e-4},
                        gpu_mem=(2.0, 4.0), gpu_util=80)
        logger.finish()
        # Find the CSV file
        csvs = [f for f in os.listdir(tmp) if f.endswith(".csv")]
        assert len(csvs) > 0
        # Read the file and verify the row
        with open(os.path.join(tmp, csvs[0])) as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) >= 2  # header + 1 step
            assert "10" in rows[1]
```

---

## 5. Tier 3: Integration tests (real GPU, 1 step)

Integration tests cover the full `Trainer` class with a tiny model on a real GPU. They run in ~30 seconds.

### 5.1 `test_train_one_step.py`

```python
# scripts/qwen_palettize/tests/integration/test_train_one_step.py
import os, tempfile
import torch
import pytest
from qwen_palettize.config import Config
from qwen_palettize.train import Trainer
from qwen_palettize.model import PalettizedLinear, PartialWrapper, PartialModel

@pytest.fixture
def tiny_cfg():
    return Config(
        sb_idx=0, max_steps=1, seq_len=16, batch_size=2,
        lora_rank=4, lora_alpha=8,
        use_soft_indices=True, tau_init=1.0, tau_final=0.1, tau_anneal_steps=100,
        eval_every=1, log_every=1, save_every=1,
        # Use a tiny model (not the real Qwen3.5-4B)
        palettized_base="/tmp/test_palettized",
        trained_base="/tmp/test_trained",
        logs_dir="/tmp/test_logs",
        eval_cache_path="/tmp/test_eval.pt",
        hyperparams_file="/tmp/test_hyperparams.json",
    )

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_train_one_step_no_crash(tiny_cfg):
    """One training step should complete without raising."""
    # Build a tiny model manually (skip the real Qwen3.5-4B load)
    # ... (mock build_student_super_block to return a tiny model)
    trainer = Trainer(tiny_cfg)
    trainer._build_student = lambda: None  # mock
    trainer.student = torch.nn.Linear(16, 16).cuda()
    trainer._build_optimizers = lambda: None
    trainer._load_teacher = lambda: None
    trainer._share_embeddings = lambda: None
    trainer._prepare_eval_set = lambda: None
    
    # Manually run one step
    batch = torch.randint(0, 100, (2, 16), device="cuda")
    trainer.global_step = 0
    # ... (mock forward, compute_loss, backward)
    # Just verify the test infrastructure works
    assert trainer.cfg.sb_idx == 0
    assert trainer.global_step == 0
```

The integration test in this example is mocked — the real test would require a tiny "Qwen-like" model that can be loaded without the 1B-parameter `from_pretrained` call. Building that tiny model is part of the Wave 3 work.

### 5.2 `test_checkpoint_resume.py`

```python
# scripts/qwen_palettize/tests/integration/test_checkpoint_resume.py
import torch
import pytest
from qwen_palettize.train import Trainer
from qwen_palettize.checkpoint import save_state, load_state

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_resume_preserves_cos(tiny_cfg, tmp_path):
    """Save at step N with cos X, load, verify cos X is restored."""
    # ... (build a tiny model, run 10 steps, save, load into fresh model,
    #      run 1 more step, verify cos matches the original run's step 11)
    pass
```

This is the **most important** integration test: it verifies that the refactor preserves the cos=0.953 baseline. If a refactor changes the random seed handling, the optimizer state layout, or the parameter order, this test fails.

---

## 6. Tier 4: Regression tests (full training run)

The Tier 4 test is the safety net. It is the only test that takes a meaningful amount of time (~78 minutes), and it is run nightly (not on every commit).

### 6.1 The regression test

```python
# scripts/qwen_palettize/tests/regression/test_full_run.py
import subprocess
import pytest
import json

@pytest.mark.regression
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sb0_cos_baseline():
    """Run 8000 steps of sb0 training, verify cos >= 0.94 (with tolerance)."""
    cmd = [
        "python", "scripts/train_qwen.py",
        "--sb_idx", "0", "--max_steps", "8000",
        "--seq_len", "512", "--batch_size", "32",
        "--lora_rank", "16", "--use_soft_indices", "1",
        "--tau_init", "2.0", "--tau_final", "0.1", "--tau_anneal_steps", "4000",
        "--log_backend", "csv",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    assert result.returncode == 0, f"Training failed:\n{result.stderr}"
    
    # Parse the final eval cos from the CSV
    # ... (read /tmp/.../eval.csv, find the last row, verify cos >= 0.94)
    last_cos = ...
    assert last_cos >= 0.94, f"cos regression: expected >= 0.94, got {last_cos}"
```

### 6.2 The baseline number

The current baseline is `cos=0.9530` (commit `b82a6be`). The regression test should verify `cos >= 0.94` (a 1.3% tolerance, accounting for the inherent nondeterminism of Gumbel-Softmax sampling and CUDA floating-point ordering).

If a refactor drops cos below 0.94, the regression test fails. The team is alerted before the change is merged.

---

## 7. The CI/CD pipeline

### 7.1 GitHub Actions workflow

The CI/CD pipeline is defined in `.github/workflows/test.yml`:

```yaml
name: Tests
on:
  push:
    branches: [main, research-architecture-review]
  pull_request:
    branches: [main]
  schedule:
    - cron: "0 2 * * *"  # Nightly at 2 AM UTC

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: pip install torch numpy pytest
      - run: pytest scripts/qwen_palettize/tests/unit/ -v

  component-tests-cpu:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: pip install torch numpy pytest
      - run: pytest scripts/qwen_palettize/tests/component/ -v -k "not cuda"

  integration-tests-gpu:
    runs-on: [self-hosted, gpu]
    if: github.event_name == 'pull_request'
    steps:
      - uses: actions/checkout@v4
      - run: pip install torch numpy pytest
      - run: pytest scripts/qwen_palettize/tests/integration/ -v

  regression-test:
    runs-on: [self-hosted, gpu]
    if: github.event_name == 'schedule'
    steps:
      - uses: actions/checkout@v4
      - run: pip install torch numpy pytest
      - run: pytest scripts/qwen_palettize/tests/regression/ -v -m regression
```

The workflow has four jobs:

1. `unit-tests` — runs on every push and PR. Tier 1 tests only. < 5 seconds.
2. `component-tests-cpu` — runs on every push and PR. Tier 2 tests that don't require CUDA. < 30 seconds.
3. `integration-tests-gpu` — runs on PRs. Tier 3 tests. Requires a self-hosted GPU runner. < 60 seconds.
4. `regression-test` — runs nightly. Tier 4 test. Requires a self-hosted GPU runner. ~78 minutes.

### 7.2 Self-hosted GPU runner

The `runs-on: [self-hosted, gpu]` label requires a self-hosted runner with a CUDA-capable GPU. The setup is:

1. Install the GitHub Actions runner on the GPU machine.
2. Label the runner with `gpu` (in the runner config).
3. Install `torch`, `numpy`, `pytest` in the runner's Python environment.
4. Configure the runner to execute only one job at a time (to avoid GPU contention).

For a research team without a dedicated GPU runner, an alternative is to use a cloud GPU service like Lambda Labs, RunPod, or AWS p4d instances. The CI workflow can spin up an instance, run the tests, and tear down the instance. The cost is ~$3 per run (for a 1-hour p4d instance), acceptable for a nightly regression check.

### 7.3 Pre-commit hook

For developers to catch issues before pushing, a pre-commit hook runs the Tier 1 tests:

```bash
# .git/hooks/pre-commit (or .pre-commit-config.yaml)
#!/bin/bash
cd /path/to/repo
pytest scripts/qwen_palettize/tests/unit/ -q || exit 1
```

This runs in < 5 seconds and catches typos, schema errors, and pure-function regressions before the developer commits.

---

## 8. Coverage measurement

The pytest-cov plugin measures test coverage:

```bash
pytest scripts/qwen_palettize/tests/ --cov=qwen_palettize --cov-report=term-missing
```

The coverage report identifies untested code paths. The target coverage is:

- `config.py`: 100% (pure dataclass, easy to test).
- `loss.py`: 100% (pure functions).
- `anneal.py`: 90% (the freeze/noise functions have edge cases that are hard to trigger).
- `palettize_core.py`: 95% (the packing functions are well-tested by `test_packing.py`).
- `model.py`: 80% (the `PalettizedLinear.forward` CUDA path is hard to test without a real GPU).
- `optim.py`: 75% (the Muon Newton-Schulz iteration is hard to test on a quadratic).
- `train.py`: 60% (the `Trainer` class is integration-tested, not unit-tested).
- `log.py`: 85% (the four backends are tested, but the W&B/TensorBoard APIs are mocked).

The overall coverage target is 80%, which is reasonable for a research codebase. The current codebase has 0% coverage (no tests at all).

---

## 9. Migration plan

The test scaffold is built incrementally, alongside the refactoring work in `05_training_loop_refactor.md`:

1. **Commit 1**: Add `tests/unit/test_config.py`, `tests/unit/test_loss.py`, `tests/unit/test_packing.py`. These test the pure functions that are extracted in the first refactor commit. Run them via `pytest` in CI.
2. **Commit 2**: Add `tests/component/test_model.py`, `tests/component/test_optim.py`, `tests/component/test_checkpoint.py`, `tests/component/test_log.py`. These test the modules extracted in the second refactor commit.
3. **Commit 3**: Add `tests/integration/test_train_one_step.py`, `tests/integration/test_checkpoint_resume.py`. These test the refactored `Trainer` class.
4. **Commit 4**: Add `tests/regression/test_full_run.py` and the `.github/workflows/test.yml`. Set up the self-hosted GPU runner. Schedule the nightly regression test.

Each commit is independently reviewable. The full test scaffold is ~1,000 LOC, built over the 3-day refactor window.

---

## 10. The cost of not testing

The user message says: "Every change is tested by running 8000 steps." At 580 ms/step, this is ~78 minutes per regression check. For a team iterating on the cos=0.999 target, this means:

- ~5 regression checks per developer per day = ~6.5 hours of GPU time per developer per day, spent on regression checks alone.
- A 1-line typo in `compute_loss` takes 78 minutes to detect, vs. 5 seconds with a unit test.
- A refactor that breaks `save_state` is detected only after a training run completes (78 minutes), tries to save, and fails.
- A CI failure on a PR blocks the merge until the developer runs the tests locally (78 minutes).

With the test scaffold:

- 1-line typos are caught in < 5 seconds by Tier 1 tests.
- Refactor regressions in `save_state` are caught in < 30 seconds by Tier 2 tests.
- Integration regressions are caught in < 60 seconds by Tier 3 tests.
- Baseline regressions are caught nightly by Tier 4.

The test scaffold pays for itself within the first week of use. It is the single highest-ROI infrastructure investment after the `PartialWrapper` → `nn.Module` fix.

---

## 11. Summary

The proposed test scaffold has four tiers, three CI jobs, and one nightly regression check. It catches typos in 5 seconds, refactor regressions in 30 seconds, integration bugs in 60 seconds, and baseline drift in 78 minutes (overnight). The full scaffold is ~1,000 LOC, built alongside the refactor in `05_training_loop_refactor.md` over 3 days. The CI workflow is one YAML file. The self-hosted GPU runner is one machine setup.

The scaffold is the prerequisite for any further architectural work. Without it, every refactor is a 78-minute gamble. With it, every refactor is a 5-second confidence check.
