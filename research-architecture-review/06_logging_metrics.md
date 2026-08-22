# 06 — Logging and Metrics: wandb / Tensorboard / Structured Output

## 1. The current state

Every metric produced by the training pipeline is emitted via a single `print()` statement at `train_qwen.py:1188`:

```python
print(f"  step={global_step:5d} loss={comps['loss']:.4f} cos={cos_val:.4f} tps={tps:.1f} "
      f"{tau_str}{gpu_str} gn=[{gn_str}] [{lrs_str}]", flush=True)
```

This single line is the **only** per-step metric output. It is joined by:

- The `[EVAL]` line at line 1196, emitted every 250 steps: `print(f"  [EVAL] step={global_step} cos={eval_cos:.6f} loss={eval_result['loss']:.6f} ...", flush=True)`.
- The `[step N] NaN loss — skipping` line at line 1113, emitted on NaN detection.
- The `[step N] Hyperparams updated` line at line 1047, emitted on JSON hot-reload.
- The `★ NEW BEST (eval)` line at line 1201, emitted when a new best cos is found.
- The `[freeze]` / `[noise]` lines, emitted by the annealing utilities when they trigger.
- The setup prints (lines 927–936): model build summary, param counts, optimizer summary, teacher load summary.

All of these are tee'd to `logs/train_sbN.log` by the `_DualStream` shim at `train_qwen.py:50–73`. The log file is overwritten on every run (the `open(log_path, "w")` at line 54 uses mode `"w"`, not `"a"`). There is no log rotation, no per-run timestamp in the filename, no JSON-lines output.

### 1.1 The seven problems

1. **No machine-readable format.** Every metric is a free-text string. To extract `cos` vs `step`, you must parse the string with a regex. To extract `grad_norm[indices]`, you must parse the `gn=[...]` substring with another regex. Any format change (e.g., `cos={cos_val:.4f}` → `cos={cos_val:.5f}`) silently breaks the parser.
2. **No time-series database.** The metrics are not stored in a queryable format. To plot `cos` vs `step` for two runs, you must read both log files, parse both, and write a custom plot script. There is no `wandb.log` or `SummaryWriter.add_scalar` call.
3. **No hyperparameter association.** The metrics are not associated with the hyperparameters that produced them. To compare `lr=3e-3` vs `lr=1e-3`, you must run two training runs, parse both logs, and tabulate the results manually. The `sweep_qwen.py` script automates this, but it works by `subprocess.Popen` + regex parsing — brittle.
4. **No structured output for eval.** The `[EVAL]` line includes `cos`, `loss`, and `n_seqs`, but not the per-layer cosines, not the per-group grad norms, not the per-step tau. If you want to know "which layer is worst at step 4000?", you have to add a print statement and re-run.
5. **No artifact tracking.** The `save_state` call at line 1205 writes 140+ `.pt` files to disk, but the log does not record which artifact corresponds to which `best_cos`. The `_resume.json` file inside the checkpoint directory records `{step, cos, loss, sb_idx}`, but this is the only artifact-metadata link. There is no `wandb.Artifact` or `mlflow.log_artifact` integration.
6. **No system metrics.** The log includes `gpu_mem_alloc`, `gpu_mem_reserved`, and `gpu_util` (line 1180–1183), but not CPU usage, not disk I/O, not network I/O, not power draw. These are available via `pynvml` and `psutil`, but the codebase does not use them.
7. **No live dashboard.** The only way to monitor a training run is to `tail -f logs/train_sbN.log`. There is no web UI, no live plot, no alerting on divergence.

### 1.2 What "good" looks like

The HuggingFace `Trainer` (used by 50,000+ models) emits metrics to:

- stdout (always)
- W&B (if `report_to="wandb"`)
- TensorBoard (if `report_to="tensorboard"`)
- CSV (if `report_to="csv"`)
- MLflow (if `report_to="mlflow"`)
- AzureML (if `report_to="azure_ml"`)
- ClearML (if `report_to="clearml"`)
- Any custom backend via the `transformers.integrations.Integration` subclass mechanism

This is the gold standard. Every metric is logged to every active backend in parallel, with one line of code in the training loop. The user can switch backends by changing one CLI flag (`--report_to wandb,tensorboard,csv`).

This document proposes a similar (but lighter) architecture for the qwen-palettize codebase.

---

## 2. The proposed `MetricsLogger` design

The `MetricsLogger` class was introduced in `05_training_loop_refactor.md` §5.1. This section elaborates on the design and provides the full implementation of the four backends.

### 2.1 The metric schema

Every metric the training loop produces is one of four types:

| Type | Example | Backend mapping |
|---|---|---|
| Scalar | `loss=0.0421`, `cos=0.9530`, `tps=1.73` | `wandb.log({key: val})`, `SummaryWriter.add_scalar(key, val, step)`, CSV cell |
| Vector | `grad_norms={indices: 0.42, other: 0.18}` | One scalar per element: `wandb.log({"gn_indices": 0.42, "gn_other": 0.18})` |
| Event | `nan_skips=1`, `config_reload=1`, `new_best=1` | Counter, incremented at each occurrence |
| Artifact | `checkpoint_path=trained/superblock_0_best/` | `wandb.log_artifact(path)`, CSV row, stdout line |

The schema is enforced by the `MetricsLogger`'s method signatures:

```python
class MetricsLogger:
    def log_step(self, *, step, loss, cos, tps, tau=None, grad_norms=None,
                 lrs=None, gpu_mem=None, gpu_util=None): ...
    def log_eval(self, step, result): ...           # result: {cos, loss, n_seqs}
    def log_nan(self, step, loss, comps): ...
    def log_config_reload(self, step): ...
    def log_new_best(self, step, cos, loss): ...
    def log_artifact(self, step, path, type="checkpoint", metadata=None): ...
    def log_setup(self, trainer): ...
    def log_final(self, best_cos, best_step): ...
    def finish(self): ...
```

Each backend implements the same interface. The training loop calls `logger.log_step(...)` once per step; the backend routes the call to its native API.

### 2.2 Why a single interface, multiple backends

The single-interface design has three advantages:

1. **Switching backends is one CLI flag.** `--log_backend wandb` vs `--log_backend tensorboard` vs `--log_backend csv`. The training loop does not change.
2. **Adding a new backend is one class.** Subclass `Backend`, implement the eight methods, register it in `_init_backend`. No changes to the training loop.
3. **Comparing across backends is trivial.** A run logged to `stdout` produces the same data as a run logged to `wandb`. A CSV export can be plotted with `matplotlib` or `seaborn`; a W&B run can be plotted with the W&B UI; both show the same curves.

---

## 3. The four backend implementations

### 3.1 StdoutBackend (current behavior, preserved)

The `StdoutBackend` reproduces the current `print()` format bit-for-bit. This is the default — if no `--log_backend` is specified, the user sees the same output as today. The backward compatibility guarantee is: **`logs/train_sbN.log` from the refactored code is byte-identical to `logs/train_sbN.log` from the current code** (modulo the timestamp, which the current code does not include either).

### 3.2 WandbBackend

The `WandbBackend` initializes a W&B run on construction, logs every metric via `wandb.log`, and finishes the run on `finish()`. It also:

- Logs the full `Config` object via `wandb.config.update(cfg.__dict__)` — every hyperparameter is queryable from the W&B UI.
- Logs the system metrics (GPU mem, GPU util, power, temp) automatically — W&B's `wandb.init` enables system stats collection by default.
- Logs artifacts via `wandb.log_artifact(path, type="checkpoint", metadata={"step": ..., "cos": ...})` — the artifact is stored in the W&B artifact store, versioned, and downloadable by other researchers.
- Logs the model architecture via `wandb.watch(model, log="gradients", log_freq=250)` — gradient histograms are available in the W&B UI.

The W&B integration requires `pip install wandb` and `wandb login` (one-time). The `WANDB_API_KEY` environment variable can also be set for non-interactive login.

### 3.3 TensorboardBackend

The `TensorboardBackend` uses `torch.utils.tensorboard.SummaryWriter`, which writes protobuf event files to a `logs/` directory. The files are readable by:

- `tensorboard --logdir logs/` (the standard TensorBoard UI, runs on `localhost:6006`).
- `tensorboard.backend.application.TENSORBOARD_API` (programmatic access).
- `mlflow.log_artifact` (for MLflow integration).

The TensorBoard backend is the **most portable** — it requires no external service, no API key, and works on any machine with PyTorch installed. It is the recommended default for researchers who do not want to use W&B.

### 3.4 CsvBackend

The `CsvBackend` writes a single `metrics.csv` file with one row per training step. The columns are:

```
step,timestamp,loss,cos,tps,tau,grad_norm_indices,grad_norm_other,
lr_palettes,lr_lora,lr_indices,lr_layernorms,
gpu_mem_alloc,gpu_mem_reserved,gpu_util,nan_skips
```

The CSV is the **most scriptable** — it can be loaded into `pandas` for analysis, plotted with `matplotlib`, or imported into Google Sheets for sharing. It is the recommended default for automated sweeps, where the `sweep_qwen.py` script can read the CSV directly instead of parsing `print()` output.

The CSV backend also writes a second file, `eval.csv`, with one row per eval:

```
step,timestamp,eval_cos,eval_loss,n_seqs,is_best
```

This separation makes it easy to plot the eval cos vs step without filtering out the per-step rows.

---

## 4. Per-step metric inventory

This section enumerates every metric the training loop should emit, with the type, source, and backend mapping.

### 4.1 Per-step metrics (every `log_every` steps, default 50)

| Metric | Type | Source | Backend key |
|---|---|---|---|
| `step` | int | `trainer.global_step` | (always logged as the step argument) |
| `loss` | float | `comps["loss"]` | `loss` |
| `cos` | float | `1.0 - comps["cos"]` | `cos` |
| `tps` | float | `(global_step - resume_step) / elapsed` | `tps` |
| `tau` | float | `trainer._current_tau` | `tau` |
| `grad_norm_indices` | float | `grad_norms["indices"]` | `grad_norm/indices` |
| `grad_norm_other` | float | `grad_norms["other"]` | `grad_norm/other` |
| `lr_palettes` | float | `group_lrs["palettes"]` | `lr/palettes` |
| `lr_lora` | float | `group_lrs["lora"]` | `lr/lora` |
| `lr_indices` | float | `group_lrs["indices"]` | `lr/indices` |
| `lr_layernorms` | float | `group_lrs["layernorms"]` | `lr/layernorms` |
| `gpu_mem_alloc` | float | `torch.cuda.memory_allocated() / (1024**3)` | `gpu/mem_alloc_gb` |
| `gpu_mem_reserved` | float | `torch.cuda.memory_reserved() / (1024**3)` | `gpu/mem_reserved_gb` |
| `gpu_util` | int | `torch.cuda.utilization()` | `gpu/util_pct` |

### 4.2 Per-eval metrics (every `eval_every` steps, default 250)

| Metric | Type | Source | Backend key |
|---|---|---|---|
| `step` | int | `trainer.global_step` | (step) |
| `eval_cos` | float | `eval_result["cos"]` | `eval/cos` |
| `eval_loss` | float | `eval_result["loss"]` | `eval/loss` |
| `eval_n_seqs` | int | `eval_result["n_seqs"]` | `eval/n_seqs` |
| `is_best` | bool | `eval_cos > best_cos` | `eval/is_best` |

### 4.3 Event metrics (on occurrence)

| Event | Type | Trigger |
|---|---|---|
| `nan_skip` | counter | NaN loss detected (line 1105) |
| `config_reload` | counter | JSON hot-reload detected (line 1043) |
| `palette_freeze` | counter | `freeze_settled_palettes` triggers (line 246) |
| `noise_inject` | counter | `inject_gradient_noise` triggers (line 315) |
| `loss_type_change` | counter | `get_loss_type_for_step` flips (line 342) |
| `new_best` | event | `eval_cos > best_cos` (line 1197) |
| `save` | artifact | `save_state` called (line 1205) |

### 4.4 Setup metrics (once at start)

| Metric | Type | Source |
|---|---|---|
| `sb_idx` | int | `cfg.sb_idx` |
| `max_steps` | int | `cfg.max_steps` |
| `lora_rank` | int | `cfg.lora_rank` |
| `seq_len` | int | `cfg.seq_len` |
| `batch_size` | int | `cfg.batch_size` |
| `n_params_total` | int | `sum(p.numel() for p in student.parameters())` |
| `n_params_trainable` | int | `sum(p.numel() for p in student.parameters() if p.requires_grad)` |
| `n_params_frozen` | int | `sum(p.numel() for p in student.parameters() if not p.requires_grad)` |
| `param_groups` | dict | `{g: c for g, c in counts.items()}` |
| `config_signature` | str | `cfg.signature()` |
| `git_commit` | str | `subprocess.check_output(["git", "rev-parse", "HEAD"])` |
| `cuda_device_name` | str | `torch.cuda.get_device_name(0)` |

---

## 5. The `wandb` integration in detail

W&B is the highest-value integration because it provides:

- A web UI for visualizing runs.
- A run-comparison feature for sweeps.
- An artifact store for checkpoints.
- A model registry for deployed models.
- A reports feature for sharing findings with collaborators.

### 5.1 Initialization

```python
class WandbBackend:
    def __init__(self, project, run_name, config=None, tags=None):
        import wandb
        self.wandb = wandb
        self.run = wandb.init(
            project=project,
            name=run_name,
            config=config or {},
            tags=tags or [],
            dir="/tmp/wandb",
        )
```

The `project` is the W&B project name (e.g., `"qwen-palettize"`). The `run_name` is a human-readable name (e.g., `"sb0_lr3e-3_bs32"`). The `config` is the full `Config` dataclass, serialized via `asdict(cfg)`. The `tags` are list of strings for filtering (e.g., `["sweep-001", "lr-3e-3"]`).

### 5.2 Step logging

```python
def log_step(self, *, step, loss, cos, tps, tau=None, grad_norms=None,
             lrs=None, gpu_mem=None, gpu_util=None):
    metrics = {
        "train/loss": loss,
        "train/cos": cos,
        "train/tps": tps,
    }
    if tau is not None: metrics["train/tau"] = tau
    if grad_norms:
        for k, v in grad_norms.items():
            metrics[f"grad_norm/{k}"] = v
    if lrs:
        for k, v in lrs.items():
            metrics[f"lr/{k}"] = v
    if gpu_mem is not None:
        metrics["gpu/mem_alloc_gb"] = gpu_mem[0]
        metrics["gpu/mem_reserved_gb"] = gpu_mem[1]
    if gpu_util is not None:
        metrics["gpu/util_pct"] = gpu_util
    self.run.log(metrics, step=step)
```

The `train/`, `grad_norm/`, `lr/`, `gpu/` prefixes group metrics in the W&B UI. Each prefix becomes a section in the charts panel.

### 5.3 Eval logging

```python
def log_eval(self, step, result):
    self.run.log({
        "eval/cos": result["cos"],
        "eval/loss": result["loss"],
        "eval/n_seqs": result["n_seqs"],
    }, step=step)
```

### 5.4 Artifact logging

```python
def log_artifact(self, step, path, type="checkpoint", metadata=None):
    artifact = self.wandb.Artifact(
        name=f"superblock_{self.run.id}_{step}",
        type=type,
        metadata=metadata or {},
    )
    artifact.add_dir(path)
    self.run.log_artifact(artifact)
```

This uploads the entire checkpoint directory (140+ `.pt` files) to W&B's artifact store. The artifact is versioned and can be downloaded by other researchers with `wandb.artifact(name).download()`.

For large checkpoints (e.g., the 232 MB `superblock_0_best/`), the upload takes ~30 seconds on a typical research network. The training loop should call `log_artifact` in a background thread to avoid blocking.

### 5.5 Final summary

```python
def log_final(self, best_cos, best_step):
    self.run.summary["best_cos"] = best_cos
    self.run.summary["best_step"] = best_step
    self.run.summary["final_step"] = self.global_step

def finish(self):
    self.run.finish()
```

The `summary` is displayed in the W&B runs table — it is the canonical "headline number" for the run.

---

## 6. The TensorBoard integration in detail

TensorBoard is the **default fallback** — no account, no API key, no network. It writes protobuf event files to a local `logs/` directory.

### 6.1 Initialization

```python
class TensorboardBackend:
    def __init__(self, logs_dir, run_name=None):
        from torch.utils.tensorboard import SummaryWriter
        import datetime
        subdir = run_name or datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(logs_dir, subdir)
        self.writer = SummaryWriter(self.log_dir)
```

The `subdir` ensures each run gets its own directory. The `tensorboard --logdir logs/` command will display all runs in the same UI, with run-comparison features.

### 6.2 Step logging

```python
def log_step(self, *, step, loss, cos, tps, tau=None, grad_norms=None,
             lrs=None, gpu_mem=None, gpu_util=None):
    self.writer.add_scalar("train/loss", loss, step)
    self.writer.add_scalar("train/cos", cos, step)
    self.writer.add_scalar("train/tps", tps, step)
    if tau is not None:
        self.writer.add_scalar("train/tau", tau, step)
    if grad_norms:
        for k, v in grad_norms.items():
            self.writer.add_scalar(f"grad_norm/{k}", v, step)
    if lrs:
        for k, v in lrs.items():
            self.writer.add_scalar(f"lr/{k}", v, step)
    if gpu_mem is not None:
        self.writer.add_scalar("gpu/mem_alloc_gb", gpu_mem[0], step)
        self.writer.add_scalar("gpu/mem_reserved_gb", gpu_mem[1], step)
    if gpu_util is not None:
        self.writer.add_scalar("gpu/util_pct", gpu_util, step)
    self.writer.flush()
```

### 6.3 Limitations

TensorBoard does not have:

- An artifact store. Checkpoints must be saved separately (the existing `save_state` mechanism is fine).
- A run-comparison feature across machines (each machine's `logs/` is independent).
- A web UI accessible from outside the machine (port forwarding is required).

For a single-machine research workflow, TensorBoard is sufficient. For multi-machine sweeps or collaboration, W&B is recommended.

---

## 7. The CSV integration in detail

CSV is the **most scriptable** — it produces a file that any tool (pandas, Excel, Google Sheets, R) can read. It is the recommended backend for automated sweeps.

### 7.1 Initialization

```python
class CsvBackend:
    def __init__(self, logs_dir, run_name=None):
        import datetime
        subdir = run_name or datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
        os.makedirs(os.path.join(logs_dir, subdir), exist_ok=True)
        self.dir = os.path.join(logs_dir, subdir)
        self.train_f = open(os.path.join(self.dir, "train.csv"), "w", buffering=1)
        self.eval_f = open(os.path.join(self.dir, "eval.csv"), "w", buffering=1)
        self.events_f = open(os.path.join(self.dir, "events.csv"), "w", buffering=1)
        # Write headers
        self.train_f.write("step,timestamp,loss,cos,tps,tau,grad_norm_indices,grad_norm_other,lr_palettes,lr_lora,lr_indices,lr_layernorms,gpu_mem_alloc,gpu_mem_reserved,gpu_util\n")
        self.eval_f.write("step,timestamp,eval_cos,eval_loss,n_seqs,is_best\n")
        self.events_f.write("step,timestamp,event_type,metadata\n")
```

### 7.2 Step logging

```python
def log_step(self, *, step, loss, cos, tps, tau=None, grad_norms=None,
             lrs=None, gpu_mem=None, gpu_util=None):
    import datetime
    ts = datetime.datetime.now().isoformat()
    gn = grad_norms or {}
    lr = lrs or {}
    self.train_f.write(
        f"{step},{ts},{loss:.6f},{cos:.6f},{tps:.3f},"
        f"{tau if tau is not None else ''},"
        f"{gn.get('indices', '')},{gn.get('other', '')},"
        f"{lr.get('palettes', '')},{lr.get('lora', '')},{lr.get('indices', '')},{lr.get('layernorms', '')},"
        f"{gpu_mem[0] if gpu_mem else ''},{gpu_mem[1] if gpu_mem else ''},{gpu_util if gpu_util is not None else ''}\n"
    )
```

### 7.3 Analysis script

```python
# scripts/plot_csv.py
import pandas as pd
import matplotlib.pyplot as plt
import sys

df = pd.read_csv(sys.argv[1])
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
df.plot(x="step", y="loss", ax=axes[0, 0], title="Loss")
df.plot(x="step", y="cos", ax=axes[0, 1], title="Cosine")
df.plot(x="step", y=["grad_norm_indices", "grad_norm_other"], ax=axes[1, 0], title="Grad Norms")
df.plot(x="step", y=["lr_palettes", "lr_lora", "lr_indices", "lr_layernorms"], ax=axes[1, 1], title="LRs")
plt.tight_layout()
plt.savefig("metrics.png", dpi=150)
```

This is a 12-line script that replaces the 213-line `sweep_qwen.py` regex parser.

---

## 8. The `sweep_qwen.py` rewrite

The current `sweep_qwen.py` (213 LOC) works by:

1. Defining a list of hyperparameter combinations.
2. For each combination, calling `subprocess.Popen(["python", "train_qwen.py", ...])`.
3. Capturing stdout.
4. Parsing the `[EVAL]` lines with a regex.
5. Tabulating the results.

This is fragile and slow. The refactored version uses the `CsvBackend` directly:

```python
# scripts/sweep_qwen.py (refactored)
import subprocess, itertools, os
from qwen_palettize.config import Config

def sweep(sweep_config):
    """Run a hyperparameter sweep, logging to CSV."""
    for combo in sweep_config.combinations:
        cfg = Config.from_dict({**sweep_config.base, **combo})
        # Each run writes to its own CSV directory
        run_dir = f"runs/sweep_{sweep_config.name}_{cfg.signature()}"
        os.makedirs(run_dir, exist_ok=True)
        # Subprocess call
        cmd = ["python", "scripts/train_qwen.py",
               "--sb_idx", str(cfg.sb_idx),
               "--max_steps", str(cfg.max_steps),
               "--seq_len", str(cfg.seq_len),
               "--batch_size", str(cfg.batch_size),
               "--lora_rank", str(cfg.lora_rank),
               "--use_soft_indices", str(int(cfg.use_soft_indices)),
               "--tau_init", str(cfg.tau_init),
               "--tau_final", str(cfg.tau_final),
               "--tau_anneal_steps", str(cfg.tau_anneal_steps),
               "--log_backend", "csv",
               "--data_cache_dir", sweep_config.cache_dir,
               "--wandb_project", sweep_config.wandb_project,
               "--wandb_run_name", f"{sweep_config.name}_{cfg.signature()}",
               ]
        # ... (write cfg to JSON for reproducibility)
        subprocess.run(cmd, check=True)
    # Aggregate CSVs into one summary
    aggregate_csvs(sweep_config.name, sweep_config.out_dir)
```

The `aggregate_csvs` function reads each run's `train.csv` and `eval.csv`, joins them by step, and writes a single `summary.csv` with one row per run. This is a 20-line pandas script, replacing the 213-line regex parser.

---

## 9. Operational considerations

### 9.1 Disk space

- W&B: artifact uploads are stored in the W&B cloud (free tier: 100 GB).
- TensorBoard: event files are ~1 KB per step, ~50 KB per eval. A 10,000-step run produces ~10 MB of TensorBoard logs.
- CSV: ~150 bytes per row. A 10,000-step run produces ~1.5 MB of CSV.

All are negligible compared to the 232 MB per checkpoint.

### 9.2 Network

- W&B: each `wandb.log` call sends ~1 KB to the W&B cloud. A 10,000-step run with `log_every=50` produces 200 calls = ~200 KB of network traffic. The artifact upload at save time (every 2000 steps) uploads ~232 MB per checkpoint — this is the dominant network cost. Use `wandb init --resume=allow` to recover from interrupted uploads.
- TensorBoard and CSV: no network.

### 9.3 Latency

- W&B: each `wandb.log` call takes ~10 ms (network round-trip). This is negligible compared to the 580 ms per step.
- TensorBoard: each `SummaryWriter.add_scalar` call takes ~0.1 ms (in-process). No network.
- CSV: each row write takes ~0.1 ms (line-buffered). No network.

### 9.4 Failure modes

- W&B: if the network is down, `wandb.log` queues metrics in memory and retries on a background thread. If the queue overflows, metrics are dropped (with a warning). The training loop is not blocked.
- TensorBoard: if the disk is full, `SummaryWriter.add_scalar` raises `OSError`. The training loop catches this and continues with stdout-only logging.
- CSV: if the disk is full, the file write raises `OSError`. Same handling.

The `MetricsLogger` wraps each backend's `log_step` in a `try/except` to ensure no logging failure can crash the training loop.

---

## 10. Summary

The current logging is a single `print()` statement. The proposed `MetricsLogger` is a multi-backend class that emits the same metrics to stdout, W&B, TensorBoard, and CSV in parallel, with one CLI flag to switch backends. The implementation is ~400 LOC (vs the current ~100 LOC of scattered `print()` statements), but it unlocks:

- Live dashboards (W&B UI, TensorBoard UI).
- Run comparison across sweeps (W&B's "group by" feature, pandas on CSV).
- Artifact tracking (W&B artifacts, manual file paths on TensorBoard/CSV).
- System metrics (W&B's automatic system stats, manual `psutil` for others).
- Reproducibility (the `Config` object is logged alongside metrics).
- Scriptability (pandas on CSV replaces the 213-line regex parser).

The `MetricsLogger` is part of the `qwen_palettize/log.py` module proposed in `05_training_loop_refactor.md`. It is the third of the three Wave 3 deliverables, alongside the training loop refactor (`05`) and the testing/CI scaffold (`07`).
