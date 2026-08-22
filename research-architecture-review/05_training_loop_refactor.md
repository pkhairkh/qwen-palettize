# 05 — Training Loop Refactor: Monolithic → Modular Proposal

## 1. The problem with `train_super_block`

`scripts/train_qwen.py` lines 922–1224 define a single Python function called `train_super_block` that spans 302 lines of body and contains, in execution order:

1. Log rewiring (call to `_rewire_log(sb_idx)`)
2. Hyperparameter loading (`write_default_hyperparams`, `load_hyperparams`)
3. Model building (`build_student_super_block`)
4. Parameter group classification (`apply_groups`)
5. Resume checkpoint loading (`load_state`)
6. Optimizer construction (`build_optimizers`)
7. LR scheduler construction (3× `LambdaLR`)
8. LR scheduler advancement (if resuming)
9. Teacher loading (`load_qwen_super_block_only`)
10. Teacher freezing (3 manual loops over `embed_tokens`, `layers`, `norm`)
11. Embed_tokens sharing (`student.model.embed_tokens = teacher.model.embed_tokens`)
12. Student train-mode switch
13. Best-tracker initialization (`best_cos`, `best_step`, `best_loss`)
14. Eval set preparation (`prepare_eval_set`)
15. Data stream initialization (`stream_training_data`)
16. Training loop, including:
    - Temperature annealing (loop over `named_modules` to set `mod.tau`)
    - JSON hot-reload (every 10 steps)
    - Teacher forward (on `stream_t`, with manual layer iteration)
    - Student forward (on default stream, with manual layer iteration)
    - Loss computation (`compute_loss`)
    - NaN detection and skip
    - Backward (`loss.backward()`)
    - Two-tier gradient clipping (indices separate from others)
    - Three optimizer steps (Muon, AdamW, indices)
    - Logit clamping (`±20`)
    - Three scheduler steps
    - Three `zero_grad` calls
    - Per-50-step print logging (with GPU mem/util, tau, grad norms, LRs)
    - Per-250-step eval (call to `evaluate`)
    - Per-2000-step save (call to `save_state`)
17. Final save (with conditional skip)
18. Shutdown hook (optional `os.system("shutdown -h now")`)

This is a single function that does everything. The function has no unit tests (because instantiating it requires a GPU, a 1B-parameter teacher, and a 10-BT dataset). The function has no A/B testability (because two versions of the loop cannot run in the same process). The function has no observability (because the `print` statements are interleaved with the loop body and cannot be extracted). The function has no composability (because no other script can import it without triggering a GPU initialization).

This document proposes a refactor that splits `train_super_block` into eight modules, each with a clear interface, each unit-testable in isolation. The refactor preserves the existing behavior bit-for-bit (the cos=0.953 baseline must reproduce), but unlocks every architectural improvement documented in `02`, `03`, `04`, `06`, `07`.

---

## 2. The proposed file tree

The refactored codebase lives under `scripts/` (same directory, no breaking moves) but is reorganized into a package structure:

```
scripts/
├── qwen_palettize/                    ← new package
│   ├── __init__.py
│   ├── config.py                      ← Config dataclass (single source of truth)
│   ├── model.py                       ← PalettizedLinear, QwenLoRA, PartialModel, PartialWrapper
│   ├── optim.py                       ← Muon, FP32MasterOptimizer, FP32MasterAdamW, FP32MasterMuon
│   ├── loss.py                        ← compute_loss, normalize_weights
│   ├── data.py                        ← stream_training_data, CachedFineWebEduDataset, AsyncPrefetchLoader
│   ├── checkpoint.py                  ← save_state, load_state, migrate_legacy_checkpoint
│   ├── evaluate.py                    ← prepare_eval_set, evaluate
│   ├── anneal.py                      ← tau annealing, gradient noise, palette freezing, cyclic loss
│   ├── log.py                         ← MetricsLogger (wandb, tensorboard, csv, stdout)
│   └── train.py                       ← Trainer class (modular training loop)
├── train_qwen.py                      ← thin CLI entry point (50 LOC)
├── calib_qwen.py                      ← (unchanged for now, but uses qwen_palettize.config)
├── convert_trained_to_packed.py       ← (unchanged)
├── fused_lut_linear_cuda.py           ← (unchanged)
├── fused_lut_kernel.cu                ← (unchanged)
├── palettize_core.py                  ← (unchanged)
└── ... (other scripts unchanged)
```

The new `qwen_palettize/` package is ~1,200 LOC total (vs the current `train_qwen.py` at 1,266 LOC). The total line count is similar, but the **modularity** is radically different: each module has a single responsibility, a clear interface, and can be unit-tested in isolation.

---

## 3. The `Config` dataclass (single source of truth)

The first module to extract is `config.py`. It defines a single `Config` dataclass that replaces the three sources of truth (CLI args, `DEFAULT_HYPERPARAMS`, `/tmp/hyperparams_qwen.json`).

### 3.1 The `Config` dataclass

```python
# scripts/qwen_palettize/config.py
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict
import json
import os

@dataclass
class GroupsConfig:
    palettes: bool = True
    lora: bool = True
    indices: bool = True
    layernorms: bool = True

@dataclass
class LrsConfig:
    palettes: float = 3e-3
    lora: float = 1e-3
    indices: float = 1e-3
    layernorms: float = 3e-4

@dataclass
class LossWeightsConfig:
    cos: float = 0.0
    mse: float = 1.0

@dataclass
class Config:
    # CLI-derived (immutable after parse)
    sb_idx: int = 0
    max_steps: int = 5000
    lora_rank: int = 16
    lora_alpha: int = 32
    seq_len: int = 1024
    batch_size: int = 32
    resume_from: Optional[str] = None
    use_soft_indices: bool = True
    tau_init: float = 2.0
    tau_final: float = 0.1
    tau_anneal_steps: int = 4000
    shutdown_on_done: bool = False
    
    # Hyperparameters (live-tunable via JSON hot-reload)
    groups: GroupsConfig = field(default_factory=GroupsConfig)
    lrs: LrsConfig = field(default_factory=LrsConfig)
    loss_type: str = "norm_mse"
    loss_weights: LossWeightsConfig = field(default_factory=LossWeightsConfig)
    gradient_clip: float = 0.3
    eval_every: int = 250
    log_every: int = 50
    save_every: int = 2000
    
    # Paths (derived)
    palettized_base: str = "/root/qwen35_palettize/palettized"
    trained_base: str = "/root/qwen35_palettize/trained"
    logs_dir: str = "/root/qwen35_palettize/logs"
    hyperparams_file: str = "/tmp/hyperparams_qwen.json"
    eval_cache_path: str = "/root/qwen35_palettize/eval_tokens.pt"
    
    # New architecture knobs
    use_gradient_checkpointing: bool = False  # ← new (default off for backward compat)
    use_torch_compile: bool = False            # ← new
    use_async_data_loader: bool = False        # ← new
    data_cache_dir: Optional[str] = None       # ← new
    log_backend: str = "stdout"                # ← new: "stdout" | "wandb" | "tensorboard" | "csv"
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    
    @classmethod
    def from_cli(cls, args) -> "Config":
        """Construct from argparse.Namespace."""
        cfg = cls(
            sb_idx=args.sb_idx,
            max_steps=args.max_steps,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            resume_from=args.resume_from,
            use_soft_indices=bool(args.use_soft_indices),
            tau_init=args.tau_init,
            tau_final=args.tau_final,
            tau_anneal_steps=args.tau_anneal_steps,
            shutdown_on_done=bool(args.shutdown_on_done),
            use_gradient_checkpointing=bool(getattr(args, "use_gradient_checkpointing", 0)),
            use_torch_compile=bool(getattr(args, "use_torch_compile", 0)),
            use_async_data_loader=bool(getattr(args, "use_async_data_loader", 0)),
            data_cache_dir=getattr(args, "data_cache_dir", None),
            log_backend=getattr(args, "log_backend", "stdout"),
            wandb_project=getattr(args, "wandb_project", None),
            wandb_run_name=getattr(args, "wandb_run_name", None),
        )
        # Merge from JSON if it exists (for resume)
        if os.path.exists(cfg.hyperparams_file):
            cfg.merge_from_json(cfg.hyperparams_file)
        return cfg
    
    def merge_from_json(self, path: str):
        """Live-update from a JSON file (the hot-reload mechanism)."""
        with open(path) as f:
            data = json.load(f)
        if "groups" in data:
            self.groups = GroupsConfig(**data["groups"])
        if "lrs" in data:
            self.lrs = LrsConfig(**data["lrs"])
        if "loss_type" in data:
            self.loss_type = data["loss_type"]
        if "loss_weights" in data:
            self.loss_weights = LossWeightsConfig(**data["loss_weights"])
        if "gradient_clip" in data:
            self.gradient_clip = data["gradient_clip"]
        if "eval_every" in data:
            self.eval_every = data["eval_every"]
        if "log_every" in data:
            self.log_every = data["log_every"]
        if "save_every" in data:
            self.save_every = data["save_every"]
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)
    
    def signature(self) -> str:
        """Stable hash for caching/sweep identification."""
        import hashlib
        return hashlib.md5(self.to_json().encode()).hexdigest()[:8]
```

### 3.2 Why this is better

The current three sources of truth interact as follows (reproduced from `01_architecture_audit.md` §5):

- `DEFAULT_HYPERPARAMS` dict at `train_qwen.py:75` — Python literal, requires code edit to change.
- `argparse` CLI args at `train_qwen.py:1227` — only top-level knobs.
- `/tmp/hyperparams_qwen.json` — re-read every 10 steps, can change LRs / loss_type / freeze groups live.

The interaction is non-obvious: the JSON file overrides `DEFAULT_HYPERPARAMS`, but only for the keys it contains. A typo like `"palattes": 3e-4` in the JSON silently creates a new key that the loader never reads. There is no schema, no validation, no typed `Config` object.

The `Config` dataclass fixes all three:

- **Schema is the dataclass itself.** A typo (`palattes`) creates a `TypeError` at `merge_from_json` time, because `LrsConfig(**data["lrs"])` rejects unknown kwargs.
- **Single source of truth.** All hyperparameters live in the `Config` object. The training loop, the optimizer builder, the loss function, and the evaluator all read from the same `cfg`.
- **Live hot-reload preserved.** `merge_from_json` can be called every 10 steps, same as today.
- **Type-checked.** Pyright/mypy can verify the config object's fields, catching typos at static-analysis time.

### 3.3 The CLI shim

`scripts/train_qwen.py` shrinks from 1,266 LOC to ~50 LOC:

```python
# scripts/train_qwen.py (refactored)
#!/usr/bin/env python3
"""train_qwen.py — Per-super-block distillation training for Qwen3.5-4B.

Thin CLI entry point. All logic in qwen_palettize.train.Trainer.
"""
import argparse
from qwen_palettize.config import Config
from qwen_palettize.train import Trainer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sb_idx", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=5000)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--resume_from", type=str, default=None)
    ap.add_argument("--use_soft_indices", type=int, default=1)
    ap.add_argument("--tau_init", type=float, default=2.0)
    ap.add_argument("--tau_final", type=float, default=0.1)
    ap.add_argument("--tau_anneal_steps", type=int, default=4000)
    ap.add_argument("--shutdown_on_done", type=int, default=0)
    # New architecture flags
    ap.add_argument("--use_gradient_checkpointing", type=int, default=0)
    ap.add_argument("--use_torch_compile", type=int, default=0)
    ap.add_argument("--use_async_data_loader", type=int, default=0)
    ap.add_argument("--data_cache_dir", type=str, default=None)
    ap.add_argument("--log_backend", type=str, default="stdout",
                    choices=["stdout", "wandb", "tensorboard", "csv"])
    ap.add_argument("--wandb_project", type=str, default=None)
    ap.add_argument("--wandb_run_name", type=str, default=None)
    args = ap.parse_args()
    
    cfg = Config.from_cli(args)
    trainer = Trainer(cfg)
    trainer.run()

if __name__ == "__main__":
    main()
```

---

## 4. The `Trainer` class (modular training loop)

The second module to extract is `train.py`. It defines a single `Trainer` class that holds the model, optimizer, data loader, and logger as instance attributes, with methods for each phase of the training loop.

### 4.1 Class interface

```python
# scripts/qwen_palettize/train.py
from dataclasses import asdict
import time, math, os, json
import torch
from .config import Config
from .model import build_student_super_block, load_qwen_super_block_only, PalettizedLinear
from .optim import build_optimizers, FP32MasterAdamW, FP32MasterMuon
from .loss import compute_loss
from .data import stream_training_data
from .checkpoint import save_state, load_state
from .evaluate import prepare_eval_set, evaluate
from .anneal import update_tau, freeze_settled_palettes, snapshot_palette_indices
from .log import MetricsLogger

class Trainer:
    """Modular training loop for one super-block."""
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = MetricsLogger(cfg)
        self.student = None
        self.teacher = None
        self.tokenizer = None
        self.opt_muon = None
        self.opt_adamw = None
        self.opt_indices = None
        self.sched_muon = None
        self.sched_adamw = None
        self.sched_indices = None
        self.best_cos = -1.0
        self.best_step = 0
        self.best_loss = float("inf")
        self.global_step = 0
        self.start_time = None
        self.n_nan_skip = 0
        self.last_hp_check = 0
        self.last_hp_sig = ""
    
    def setup(self):
        """Build student, optimizers, schedulers, teacher, eval set."""
        self._build_student()
        self._build_optimizers()
        self._build_schedulers()
        self._load_teacher()
        self._share_embeddings()
        self._prepare_eval_set()
        self.logger.log_setup(self)
    
    def run(self):
        """Top-level entry point: setup → train → finalize."""
        self.setup()
        self._train_loop()
        self._finalize()
    
    def _build_student(self):
        self.student, self.tokenizer = build_student_super_block(
            self.cfg.sb_idx,
            lora_rank=self.cfg.lora_rank,
            lora_alpha=self.cfg.lora_alpha,
            use_soft_indices=self.cfg.use_soft_indices,
        )
        if self.student is None:
            raise RuntimeError(f"Failed to build student for sb_idx={self.cfg.sb_idx}")
        if self.cfg.resume_from:
            resume_step, resume_cos, resume_loss = load_state(self.student, self.cfg.resume_from)
            self.global_step = resume_step
            self.best_cos = resume_cos
            self.best_step = resume_step
            self.best_loss = resume_loss
        # Optional torch.compile
        if self.cfg.use_torch_compile:
            self.student = torch.compile(self.student, mode="reduce-overhead")
    
    def _build_optimizers(self):
        self.opt_muon, self.opt_adamw, self.opt_indices = build_optimizers(
            self.student, self.cfg, self.cfg.sb_idx
        )
    
    def _build_schedulers(self):
        WARMUP_STEPS = 100
        def lr_lambda(step):
            if step < WARMUP_STEPS:
                return float(step) / float(WARMUP_STEPS)
            return 0.5 * (1.0 + math.cos(math.pi * (step - WARMUP_STEPS) /
                                          max(self.cfg.max_steps - WARMUP_STEPS, 1)))
        self.sched_muon = torch.optim.lr_scheduler.LambdaLR(self.opt_muon.opt, lr_lambda) if self.opt_muon else None
        self.sched_adamw = torch.optim.lr_scheduler.LambdaLR(self.opt_adamw.opt, lr_lambda) if self.opt_adamw else None
        self.sched_indices = torch.optim.lr_scheduler.LambdaLR(self.opt_indices.opt, lr_lambda) if self.opt_indices else None
        if self.global_step > 0:
            if self.sched_muon: self.sched_muon.step(epoch=self.global_step)
            if self.sched_adamw: self.sched_adamw.step(epoch=self.global_step)
            if self.sched_indices: self.sched_indices.step(epoch=self.global_step)
    
    def _load_teacher(self):
        """Load teacher (sharing embed_tokens + frozen prefix with student)."""
        self.teacher, _ = load_qwen_super_block_only(self.cfg.sb_idx, device="cuda",
                                                     dtype=torch.bfloat16)
        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad_(False)
    
    def _share_embeddings(self):
        """Share embed_tokens between teacher and student. Frees the student's copy."""
        # Free student's embed_tokens first
        del self.student.model.embed_tokens
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        # Share teacher's
        self.student.model.embed_tokens = self.teacher.model.embed_tokens
    
    def _prepare_eval_set(self):
        self.eval_tokens = prepare_eval_set(self.tokenizer, device="cuda",
                                           cache_path=self.cfg.eval_cache_path,
                                           seq_len=self.cfg.seq_len)
    
    def _train_loop(self):
        """The actual training loop. Each step is one call to self._train_step()."""
        self.start_time = time.time()
        self.last_hp_sig = json.dumps(asdict(self.cfg), sort_keys=True, default=str)
        data_stream = stream_training_data(
            self.tokenizer, n_seqs=10**12, seq_len=self.cfg.seq_len,
            device="cuda", batch_size=self.cfg.batch_size,
            cache_dir=self.cfg.data_cache_dir,
        )
        for batch_ids in data_stream:
            if self.global_step >= self.cfg.max_steps: break
            self._train_step(batch_ids)
    
    def _train_step(self, batch_ids):
        """One training step. Atomic: either completes the step or skips it."""
        self._maybe_anneal_tau()
        self._maybe_hot_reload_config()
        teacher_out = self._teacher_forward(batch_ids)
        student_out = self._student_forward(batch_ids)
        loss, comps = compute_loss(student_out, teacher_out, self.cfg)
        if not torch.isfinite(loss):
            self._handle_nan(loss, comps)
            return
        loss.backward()
        self._clip_and_step()
        self.global_step += 1
        self._maybe_log(comps)
        self._maybe_eval_and_save()
        # Cleanup
        del batch_ids, teacher_out, student_out, loss, comps
    
    def _maybe_anneal_tau(self):
        if not self.cfg.use_soft_indices: return
        tau = max(self.cfg.tau_final,
                  self.cfg.tau_init * (1.0 - self.global_step / self.cfg.tau_anneal_steps))
        for name, mod in self.student.named_modules():
            if hasattr(mod, "tau"):
                mod.tau = tau
    
    def _maybe_hot_reload_config(self):
        if self.global_step - self.last_hp_check < 10: return
        self.last_hp_check = self.global_step
        if not os.path.exists(self.cfg.hyperparams_file): return
        sig = json.dumps(asdict(self.cfg), sort_keys=True, default=str)
        self.cfg.merge_from_json(self.cfg.hyperparams_file)
        new_sig = json.dumps(asdict(self.cfg), sort_keys=True, default=str)
        if new_sig != sig:
            self.logger.log_config_reload(self.global_step)
            self._apply_groups()
            self._update_lrs()
    
    def _teacher_forward(self, batch_ids):
        stream_t = torch.cuda.Stream()
        with torch.cuda.stream(stream_t):
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                h = self.teacher.model.embed_tokens(batch_ids)
                position_ids = torch.arange(batch_ids.shape[1], device=batch_ids.device).unsqueeze(0)
                pos_emb = self.teacher.model.rotary_emb(h, position_ids) if self.teacher.model.rotary_emb is not None else None
                for layer in self.teacher.model.layers:
                    out = layer(h, position_embeddings=pos_emb) if pos_emb is not None else layer(h)
                    h = out[0] if isinstance(out, tuple) else out
                return h.detach()
    
    def _student_forward(self, batch_ids):
        s_h = self.student.model.embed_tokens(batch_ids)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            position_ids = torch.arange(batch_ids.shape[1], device=batch_ids.device).unsqueeze(0)
            s_pos_emb = self.student.model.rotary_emb(s_h, position_ids) if self.student.model.rotary_emb is not None else None
            for layer in self.student.model.layers:
                if self.cfg.use_gradient_checkpointing and self.student.training:
                    out = torch.utils.checkpoint.checkpoint(layer, s_h, s_pos_emb, use_reentrant=False)
                else:
                    out = layer(s_h, position_embeddings=s_pos_emb) if s_pos_emb is not None else layer(s_h)
                s_h = out[0] if isinstance(out, tuple) else out
            return s_h
    
    def _handle_nan(self, loss, comps):
        self.n_nan_skip += 1
        self.logger.log_nan(self.global_step, loss, comps)
        if self.opt_muon: self.opt_muon.zero_grad(set_to_none=True)
        if self.opt_adamw: self.opt_adamw.zero_grad(set_to_none=True)
        if self.opt_indices: self.opt_indices.zero_grad(set_to_none=True)
        self.global_step += 1
    
    def _clip_and_step(self):
        """Two-tier gradient clipping + three optimizer steps + clamp + schedulers + zero_grad."""
        clip_val = self.cfg.gradient_clip
        indices_params = [p for n, p in self.student.named_parameters()
                          if p.grad is not None and "index_logits" in n]
        other_params = [p for n, p in self.student.named_parameters()
                         if p.grad is not None and "index_logits" not in n]
        grad_norms = {}
        if indices_params:
            gn = torch.nn.utils.clip_grad_norm_(indices_params, 1.0)
            grad_norms["indices"] = gn.item()
        if other_params:
            gn_other = torch.nn.utils.clip_grad_norm_(other_params, clip_val)
            grad_norms["other"] = gn_other.item()
        if self.opt_muon: self.opt_muon.step()
        if self.opt_adamw: self.opt_adamw.step()
        if self.opt_indices: self.opt_indices.step()
        # Clamp index_logits to prevent fp16 overflow
        if self.opt_indices:
            with torch.no_grad():
                for n, p in self.student.named_parameters():
                    if "index_logits" in n:
                        p.data.clamp_(-20.0, 20.0)
        if self.sched_muon: self.sched_muon.step()
        if self.sched_adamw: self.sched_adamw.step()
        if self.sched_indices: self.sched_indices.step()
        if self.opt_muon: self.opt_muon.zero_grad(set_to_none=True)
        if self.opt_adamw: self.opt_adamw.zero_grad(set_to_none=True)
        if self.opt_indices: self.opt_indices.zero_grad(set_to_none=True)
        self._grad_norms = grad_norms
    
    def _maybe_log(self, comps):
        if self.global_step % self.cfg.log_every != 0: return
        cos_val = 1.0 - comps["cos"]
        elapsed = max(1e-6, time.time() - self.start_time)
        tps = (self.global_step - self.best_step) / elapsed
        self.logger.log_step(
            step=self.global_step, loss=comps["loss"], cos=cos_val, tps=tps,
            tau=getattr(self, "_current_tau", None),
            grad_norms=self._grad_norms, lrs=self._current_lrs(),
            gpu_mem=self._gpu_mem(), gpu_util=self._gpu_util(),
        )
    
    def _maybe_eval_and_save(self):
        if self.global_step % self.cfg.eval_every != 0: return
        result = evaluate(self.student, self.teacher, self.eval_tokens,
                          self.cfg.sb_idx, self.cfg, max_batches=8)
        self.logger.log_eval(self.global_step, result)
        if result["cos"] > self.best_cos and result["cos"] > 0:
            self.best_cos = result["cos"]
            self.best_step = self.global_step
            self.best_loss = result["loss"]
            if self.global_step % self.cfg.save_every == 0 or self.global_step >= self.cfg.max_steps:
                out_dir = os.path.join(self.cfg.trained_base, f"superblock_{self.cfg.sb_idx}_best")
                save_state(self.student, self.cfg.sb_idx, self.best_step,
                           self.best_cos, self.best_loss, out_dir)
    
    def _finalize(self):
        if self.best_cos > 0:
            out_dir = os.path.join(self.cfg.trained_base, f"superblock_{self.cfg.sb_idx}_final")
            save_state(self.student, self.cfg.sb_idx, self.global_step,
                       self.best_cos, self.best_loss, out_dir)
        self.logger.log_final(self.best_cos, self.best_step)
        if self.cfg.shutdown_on_done:
            import time, os
            time.sleep(10)
            os.system("shutdown -h now")
    
    def _current_lrs(self):
        lrs = {}
        for opt, name in [(self.opt_muon, "muon"), (self.opt_adamw, "adamw"), (self.opt_indices, "indices")]:
            if opt is None: continue
            for g in opt.param_groups:
                grp = g.get("group", name)
                if grp not in lrs:
                    lrs[grp] = g["lr"]
        return lrs
    
    def _gpu_mem(self):
        try:
            return torch.cuda.memory_allocated() / (1024**3), torch.cuda.memory_reserved() / (1024**3)
        except Exception:
            return None, None
    
    def _gpu_util(self):
        try:
            return torch.cuda.utilization()
        except Exception:
            return None
```

### 4.2 Why this is better

The `Trainer` class exposes eight "hook" methods (`_build_student`, `_build_optimizers`, `_build_schedulers`, `_load_teacher`, `_share_embeddings`, `_prepare_eval_set`, `_train_loop`, `_finalize`) that can be individually overridden in subclasses. For example, a research variant that uses a different teacher forward (e.g., a quantized teacher) can subclass `Trainer` and override only `_teacher_forward`:

```python
class QuantizedTeacherTrainer(Trainer):
    def _load_teacher(self):
        self.teacher = load_int8_teacher(self.cfg.sb_idx)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
```

This is impossible with the current monolithic `train_super_block` function — every variant requires copy-pasting the entire 302-line body.

### 4.3 Testability

The `Trainer` class is unit-testable in isolation. The `setup()` method can be mocked to return a tiny student (e.g., 2 layers, 10K parameters) and a tiny teacher (same). The `_train_step()` method can be called once with a fixed batch, and the resulting `global_step` increment, `best_cos` update, and `logger.log_step` call can be asserted. This is the topic of `07_testing_ci.md`.

---

## 5. The `MetricsLogger` (logging backend)

The current logging is a single `print()` statement at `train_qwen.py:1188`. The `MetricsLogger` class in `qwen_palettize/log.py` supports four backends: stdout (current behavior), wandb, tensorboard, csv. The backend is selected by `cfg.log_backend`.

### 5.1 Interface

```python
# scripts/qwen_palettize/log.py
from .config import Config
import json, time

class MetricsLogger:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._backend = self._init_backend()
    
    def _init_backend(self):
        if self.cfg.log_backend == "stdout":
            return StdoutBackend()
        elif self.cfg.log_backend == "wandb":
            return WandbBackend(self.cfg.wandb_project, self.cfg.wandb_run_name)
        elif self.cfg.log_backend == "tensorboard":
            return TensorboardBackend(self.cfg.logs_dir)
        elif self.cfg.log_backend == "csv":
            return CsvBackend(self.cfg.logs_dir)
        else:
            raise ValueError(f"Unknown log_backend: {self.cfg.log_backend}")
    
    def log_setup(self, trainer):
        self._backend.log_setup(trainer)
    
    def log_step(self, **kwargs):
        self._backend.log_step(**kwargs)
    
    def log_eval(self, step, result):
        self._backend.log_eval(step, result)
    
    def log_nan(self, step, loss, comps):
        self._backend.log_nan(step, loss, comps)
    
    def log_config_reload(self, step):
        self._backend.log_config_reload(step)
    
    def log_final(self, best_cos, best_step):
        self._backend.log_final(best_cos, best_step)
    
    def finish(self):
        self._backend.finish()


class StdoutBackend:
    def log_setup(self, trainer):
        print(f"\n=== Training super-block {trainer.cfg.sb_idx} ===", flush=True)
        # ... (current setup prints)
    
    def log_step(self, **kw):
        step = kw["step"]; loss = kw["loss"]; cos = kw["cos"]; tps = kw["tps"]
        tau = kw.get("tau"); grad_norms = kw.get("grad_norms", {}); lrs = kw.get("lrs", {})
        gpu_mem = kw.get("gpu_mem"); gpu_util = kw.get("gpu_util")
        tau_str = f" tau={tau:.3f}" if tau is not None else ""
        gpu_str = ""
        if gpu_mem is not None and gpu_util is not None:
            gpu_str = f" GPU[{gpu_mem[0]:.1f}/{gpu_mem[1]:.1f}G util={gpu_util}%]"
        gn_str = " ".join(f"{k}={v:.2f}" for k, v in sorted(grad_norms.items()))
        lrs_str = " ".join(f"{k}={v:.1e}" for k, v in sorted(lrs.items()))
        print(f"  step={step:5d} loss={loss:.4f} cos={cos:.4f} tps={tps:.1f}{tau_str}{gpu_str} gn=[{gn_str}] [{lrs_str}]", flush=True)
    
    def log_eval(self, step, result):
        print(f"  [EVAL] step={step} cos={result['cos']:.6f} loss={result['loss']:.6f} ({result['n_seqs']} held-out seqs)", flush=True)
    
    def log_nan(self, step, loss, comps):
        print(f"  [step {step}] NaN loss — skipping", flush=True)
    
    def log_config_reload(self, step):
        print(f"  [step {step}] Hyperparams updated", flush=True)
    
    def log_final(self, best_cos, best_step):
        print(f"\n=== Training complete: best cos={best_cos:.6f} @ step {best_step} ===", flush=True)
    
    def finish(self):
        pass


class WandbBackend:
    def __init__(self, project, run_name):
        import wandb
        self.wandb = wandb
        self.run = wandb.init(project=project, name=run_name)
    
    def log_setup(self, trainer):
        self.run.config.update(trainer.cfg.__dict__, allow_val_change=True)
    
    def log_step(self, **kw):
        self.run.log({k: v for k, v in kw.items() if v is not None and not isinstance(v, tuple)}, step=kw["step"])
    
    def log_eval(self, step, result):
        self.run.log({"eval/cos": result["cos"], "eval/loss": result["loss"]}, step=step)
    
    def log_nan(self, step, loss, comps):
        self.run.log({"nan_skips": 1}, step=step)
    
    def log_config_reload(self, step):
        self.run.log({"config_reload": 1}, step=step)
    
    def log_final(self, best_cos, best_step):
        self.run.summary["best_cos"] = best_cos
        self.run.summary["best_step"] = best_step
    
    def finish(self):
        self.run.finish()


class TensorboardBackend:
    def __init__(self, logs_dir):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(logs_dir)
    
    def log_step(self, **kw):
        step = kw["step"]
        for k, v in kw.items():
            if k == "step" or v is None: continue
            if isinstance(v, dict):
                for sk, sv in v.items():
                    self.writer.add_scalar(f"step/{k}_{sk}", sv, step)
            elif isinstance(v, (int, float)):
                self.writer.add_scalar(f"step/{k}", v, step)
    
    def log_eval(self, step, result):
        self.writer.add_scalar("eval/cos", result["cos"], step)
        self.writer.add_scalar("eval/loss", result["loss"], step)
    
    def log_final(self, *args, **kw):
        self.writer.close()
    
    def finish(self):
        self.writer.close()


class CsvBackend:
    def __init__(self, logs_dir):
        self.path = f"{logs_dir}/metrics.csv"
        self.f = open(self.path, "w")
        self.f.write("step,loss,cos,tps,tau,grad_norm_indices,grad_norm_other,lr_palettes,lr_lora,lr_indices,lr_layernorms,gpu_mem_alloc,gpu_mem_reserved,gpu_util\n")
    
    def log_step(self, **kw):
        # ... (write CSV row)
        pass
    
    def finish(self):
        self.f.close()
```

### 5.2 Why this is better

The current code has 8 `print()` statements scattered across the `train_super_block` function. Each statement formats a different string with different fields. Changing any field's format breaks `sweep_qwen.py`'s regex parser. The `MetricsLogger` centralizes all formatting into the backend, so:

- Adding a new metric (e.g., `grad_norm_indices_per_layer`) requires adding one field to `log_step` and one line to each backend.
- Switching backends (stdout → wandb) requires changing one CLI flag, not editing the training loop.
- Comparing runs across backends is trivial: wandb's UI shows all metrics from all runs in one chart.

The full logging proposal is in `06_logging_metrics.md`.

---

## 6. The other modules (briefly)

The remaining modules (`model.py`, `optim.py`, `loss.py`, `data.py`, `checkpoint.py`, `evaluate.py`, `anneal.py`) are mechanical extractions of the corresponding functions from `train_qwen.py` and `qwen_model.py`. Each is a few hundred lines, each has a clear interface, and each can be unit-tested in isolation. The proposed interfaces are:

### 6.1 `model.py`

```python
# qwen_palettize/model.py
from torch import nn

class PalettizedLinear(nn.Module):
    # (unchanged from qwen_model.py)
    ...

class QwenLoRA(nn.Module):
    # (unchanged)
    ...

class PartialModel(nn.Module):  # ← FIXED: was a plain class
    """Prefix of Qwen3.5: embed_tokens + rotary_emb + first N layers + norm."""
    def __init__(self, embed_tokens, rotary_emb, layers, norm=None, config=None):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.rotary_emb = rotary_emb
        self.layers = nn.ModuleList(layers)  # ← FIXED: was a plain list
        self.norm = norm
        self.config = config
    
    def forward(self, input_ids, position_ids=None):
        h = self.embed_tokens(input_ids)
        pos_emb = self.rotary_emb(h, position_ids) if self.rotary_emb is not None else None
        for layer in self.layers:
            out = layer(h, position_embeddings=pos_emb) if pos_emb is not None else layer(h)
            h = out[0] if isinstance(out, tuple) else out
        if self.norm is not None:
            h = self.norm(h)
        return h

class PartialWrapper(nn.Module):  # ← FIXED: was a plain class
    def __init__(self, partial_model, config):
        super().__init__()
        self.model = partial_model
        self.config = config
    
    def forward(self, input_ids, position_ids=None):
        return self.model(input_ids, position_ids)

def build_student_super_block(sb_idx, lora_rank=16, lora_alpha=32,
                              use_soft_indices=False, device="cuda", dtype=torch.bfloat16):
    # ... (extracted from train_qwen.py:654-740)
    ...

def load_qwen_super_block_only(sb_idx, device="cuda", dtype=torch.bfloat16):
    # ... (extracted from qwen_model.py:load_qwen_super_block_only)
    ...
```

### 6.2 `optim.py`

```python
# qwen_palettize/optim.py
from torch.optim import Optimizer, AdamW

class Muon(Optimizer):
    # (unchanged from train_qwen.py:106-150)
    ...

class FP32MasterOptimizer:
    # (unchanged, but made into a proper Optimizer subclass — see issue #5)
    ...

class FP32MasterAdamW(FP32MasterOptimizer):
    ...

class FP32MasterMuon(FP32MasterOptimizer):
    ...

def build_optimizers(model, cfg, sb_idx):
    # (extracted from train_qwen.py:552-614)
    ...
```

### 6.3 `loss.py`, `data.py`, `checkpoint.py`, `evaluate.py`, `anneal.py`

Each is a straightforward extraction of the corresponding functions. The interfaces are:

- `loss.py`: `compute_loss(student_out, teacher_out, cfg) -> (loss, comps_dict)`
- `data.py`: `stream_training_data(tokenizer, n_seqs, seq_len, device, batch_size, cache_dir=None) -> Iterator[Tensor]`
- `checkpoint.py`: `save_state(model, sb_idx, step, cos, loss, out_dir)`, `load_state(model, resume_dir) -> (step, cos, loss)`
- `evaluate.py`: `prepare_eval_set(tokenizer, device, cache_path, seq_len) -> Tensor`, `evaluate(student, teacher, eval_tokens, sb_idx, cfg, max_batches=8) -> dict`
- `anneal.py`: `update_tau(model, step, cfg)`, `freeze_settled_palettes(model, sb_idx, prev_snapshot, curr_snapshot)`, `snapshot_palette_indices(model) -> dict`

---

## 7. Migration plan

The refactor is done in three commits, each preserving the cos=0.953 baseline:

### Commit 1: Extract `Config` and `MetricsLogger` (no behavior change)

1. Create `qwen_palettize/config.py` with the `Config` dataclass.
2. Create `qwen_palettize/log.py` with the four backends.
3. Modify `train_qwen.py` to construct a `Config` from CLI args, but otherwise use the same training loop. The loop still calls `print()`, but through `MetricsLogger` (which routes to `StdoutBackend`).
4. Run training for 100 steps. Verify cos matches the baseline.

### Commit 2: Extract `Trainer` and other modules

1. Move `PalettizedLinear`, `QwenLoRA`, `PartialModel`, `PartialWrapper` to `qwen_palettize/model.py`.
2. Move `Muon`, `FP32MasterOptimizer`, etc. to `qwen_palettize/optim.py`.
3. Move `compute_loss` to `qwen_palettize/loss.py`.
4. Move `stream_training_data` to `qwen_palettize/data.py`.
5. Move `save_state`, `load_state` to `qwen_palettize/checkpoint.py`.
6. Move `prepare_eval_set`, `evaluate` to `qwen_palettize/evaluate.py`.
7. Move `update_tau`, `freeze_settled_palettes`, etc. to `qwen_palettize/anneal.py`.
8. Extract `Trainer` from the `train_super_block` function. The `train_qwen.py` becomes a thin CLI shim.
9. Run training for 1000 steps. Verify cos matches the baseline.

### Commit 3: Fix `PartialWrapper` (the breaking change)

1. Change `PartialModel` to inherit from `nn.Module`.
2. Change `self.layers = layers` to `self.layers = nn.ModuleList(layers)`.
3. Add `forward()` method to both `PartialModel` and `PartialWrapper`.
4. Delete the hand-rolled `parameters()`, `named_parameters()`, `named_modules()`, `to()`, `eval()`, `train()`, `get_submodule()` methods.
5. Add a one-time migration for legacy `.pt` checkpoints.
6. Run training for 1000 steps. Verify cos matches the baseline.
7. Run `torch.compile(student)` — verify it no longer raises.
8. Set `use_gradient_checkpointing=1` — verify the batch size can be raised to 128 without OOM.

Each commit is independently reviewable. Each commit preserves the cos=0.953 baseline. The full refactor is ~3 days of work (see `09_refactoring_roadmap.md`).

---

## 8. What this enables

After the refactor:

- `torch.compile(student)` works → 1.5–2× speedup.
- Gradient checkpointing works → batch=128, 4× speedup.
- `accelerate.prepare(student, opt)` works → free DDP, free FSDP.
- `transformers.Trainer` works → free LR scheduling, free W&B logging, free checkpointing.
- `lightning.LightningModule` works → clean separation of concerns.
- Unit tests for `compute_loss`, `apply_groups`, `build_optimizers` work → 60-second smoke test replaces the 1.5-hour training regression.
- Hyperparameter sweeps via `sweep_qwen.py` work → no fragile `print()` regex parsing.
- The `Config` object can be serialized to JSON for reproducibility → `cfg.to_json()` is one call.
- The `Trainer` class can be subclassed for research variants → no more copy-pasting the 302-line function.

The refactor is the **single highest-leverage architectural change** in the roadmap. It is the prerequisite for issues #5 (mixed precision), #6 (data prefetch — the `Trainer` class accepts a `data_loader` argument), #7 (logging — the `MetricsLogger` is the centralized sink), #8 (testing — the `Trainer` class is unit-testable), and #9 (configuration — the `Config` dataclass is the single source of truth).

---

## 9. Risks and mitigations

The refactor has two risks:

1. **Behavioral drift.** Each commit must reproduce the cos=0.953 baseline to within floating-point tolerance. The migration plan above calls for 100 steps of verification after Commit 1, 1000 steps after Commit 2, 1000 steps after Commit 3. If any verification fails, the commit is reverted and the cause is identified before proceeding.

2. **Test coverage.** The refactor introduces new code paths (e.g., the `nn.Module`-based `PartialWrapper`) that have no test coverage. The migration is brittle if any code path is not tested. **Mitigation**: write the unit tests in `07_testing_ci.md` *before* Commit 3, and run them after Commit 3 to verify no regression.

The refactor is **not** a rewrite. Every function in the new `qwen_palettize/` package is a verbatim extraction of the corresponding function from `train_qwen.py` or `qwen_model.py`, with the `PartialWrapper` and `PartialModel` fixes from `02_partial_wrapper_problem.md` applied. The risk of behavioral drift is low because the diff is mechanical.
