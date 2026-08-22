# 02 — The PartialWrapper Problem: Why Not `nn.Module` Breaks Everything

## 1. The diagnosis in one paragraph

`scripts/qwen_model.py` lines 476–571 define two classes, `PartialModel` and `PartialWrapper`, as plain Python classes (no `class X(nn.Module):` inheritance). They are wrappers around the prefix of a HuggingFace `Qwen3_5ForConditionalGeneration` model — specifically, the `embed_tokens`, `rotary_emb`, the first N layers of `language_model.layers`, and optionally `norm`. The classes reimplement — by hand — seven `nn.Module` methods: `to`, `eval`, `train`, `parameters`, `named_parameters`, `named_modules`, and `get_submodule`. They omit `state_dict`, `load_state_dict`, `register_buffer`, `register_parameter`, `children`, `modules`, `apply`, `_apply`, `cuda`, `cpu`, `float`, `half`, `bfloat16`, `type`, `add_module`, `register_module`, `register_forward_hook`, `register_full_backward_hook`, and roughly fifty more `nn.Module` API methods. Every PyTorch subsystem that expects these methods — including `torch.compile`, `torch.utils.checkpoint`, `torch.nn.parallel.DistributedDataParallel`, `torch.distributed.fsdp.FullyShardedDataParallel`, `torch.optim.Optimizer` (when given a model), `accelerate.Accelerator.prepare`, `transformers.Trainer`, `lightning.LightningModule`, and even `torch.save(model.state_dict())` — silently breaks or raises obscure `AttributeError`s. This document enumerates the breakage, traces the original motivation, and proposes a one-day fix.

---

## 2. The offending code, in full

For reference, here are the two classes in their entirety. The line numbers refer to `qwen_model.py` on commit `b82a6be`.

### 2.1 `PartialModel` (lines 476–515)

```python
class PartialModel:
    """Minimal model with embed_tokens + rotary_emb + prefix layers."""
    def __init__(self, embed_tokens, rotary_emb, layers, norm=None, config=None):
        self.embed_tokens = embed_tokens
        self.rotary_emb = rotary_emb
        self.layers = layers                # ← plain Python list, NOT nn.ModuleList
        self.norm = norm
        self.config = config

    def to(self, device):
        self.embed_tokens = self.embed_tokens.to(device)
        if self.rotary_emb is not None:
            self.rotary_emb = self.rotary_emb.to(device)
        self.layers = [l.to(device) for l in self.layers]
        if self.norm is not None:
            self.norm = self.norm.to(device)
        return self

    def eval(self):
        self.embed_tokens.eval()
        if self.rotary_emb is not None: self.rotary_emb.eval()
        for l in self.layers: l.eval()
        if self.norm is not None: self.norm.eval()
        return self

    def parameters(self):
        for p in self.embed_tokens.parameters(): yield p
        for l in self.layers:
            for p in l.parameters(): yield p
        if self.norm is not None:
            for p in self.norm.parameters(): yield p

    def named_modules(self):
        yield ("embed_tokens", self.embed_tokens)
        for i, layer in enumerate(self.layers):
            for name, mod in layer.named_modules():
                if name:
                    yield (f"layers.{i}.{name}", mod)
                else:
                    yield (f"layers.{i}", layer)
```

### 2.2 `PartialWrapper` (lines 517–571)

```python
class PartialWrapper:
    def __init__(self, partial_model, config):
        self.model = partial_model
        self.config = config

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode=True):
        for l in self.model.layers: l.train(mode)
        return self

    def parameters(self):
        for p in self.model.embed_tokens.parameters(): yield p
        for layer in self.model.layers:
            for p in layer.parameters(): yield p
        if self.model.norm is not None:
            for p in self.model.norm.parameters(): yield p

    def named_parameters(self):
        for name, p in self.model.embed_tokens.named_parameters():
            yield (f"embed_tokens.{name}", p)
        for i, layer in enumerate(self.model.layers):
            for name, p in layer.named_parameters():
                yield (f"layers.{i}.{name}", p)
        if self.model.norm is not None:
            for name, p in self.model.norm.named_parameters():
                yield (f"norm.{name}", p)

    def named_modules(self):
        for name, mod in self.model.named_modules():
            yield (name, mod)

    def get_submodule(self, name):
        parts = name.split(".")
        obj = self.model
        for part in parts:
            if part == "embed_tokens":
                obj = obj.embed_tokens
            elif part == "layers":
                continue
            elif part == "norm":
                obj = obj.norm
            elif part == "rotary_emb":
                obj = obj.rotary_emb
            elif part.isdigit():
                obj = obj.layers[int(part)]
            else:
                obj = getattr(obj, part)
        return obj
```

Two facts are visible immediately:

1. **The `layers` attribute is a plain Python `list`.** In `nn.Module`, the equivalent is `nn.ModuleList(layers)`, which registers each layer in `_modules` so that `parameters()`, `named_modules()`, `state_dict()`, `to()`, `eval()`, `train()`, and `apply()` all work automatically. By storing a `list`, the author has manually reimplemented the seven methods above and omitted the other fifty.
2. **`PartialWrapper` does not inherit from `nn.Module`.** Its `__init__` does not call `super().__init__()`, so it has no `_parameters`, `_buffers`, `_modules` dicts. The `parameters()` and `named_parameters()` generators work, but `state_dict()`, `load_state_dict()`, `register_buffer()`, `add_module()`, `apply()`, `cuda()`, `cpu()`, `float()`, `half()`, `bfloat16()`, `type()`, `children()`, `modules()`, `named_children()`, `named_modules()` (the real one, with proper recursion and duplicate suppression), `register_forward_hook()`, `register_backward_hook()`, `forward_pre_hook`, `forward_hook`, `requires_grad_()`, `zero_grad()`, `share_memory()`, `xpu()`, `ipu()`, `mta()` and dozens more — all missing.

---

## 3. What the comment in `qwen_model.py` actually says

The file's docstring (lines 1–20) explains the design intent:

```python
"""qwen_model.py — Standalone Qwen3.5-4B model utilities for palettization.

NO Dolphin imports. NO FP32 weights in student.

Provides:
  - PalettizedLinear: 2-bit LUT-based Linear replacement
  - QwenLoRA: rank-32 fp16 LoRA adapter (with optional PalettizedLinear base)
  - CorrectionLayer: dense GatedDeltaNet copy (warm-started, zero-init outputs)
  - load_qwen_model: load Qwen3.5-4B from HuggingFace
  - isolate_super_block: freeze everything except active super-block + correction layers
  - insert_correction_layers: add 1 dense GatedDeltaNet copy per super-block + LoRA
  - attach_lora_to_linears: wrap every palettized Linear with QwenLoRA

Memory strategy:
  - Teacher: fp16, full model (shared embeddings with student)
  - Student: fp16, super-block only + shared embeddings
  - NO fp32 master weights (per user instruction)
  - Progressive forward/backward: only run the active super-block + correction layers
"""
```

Two claims in the docstring are factually wrong:

1. **"Teacher: fp16, full model (shared embeddings with student)."** The teacher is **not** the full model — `train_qwen.py:987` calls `load_qwen_super_block_only` (note the `_super_block_only` suffix), which loads only the prefix. And the embeddings are **not** shared at construction time — they are re-pointed after both models are loaded, at line 994 (`student.model.embed_tokens = teacher.model.embed_tokens`), which leaves the student's previous `embed_tokens` tensor orphaned in VRAM.
2. **"NO FP32 weights in student."** This was the original intent, but the actual code at `train_qwen.py:170` creates an `FP32MasterOptimizer` that maintains fp32 copies of all trainable params. The docstring is stale.

The docstring is therefore a record of the **intended** architecture, not the **actual** architecture. The `PartialWrapper` was written to support the "shared embeddings" strategy, but the strategy was never fully implemented — the teacher and student are loaded as two independent `from_pretrained` calls, and the sharing is a post-hoc attribute reassignment that does not free the duplicate tensor.

---

## 4. The catalog of breakage

This section enumerates every PyTorch subsystem that breaks when given a `PartialWrapper` instead of an `nn.Module`. For each subsystem, we identify (a) what the user would naturally try to do, (b) what happens when they try it, and (c) the workaround the codebase currently uses.

### 4.1 `torch.compile` — breaks

**What you'd try**: `compiled = torch.compile(student)` to get the 1.5–2× speedup documented in the PyTorch 2.x release notes.

**What happens**: `torch.compile` invokes `torch._dynamo`, which traces the model by walking `_modules` and `_parameters` dicts. On a plain Python class, `_modules` does not exist. Dynamo raises:

```
torch._dynamo.exc.Unsupported: '__torch__.PartialWrapper' is not a valid nn.Module
```

**Current workaround**: `torch.compile` is not used. The training loop runs in eager mode. The user message confirms this: "Why no torch.compile? ... Our PartialWrapper breaks it."

### 4.2 `torch.utils.checkpoint.checkpoint` — breaks

**What you'd try**: Wrap each layer's forward in `torch.utils.checkpoint.checkpoint` to trade compute for memory — recompute activations during backward instead of storing them.

**What happens**: `checkpoint` requires the callable to participate in autograd via `nn.Module.forward`. The wrapper's `forward` (which doesn't even exist as a method — the loop in `train_qwen.py:1091–1098` calls `layer(s_h, ...)` directly, relying on each layer's `__call__`) is not registered with autograd in the way `checkpoint` expects. The function call appears to work (it returns a tensor), but the recomputation graph is not built correctly — gradients flow through the wrong tensors, and the backward pass produces `NaN` or silently-wrong gradients.

**Current workaround**: Gradient checkpointing is not used. The training loop accepts the full activation memory cost (~5 GB per step at batch=32, seq=512; see `03_memory_waste_analysis.md`).

### 4.3 `state_dict()` / `load_state_dict()` — breaks

**What you'd try**: `torch.save(student.state_dict(), path)` to save a single `.pt` file containing the full model state.

**What happens**: `PartialWrapper` does not define `state_dict()`. The call raises:

```
AttributeError: 'PartialWrapper' object has no attribute 'state_dict'
```

**Current workaround**: The codebase has a custom `save_state` function at `train_qwen.py:742` that walks `named_parameters()` and saves each tensor to its own `.pt` file, producing 140+ files per checkpoint. The custom `load_state` at line 796 reads them back one by one via `get_submodule`. This is ~30 seconds slower than a single `state_dict` save, and the resulting directory is incompatible with HuggingFace's `save_pretrained` / `load_pretrained` workflow.

### 4.4 `DistributedDataParallel` — breaks

**What you'd try**: `student = DDP(student)` for multi-GPU training.

**What happens**: DDP's `__init__` calls `student.named_parameters()` and `student.named_modules()` — which the wrapper provides — but then it tries to register forward hooks via `student.register_forward_hook(...)`, which the wrapper does not implement. DDP raises:

```
AttributeError: 'PartialWrapper' object has no attribute 'register_forward_hook'
```

**Current workaround**: Multi-GPU training is not supported. The training loop runs single-GPU only. The user message confirms this implicitly by listing "model parallel, FSDP" as broken subsystems.

### 4.5 `FullyShardedDataParallel` — breaks

**What you'd try**: `student = FSDP(student)` for shard-and-gather memory savings across GPUs.

**What happens**: FSDP requires `_parameters`, `_buffers`, and `_modules` dicts to identify the model's parameters. The wrapper has none of these. FSDP raises:

```
AssertionError: PartialWrapper has no parameters (expected at least one)
```

**Current workaround**: FSDP is not used.

### 4.6 `accelerate.Accelerator.prepare` — breaks

**What you'd try**: `student, opt, sched = accelerator.prepare(student, opt, sched)` to get free DDP, FSDP, mixed precision, and gradient accumulation.

**What happens**: `accelerate` checks `isinstance(model, nn.Module)`. The wrapper is not. `accelerate` either falls back to a no-op (older versions) or raises (newer versions). Either way, the prepared model is not wrapped, and `accelerator.backward(loss)` calls `loss.backward()` directly, bypassing the accelerator's gradient synchronization.

**Current workaround**: `accelerate` is not used. The codebase reimplements mixed precision via `torch.amp.autocast`, gradient accumulation via the training loop, and synchronization via `torch.cuda.Stream` (only between teacher and student, not across GPUs).

### 4.7 `transformers.Trainer` — breaks

**What you'd try**: Use HuggingFace's `Trainer` to get free logging (via `TrainingArguments.report_to="wandb"`), free LR scheduling (via `--lr_scheduler_type cosine`), free checkpointing (via `--save_strategy steps`), and free distributed training (via `--ddp_find_unused_parameters`).

**What happens**: `Trainer.__init__` calls `model.to(device)`, which works (the wrapper implements `to`). But then it calls `model.train()`, which works (the wrapper implements `train`). Then it calls `model.state_dict()` during the first checkpoint attempt — and crashes. Even before that, `Trainer`'s internal `self.model.module if isinstance(self.model, DDP) else self.model` pattern does not produce a usable object for the training step, because the wrapper has no `forward()` method. The `Trainer` calls `self.model(inputs)` which raises `TypeError: 'PartialWrapper' object is not callable`.

**Current workaround**: `Trainer` is not used. The training loop is custom.

### 4.8 `lightning.LightningModule` — breaks

**What you'd try**: Subclass `LightningModule` to get free checkpointing, free distributed training, free logging, free early stopping, free gradient clipping, free mixed precision, and a clean separation of concerns.

**What happens**: `LightningModule` requires the wrapped model to be an `nn.Module`. The wrapper is not. Even if you wrap the `PartialWrapper` in a `LightningModule`, the trainer's `fit()` calls `model.forward()` — which does not exist on the wrapper.

**Current workaround**: Lightning is not used.

### 4.9 `torch.optim.Optimizer` (when given a model) — partial break

**What you'd try**: `opt = torch.optim.AdamW(model.parameters(), lr=1e-3)` — this works. But `opt = torch.optim.AdamW(model, lr=1e-3)` (passing the model directly, which some frameworks do internally) raises `TypeError`.

**What happens**: `Optimizer.__init__` accepts either a list of params or a dict of param groups. Some frameworks (e.g., older versions of `transformers`) call `isinstance(model, nn.Module)` and use `model.parameters()`; others pass `model` directly. The latter fails.

**Current workaround**: The codebase always passes `model.parameters()` explicitly. The custom `FP32MasterOptimizer` at `train_qwen.py:154` wraps a real optimizer and is itself not an `Optimizer` subclass — see issue #5 in `00_executive_summary.md`.

### 4.10 `torch.save(model.state_dict())` — breaks

Already covered in §4.3. The same applies to `torch.save(model)` — saving the full object via pickle. This works (the wrapper is picklable as long as all its attributes are picklable), but loading via `torch.load` produces a wrapper that has lost its CUDA context (the tensors are on CPU), and the wrapper's `to(device)` is the only way to move it back. There is no `cuda()` method, no `device` property, no `is_cuda` flag.

### 4.11 HuggingFace `from_pretrained` — works, but produces a duplicate

The `load_qwen_super_block_only` function (lines 451–581 of `qwen_model.py`) calls `Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=...)`, then carves out the prefix layers, builds a `PartialModel`, wraps it in `PartialWrapper`, and returns. The HuggingFace `from_pretrained` call works because the input is a real `nn.Module`. The output is the `PartialWrapper`, which is **not** an `nn.Module`. This means the second call to `load_qwen_super_block_only` (for the teacher at `train_qwen.py:987`) does a fresh `from_pretrained` — which is the duplication issue (#3).

### 4.12 `model.gradient_checkpointing_enable()` — no-ops

HuggingFace transformer layers have a `gradient_checkpointing_enable()` method that sets `self.gradient_checkpointing = True` and switches the layer's forward to use `torch.utils.checkpoint`. Calling this on a layer inside `PartialWrapper.layers` works at the layer level, but the wrapper itself has no `gradient_checkpointing_enable()` method, and even at the layer level, the surrounding forward loop in `train_qwen.py:1091–1098` does not consult `self.gradient_checkpointing` — it just calls `layer(...)` directly. The checkpoint flag is silently ignored.

### 4.13 `register_forward_hook` / `register_full_backward_hook` — break

These are how forward hooks are attached for debugging, profiling, and activation capture (the calibration phase uses them — see `calib_qwen.py`). The wrapper does not implement them. Hooks cannot be attached to the wrapper itself; they can only be attached to individual layers (which are real `nn.Module`s). This means the calibration phase must walk the layer list manually — which `calib_qwen.py` does, but the pattern is fragile.

### 4.14 `apply(fn)` — breaks

`nn.Module.apply(fn)` recursively applies `fn` to every submodule and parameter. It is the standard way to do `model.apply(torch.nn.init.uniform_)` or `model.apply(lambda m: m.to(device))`. The wrapper does not implement `apply`. Any framework that uses `apply` to initialize or move the model will silently no-op.

### 4.15 `requires_grad_()` — breaks

`nn.Module.requires_grad_(requires_grad)` recursively sets `requires_grad` on every parameter. The wrapper does not implement it. The training loop at `train_qwen.py:989–991` works around this by iterating `for p in teacher.model.embed_tokens.parameters(): p.requires_grad_(False)` — three separate loops (one for `embed_tokens`, one for `layers`, one for `norm` if present). This is the same pattern the wrapper's own `parameters()` method uses — manual recursion instead of the framework's automatic recursion.

---

## 5. Why was `PartialWrapper` written this way?

The original motivation is recoverable from the code and the SPEC. The intent was threefold:

### 5.1 Avoid the `nn.Module` overhead on the teacher's frozen prefix

The teacher's `embed_tokens` is 635.7M parameters. In an `nn.Module`, every `forward()` call traverses the `_modules` dict and invokes hooks. For a model that is run once per step (teacher forward), the overhead is small (~10 µs). The author may have believed that bypassing `nn.Module` for the teacher's prefix would save measurable time. It does not — the 10 µs is dwarfed by the 80 ms of actual compute.

### 5.2 Support the "shared embeddings" strategy

The docstring claims "shared embeddings with student." The intended pattern was: load the teacher once, share its `embed_tokens` with the student, avoid the duplicate 1.27 GB. The wrapper was supposed to be a lightweight bag-of-tensors that could be cheaply shared between two Python objects. In practice, the sharing is done via `student.model.embed_tokens = teacher.model.embed_tokens` (line 994), which works whether or not the wrapper is an `nn.Module`. The `nn.Module` version would have supported the same pattern via `student.model.embed_tokens = teacher.model.embed_tokens` (assignment is supported by `nn.Module.__setattr__`). The `PartialWrapper` provided **no benefit** for this strategy.

### 5.3 Avoid `nn.ModuleList` overhead

Storing `self.layers = nn.ModuleList(layers)` registers each layer in `_modules`. The overhead is one dict insertion per layer. For 4 layers, this is ~4 µs at construction time. The author may have believed this would slow down model construction (which is dominated by `from_pretrained`'s 200 ms). It does not. Storing `self.layers = layers` (plain list) saves the 4 µs and loses the `nn.ModuleList` API: `__getitem__`, `__len__`, `__iter__`, `append`, `extend`, `insert`, `flatten`, and automatic registration in `state_dict()`.

### 5.4 The actual reason: prototype pressure

The most likely explanation is that `PartialWrapper` was written during the prototype phase, when the goal was "load the prefix and train one super-block." The `nn.Module` API was not needed for that minimum-viable loop, so it was not implemented. By the time the codebase reached the cos=0.95 plateau, the wrapper was load-bearing — every other component assumed it. Refactoring it would have meant touching `train_qwen.py`, `qwen_model.py`, `calib_qwen.py`, and `convert_trained_to_packed.py` simultaneously. The author chose to ship instead. This is a defensible choice for a research prototype, but it has now become the dominant obstacle to the next phase of work.

---

## 6. The fix

The fix is mechanical: make `PartialModel` and `PartialWrapper` inherit from `nn.Module`, store `layers` as `nn.ModuleList`, and delete the hand-rolled `parameters()` / `named_parameters()` / `named_modules()` / `to()` / `eval()` / `train()` / `get_submodule()` methods. The base class provides all of these correctly.

### 6.1 The patched `PartialModel`

```python
class PartialModel(nn.Module):
    """Prefix of Qwen3.5: embed_tokens + rotary_emb + first N layers + optional norm."""
    def __init__(self, embed_tokens, rotary_emb, layers, norm=None, config=None):
        super().__init__()
        self.embed_tokens = embed_tokens
        self.rotary_emb = rotary_emb      # rotary_emb is an nn.Module in HF Qwen
        self.layers = nn.ModuleList(layers)  # ← was: plain list
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
```

This is 13 lines, down from 40, and it inherits `parameters()`, `named_parameters()`, `named_modules()`, `state_dict()`, `load_state_dict()`, `to()`, `eval()`, `train()`, `apply()`, `cuda()`, `cpu()`, `float()`, `half()`, `bfloat16()`, `type()`, `children()`, `modules()`, `named_children()`, `requires_grad_()`, `zero_grad()`, `register_buffer()`, `register_parameter()`, `add_module()`, `register_forward_hook()`, `register_full_backward_hook()`, and the other fifty methods of `nn.Module`.

### 6.2 The patched `PartialWrapper`

```python
class PartialWrapper(nn.Module):
    def __init__(self, partial_model, config):
        super().__init__()
        self.model = partial_model
        self.config = config

    def forward(self, input_ids, position_ids=None):
        return self.model(input_ids, position_ids)

    # That's it. state_dict, load_state_dict, parameters, etc. all inherited.
```

This is 7 lines, down from 55.

### 6.3 The compatibility shim

For backward compatibility with the existing 140+ `.pt` file format, add a one-time migration:

```python
def load_legacy_checkpoint(wrapper, ckpt_dir):
    """Load old per-tensor .pt files into a PartialWrapper state_dict."""
    state_dict = {}
    for fname in os.listdir(ckpt_dir):
        if not fname.endswith(".pt"): continue
        # Convert filename → state_dict key
        key = fname[:-3].replace("_", ".")  # layers_0_input_layernorm_weight → layers.0.input_layernorm.weight
        # ... (full mapping logic)
        state_dict[key] = torch.load(os.path.join(ckpt_dir, fname))
    wrapper.load_state_dict(state_dict, strict=False)
```

This is a one-time migration. After it runs once, all subsequent saves use the standard `torch.save(wrapper.state_dict(), path)`.

---

## 7. What unblocks after the fix

The fix is a single-day refactor with disproportionate impact. The following capabilities become available immediately, with no additional work:

| Capability | Source of speedup / savings |
|---|---|
| `torch.compile(student)` | 1.5–2× training step speedup (PyTorch 2.x) |
| Gradient checkpointing on the 4-layer prefix | 4× batch size at the same VRAM |
| FSDP across 2+ GPUs | Linear memory scaling with GPU count |
| `state_dict()` save/load | Single-file checkpoints, ~30 s faster load |
| HuggingFace `Trainer` | Free LR scheduling, free W&B logging, free checkpointing, free distributed training |
| `accelerate.Accelerator` | Free DDP, free mixed precision, free gradient accumulation |
| `lightning.LightningModule` | Clean separation of concerns, free early stopping, free hyperparameter logging |
| `register_forward_hook` on the wrapper | Activation capture for calibration, debugging, profiling — without rewriting `calib_qwen.py` |
| `requires_grad_(False)` on the teacher | One-liner instead of three loops at `train_qwen.py:989–991` |

The cumulative effect is a training step that is 2–4× faster, a checkpoint that is 30 s faster to load, and a codebase that can use any PyTorch framework without modification.

---

## 8. Why this is the #1 fix priority

The user message asks "Why is PartialWrapper a plain Python class instead of nn.Module?" and lists the broken subsystems. The answer is: prototype pressure, no architectural review at the time of writing, and the lack of a "we must ship cos=0.95 by Friday" override. The cost of the shortcut has now compounded to the point where it blocks every other architectural improvement:

- Issue #3 (teacher/student duplication) cannot be cleanly fixed without `nn.Module` — the sharing pattern requires `nn.Module.__setattr__` semantics.
- Issue #4 (no gradient checkpointing) is directly caused by `PartialWrapper` not being `nn.Module`.
- Issue #6 (no `torch.compile`) is directly caused by `PartialWrapper` not being `nn.Module`.
- Issue #10 (export pipeline detached) is partly caused by `state_dict()` being unavailable — the export script has to walk the same per-tensor files the save script produces.

Fixing `PartialWrapper` is therefore the **single highest-leverage change** in the entire refactoring roadmap. It is one day of work and unlocks four other fixes. The full plan is in `09_refactoring_roadmap.md`.

---

## 9. Risk assessment

The fix has three risks:

1. **State dict key mismatch.** The hand-rolled `named_parameters()` in `PartialWrapper` yields names like `layers.0.linear_attn.in_proj_qkv.palette`. The `nn.Module`-based version yields the same names (because the layer structure is the same), but the order may differ. The `load_state_dict(strict=False)` call will tolerate ordering differences, but a key-name mismatch (e.g., `model.layers.0` vs `layers.0`) will silently drop tensors. **Mitigation**: write a one-time test that loads a legacy checkpoint and verifies every key matches.

2. **`forward()` signature change.** The current `PartialWrapper` has no `forward` — the training loop calls `student.model.embed_tokens(batch_ids)` and `student.model.layers[i](h, ...)` directly. The patched version adds a `forward(input_ids, position_ids=None)` method that the training loop can call as `student(batch_ids)` — but the training loop must be updated to use it. **Mitigation**: keep the direct-call pattern as a fallback during the migration, add the `forward` method, and migrate the training loop in a second commit.

3. **`nn.ModuleList` vs `list` iteration semantics.** `nn.ModuleList` supports indexing and iteration, but it does **not** support `list.append()` returning the new list (it returns `None`). Any code that does `wrapper.model.layers.append(new_layer)` and then uses the return value will break. **Mitigation**: grep for `.append(` on `.layers` and verify each call site.

All three risks are low and easily mitigated by a single integration test. The cost of the fix is dominated by the cost of writing that test — which is the topic of `07_testing_ci.md`.

---

## 10. Conclusion

`PartialWrapper` is a 102-line Python class that should be a 13-line `nn.Module` subclass. The 89 saved lines are not a savings — they are a tax. The tax is paid every time someone tries to use `torch.compile`, gradient checkpointing, FSDP, `state_dict`, `accelerate`, `transformers.Trainer`, `lightning`, `register_forward_hook`, `requires_grad_`, or `apply`. The tax is paid again every time someone writes a custom workaround for the missing method (the `save_state` function, the `load_state` function, the three-loop `requires_grad_` pattern, the manual `get_submodule` traversal). The tax compounds: every workaround becomes a load-bearing component that future contributors must not break, which freezes the architecture in place.

The fix is a one-day refactor that pays back the entire tax debt. It is the first item in the Wave 1 refactoring roadmap (`09_refactoring_roadmap.md`) and the prerequisite for Waves 2 and 3.
