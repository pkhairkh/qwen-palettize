# 03 — Optimizer & Speed Enhancement Patches (8-9)

> **Wave 2 deliverable.** Full detail for the 2 optimizer/speed enhancements. Code patches included as REFERENCE ONLY — not yet applied.

---

## Patch 8: Fused AdamW for Indices

**Source:** [Agent 2 (kernel-efficiency), `08_recommendations.md` Patch 6; Agent 5 (architecture-review), `03_memory_waste_analysis.md`]

### Problem

Current optimizer for `index_logits`: `FP32MasterAdamW` (custom wrapper around `torch.optim.AdamW`).

The `FP32MasterOptimizer` wrapper (in `train_qwen.py:154-200`):
1. Copies bf16 model grads → fp32 master grads
2. Steps on fp32 masters (8 passes, ~200 kernel launches)
3. Copies fp32 masters back → bf16 model params

This takes **113ms/step** for 1.78B index_logits params. The overhead comes from:
- 8 separate kernel passes (copy grad, step m, step v, bias correction, update, copy back, zero grad)
- ~200 kernel launches per step (one per param group)
- No kernel fusion

### Fix

Use `torch.optim.AdamW(fused=True)` — PyTorch's built-in fused AdamW that combines all passes into a **single kernel launch**.

**Alternative:** `bitsandbytes.optim.AdamW8bit` — uses 8-bit optimizer state, ~3× faster than fp32, saves 14GB VRAM.

### Expected Impact

- **Optimizer step: 113ms → ~10-20ms** (saves ~90-100ms/step)
- With bitsandbytes 8-bit: saves ~14GB VRAM (21.4GB → 7.1GB optimizer state)
- Enables larger batch sizes with freed VRAM

### Code Patch (REFERENCE ONLY — not yet applied)

**File:** `scripts/train_qwen.py`
**Lines:** ~563-597 (build_optimizers)

#### Option A: PyTorch fused AdamW

```python
# BEFORE (current, train_qwen.py:594-597):
    opt_indices = FP32MasterAdamW(plain_adamw_groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0) if plain_adamw_groups else None

# AFTER (Option A — PyTorch fused):
    # Fused AdamW: single kernel launch for all param groups
    # NOTE: fused=True requires fp32 params (not bf16/fp16)
    # index_logits are fp16 → must cast to fp32 first, or use bitsandbytes
    opt_indices = torch.optim.AdamW(
        plain_adamw_groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
        fused=True  # ← single kernel launch
    ) if plain_adamw_groups else None
```

**Problem:** `fused=True` requires fp32 params. Our `index_logits` are fp16 (kernel requirement). Options:
1. Cast index_logits to fp32 (breaks the soft kernel's fp16 requirement)
2. Use bitsandbytes 8-bit (handles mixed precision)

#### Option B: bitsandbytes 8-bit AdamW (RECOMMENDED)

```python
# AFTER (Option B — bitsandbytes 8-bit):
    import bitsandbytes as bnb
    # AdamW8bit: 8-bit state (m, v), handles fp16/bf16 params natively
    # State: 1.78B × 2 bytes (8-bit m+v) = 3.56 GB (was 21.4 GB with fp32 master)
    opt_indices = bnb.optim.AdamW8bit(
        plain_adamw_groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0
    ) if plain_adamw_groups else None
```

#### Option C: Hybrid (fp32 master + fused step)

Keep fp32 master (for numerical stability) but use a custom fused CUDA kernel for the AdamW step:

```python
# AFTER (Option C — custom fused kernel):
    # Keep FP32MasterAdamW but add a fused step kernel
    opt_indices = FP32MasterAdamW(plain_adamw_groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
                                   fused=True)  # ← new flag, requires custom CUDA kernel
```

This requires writing a custom CUDA kernel that fuses: copy_grad + adamw_step + copy_back into one launch. More work but preserves numerical stability.

### Verification

After applying:
- Optimizer step time: 113ms → ~20ms (Option B) or ~10ms (Option C)
- No NaN (8-bit state is less precise but stable for Gumbel-Softmax)
- VRAM: 21.4GB → 7.1GB (Option B) — enables batch=64
- `gn=[indices=...]` should be same magnitude (8-bit state doesn't change gradient)

### Dependencies

- **Option B:** `pip install bitsandbytes` (latest stable)
- **Option C:** Custom CUDA kernel for fused AdamW step (new file in `fused_lut_kernel.cu`)
- All options require `index_logits` to remain fp16 for the soft kernel

### Risks

- **Option A (fused=True):** Requires fp32 params — breaks soft kernel. NOT RECOMMENDED.
- **Option B (bitsandbytes 8-bit):** 8-bit state may lose precision for very small gradients. Test for NaN.
- **Option C (custom fused):** Most work, but best precision + speed. ~200 lines of CUDA.

### Recommendation

**Use Option B (bitsandbytes 8-bit)** — it's the fastest to implement, handles fp16 params natively, and saves the most VRAM. If precision issues arise, fall back to Option C.

---

## Patch 9: PartialWrapper → nn.Module

**Source:** [Agent 5 (architecture-review), `02_partial_wrapper_problem.md`, `09_refactoring_roadmap.md`; Agent 2 (kernel-efficiency), `07_literature_comparison.md`]

### Problem

`PartialWrapper` and `PartialModel` (in `qwen_model.py:438-600`) are **plain Python classes**, NOT `nn.Module` subclasses. This breaks:
- **`torch.compile`** — requires `nn.Module` for graph capture (1.5-2× speedup)
- **Gradient checkpointing** — `torch.utils.checkpoint.checkpoint()` requires `nn.Module`
- **FSDP/DDP** — distributed training requires `nn.Module`
- **`state_dict()` save/load** — currently using 140+ per-tensor `.pt` files
- **HF Trainer / Lightning** — require `nn.Module`
- **`register_forward_hook`** — debugging/profiling requires `nn.Module`

**This is a SPEED ENHANCEMENT, not an architecture change.** It does NOT change the training approach (Gumbel-Softmax + STE + k-means + LoRA). It only changes the container class to unlock framework optimizations.

### Fix

Make `PartialModel` and `PartialWrapper` inherit from `torch.nn.Module`:
- Store `layers` as `nn.ModuleList`
- Add `forward()` method
- Delete hand-rolled `parameters()`, `named_parameters()`, `to()`, `eval()`, `train()`, `get_submodule()`

### Expected Impact

- **Unlocks `torch.compile`**: 1.5-2× speedup on forward+backward (from Agent 2, `07_literature_comparison.md`)
- **Unlocks gradient checkpointing**: batch=128+ (from Agent 5, `03_memory_waste_analysis.md`)
- **Unlocks `state_dict()`**: single-file checkpoints (faster save/load, 30s faster)
- **Unlocks HF Trainer / Lightning**: industry-standard training loops
- **No cos impact** (pure speed enhancement)

### Code Patch (REFERENCE ONLY — not yet applied)

**File:** `scripts/qwen_model.py`
**Lines:** ~438-600 (PartialModel + PartialWrapper classes)

```python
# BEFORE (current — plain Python class):
class PartialModel:
    """Minimal model with embed_tokens + rotary_emb + prefix layers."""
    def __init__(self, embed_tokens, rotary_emb, layers, norm=None, config=None):
        self.embed_tokens = embed_tokens  # NOT nn.Module — no parameter tracking
        self.rotary_emb = rotary_emb
        self.layers = layers  # plain list, NOT nn.ModuleList
        self.norm = norm
        self.config = config

    def to(self, device):
        self.embed_tokens = self.embed_tokens.to(device)
        # ... hand-rolled to() ...

    def parameters(self):
        for p in self.embed_tokens.parameters(): yield p
        for l in self.layers:
            for p in l.parameters(): yield p
        # ... hand-rolled parameters() ...

    # ... 89 lines of hand-rolled reimplementations ...

class PartialWrapper:
    def __init__(self, partial_model, config):
        self.model = partial_model
        self.config = config
    # ... hand-rolled everything ...

# AFTER (proposed — nn.Module subclass):
class PartialModel(nn.Module):
    """Minimal model with embed_tokens + rotary_emb + prefix layers.
    Now an nn.Module — unlocks torch.compile, checkpointing, FSDP."""
    def __init__(self, embed_tokens, rotary_emb, layers, norm=None, config=None):
        super().__init__()
        self.embed_tokens = embed_tokens  # auto-tracked by nn.Module
        self.rotary_emb = rotary_emb
        self.layers = nn.ModuleList(layers)  # ← nn.ModuleList, not plain list
        self.norm = norm
        self.config = config

    def forward(self, input_ids, position_ids=None):
        """Forward method — enables torch.compile + checkpoint."""
        h = self.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        pos_emb = self.rotary_emb(h, position_ids) if self.rotary_emb is not None else None
        for layer in self.layers:
            h = layer(h, position_embeddings=pos_emb) if pos_emb is not None else layer(h)
            if isinstance(h, tuple): h = h[0]
        if self.norm is not None:
            h = self.norm(h)
        return h

    # DELETE: to(), parameters(), named_parameters(), eval(), train(), get_submodule()
    # — all inherited from nn.Module now

class PartialWrapper(nn.Module):
    """Wrapper that's now an nn.Module — unlocks state_dict, compile, etc."""
    def __init__(self, partial_model, config):
        super().__init__()
        self.model = partial_model
        self.config = config

    def forward(self, input_ids, position_ids=None):
        return self.model(input_ids, position_ids)

    # DELETE: to(), eval(), train(), parameters(), named_parameters(), get_submodule()
    # — all inherited from nn.Module now
```

### Migration: Legacy Checkpoint Compatibility

Existing checkpoint (`trained/superblock_0_best/`) uses 140+ per-tensor `.pt` files. After converting to `nn.Module`, use `state_dict()` instead:

**New file:** `scripts/qwen_palettize/checkpoint.py` (migration helper)

```python
def migrate_legacy_checkpoint(wrapper, ckpt_dir):
    """Load 140+ per-tensor .pt files into nn.Module state_dict."""
    state_dict = {}
    for pt_file in os.listdir(ckpt_dir):
        if not pt_file.endswith('.pt'): continue
        # Convert filename: layers_0_input_layernorm_weight.pt → layers.0.input_layernorm.weight
        key = pt_file.replace('.pt', '').replace('_', '.')
        state_dict[key] = torch.load(os.path.join(ckpt_dir, pt_file))
    wrapper.load_state_dict(state_dict, strict=False)
    return wrapper
```

### Verification

After applying:
1. `isinstance(student, nn.Module)` → True
2. `student.parameters()` works without custom implementation
3. `torch.compile(student)` succeeds (no graph breaks)
4. `torch.utils.checkpoint.checkpoint(student.model.layers[0], h)` works
5. `student.state_dict()` returns a single dict (not 140 files)
6. Training runs without NaN (behavior unchanged)

### Dependencies

- None (standalone refactor)
- Migration script needed to load existing checkpoint (above)
- After this, Patches 6+7 benefit (torch.compile can fuse the 25 compute_P_W calls automatically)

### Risks

- **Behavioral change:** `nn.Module.to(dtype)` casts ALL buffers (including int8 indices). Need to verify `indices_int8` stays int8.
  - Mitigation: PyTorch's `_apply` skips non-floating-point buffers automatically. int8 is safe.
- **`state_dict()` format change:** existing 140-file checkpoints incompatible.
  - Mitigation: migration script (above) converts legacy format to state_dict.
- **`torch.compile` compatibility:** custom `autograd.Function` (CUDAFusedLUTLinearSoft) may cause graph breaks.
  - Mitigation: `torch.compile` with `mode="reduce-overhead"` handles custom autograd. Test for graph breaks.

### What This Unlocks (Future)

After Patch 9 is applied:
- **Patch 6 (stream double-buffering):** easier with nn.Module forward
- **Patch 7 (batched compute_P_W):** torch.compile may fuse automatically
- **Gradient checkpointing:** `torch.utils.checkpoint.checkpoint(layer, h, use_reentrant=False)` — saves 5.6GB activations
- **torch.compile:** 1.5-2× speedup on forward+backward
- **HF Trainer / Lightning:** industry-standard training loops (future option)

---

## Summary of Optimizer & Speed Patches

| # | Patch | Expected speedup | VRAM impact | Risk | Unlocks |
|---|-------|-----------------|-------------|------|---------|
| 8 | Fused AdamW (bitsandbytes 8-bit) | 113ms→20ms (93ms saved) | -14GB (21.4→7.1) | Low (8-bit precision) | Larger batch |
| 9 | PartialWrapper → nn.Module | 1.5-2× (via torch.compile) | 0 (cleanup) | Medium (behavioral) | compile, checkpointing, state_dict |
| **Total** | | **~100ms + 1.5-2× compile** | **-14GB** | | |

Both patches are speed enhancements — they do NOT change the training approach (Gumbel-Softmax + STE + k-means + LoRA). They only change the optimizer implementation and container class.
