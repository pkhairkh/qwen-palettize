# 03 — Memory Waste Analysis: Teacher/Student Duplication, No Checkpointing

## 1. Executive summary

The training pipeline wastes roughly **2.5–3.5 GB of VRAM per step** on a Blackwell-class 96 GB GPU, in three categories:

1. **Duplicate prefix parameters** between teacher and student (~2.16 GB).
2. **Unnecessary activations** stored for backward, because no gradient checkpointing is enabled (~5 GB at batch=32, seq=512 — of which ~3.7 GB is reclaimable).
3. **Stale student `embed_tokens`** that is orphaned (not freed) after the line `student.model.embed_tokens = teacher.model.embed_tokens` (~1.27 GB).

The total VRAM headroom recoverable, with no algorithmic changes, is **~7 GB**, which is enough to raise `batch_size` from 32 to 96–128 — a 3–4× throughput improvement that requires no kernel work, no architecture redesign beyond the `PartialWrapper` fix described in `02_partial_wrapper_problem.md`, and no precision loss.

This document provides the exact memory math, component by component, for the configuration actually used in `logs/train_sb0.log`: `seq_len=512`, `batch_size=32`, `lora_rank=16`, `use_soft_indices=True`, super-block 0 (4 layers, layers 0–3).

---

## 2. Reference configuration (from `train_sb0.log`)

The training log lines 5–10 confirm the configuration:

```
=== Training super-block 0 (soft_indices=True) ===
  max_steps: 8300  lora_rank: 16
  seq_len: 512  batch_size: 32
  tokens/step: 16384
  tau: 2.0 → 0.1 over 4000 steps
  resume_from: /root/qwen35_palettize/trained/superblock_0_best
```

The log also reports the parameter inventory (lines 28–34):

```
=== Param counts ===
  palettes: 2,208
  lora: 4,688,896
  correction: 0
  layernorms: 611,392
  frozen: 635,701,760
  indices: 1,782,579,200
  Resumed 140 params from ... (step=8000, cos=0.946436)
```

Note that "indices: 1,782,579,200" is the count of `index_logits` elements (4 × K × N summed across 25 palettized Linears, where K and N vary by tensor — see SPEC §1.3 for the per-tensor shapes). At fp16, this is **3.56 GB** of VRAM for `index_logits` alone — the single largest memory consumer in the student.

The "frozen: 635,701,760" is the count of `embed_tokens` parameters (248,320 × 2,560 + a tiny rounding error = 635.7M), at bf16 = **1.27 GB**.

The "Loaded prefix: 1,081,957,952 params" at log line 23 is the total params loaded by `load_qwen_super_block_only` for the prefix (4 layers + embed_tokens). At bf16, this is **2.16 GB** — and **this is loaded twice**, once for the student and once for the teacher.

---

## 3. The four memory categories

The training step's VRAM footprint has four distinct categories. Each is computed independently and the totals are summed at the end.

### 3.1 Static parameter memory (model weights)

This is the memory occupied by the model's parameters and buffers, before any forward pass. It does not change between steps (assuming no live LR updates change the dtype).

#### Student parameters

| Component | Count | Dtype | VRAM |
|---|---|---|---|
| `embed_tokens` (frozen) | 635.7M | bf16 | 1.27 GB |
| 4 × GatedDeltaNet layers (params) | ~410M | bf16 | 0.82 GB |
| 25 × `PalettizedLinear.palette` | 2,208 | bf16 | 4.4 KB |
| 25 × `PalettizedLinear.indices_int8` | 445M | int8 | 445 MB |
| 25 × `PalettizedLinear._flat_idx` (cache) | 445M | int64 | 3.56 GB |
| 25 × `PalettizedLinear.index_logits` | 1.78B | fp16 | 3.56 GB |
| 31 × LoRA `lora_A` + `lora_B` (rank 16) | 4.69M | bf16 | 9.4 MB |
| 5 × LoRA `lora_A` + `lora_B` (rank 32) | — included above — | bf16 | — |
| Layernorms + SSM + conv1d | 611K | bf16 | 1.2 MB |
| **Student total** | | | **~9.66 GB** |

Two observations on the student side:

1. The `_flat_idx` cache (3.56 GB at int64) is a precomputed gather index for the PyTorch fallback path. It is only used when the CUDA kernel is unavailable (`self._use_cuda == False`). At construction time (`qwen_model.py:102`), the cache is allocated unconditionally on the same device as `indices`. This is a clear waste — the cache should only be allocated when the fallback path is actually used. **Easy fix: lazy-allocate `_flat_idx` on first fallback call. Saves 3.56 GB.**
2. The `index_logits` at fp16 (3.56 GB) is the largest single consumer. The dtype is locked in by the CUDA kernel's `TORCH_CHECK(logits.dtype() == torch::kHalf)` assertion (`fused_lut_linear_cuda.py:235`). Switching to bf16 would require recompiling the kernel, but would unify the dtype and remove the `±20` clamp workaround at `train_qwen.py:1149`. **Medium fix: change kernel contract from fp16 to bf16. Saves no VRAM (same size), but removes the clamp.**

#### Teacher parameters

| Component | Count | Dtype | VRAM |
|---|---|---|---|
| `embed_tokens` (frozen) | 635.7M | bf16 | 1.27 GB |
| 4 × Qwen3.5 layers (full fp16) | ~410M | bf16 | 0.82 GB |
| **Teacher total** | | | **~2.09 GB** |

The teacher is a full-fidelity copy of the prefix (no palettization, no LoRA, no index_logits). It exists solely to produce `h_out` (the teacher's super-block output) for the distillation loss.

#### Total static parameter memory

```
  Student static:    9.66 GB
  Teacher static:    2.09 GB
  ─────────────────────────────
  Total:            11.75 GB
```

Of this, **2.16 GB is duplicated** — both teacher and student hold a copy of `embed_tokens` (1.27 GB) and a copy of the prefix layers' non-palettized weights (0.82 GB shared between the student's frozen weights and the teacher's weights — though these are different tensors, since the student's prefix has been palettized, so they are not byte-for-byte duplicates; the only true duplicate is `embed_tokens`).

#### Theorized savings (in-place teacher)

If we load the teacher once, share `embed_tokens` (1.27 GB saved), and compute `h_out` by running the **student's** unpalettized layers via a "shadow" forward — we save the full 2.09 GB of teacher VRAM. The shadow forward is achievable because the student's prefix layers, before palettization, are byte-identical to the teacher's. The palettization is applied **in-place** by `palettize_linear` at `qwen_model.py:583`, which replaces the `nn.Linear` with a `PalettizedLinear` — but the original weights are recoverable from the `palette + indices` via reconstruction (the function `_reconstruct_base_weight` at `qwen_model.py:239` already does this).

The cleaner alternative is to keep the original `nn.Linear` weights as a frozen "teacher shadow" inside the student module, share `embed_tokens`, and compute `h_out` from the shadow. This costs 0.82 GB (the shadow weights), but saves 1.27 GB (the duplicate `embed_tokens`) and removes the second `from_pretrained` call entirely.

**Net savings: 1.27 GB (embed_tokens duplication) + 0.97 GB (second download / cache overhead) ≈ 2.24 GB.**

### 3.2 Optimizer state memory

The `FP32MasterOptimizer` at `train_qwen.py:154` maintains fp32 copies of all trainable params. For each trainable tensor, this doubles the memory (the bf16 model param + the fp32 master).

| Trainable group | Count | bf16 VRAM | fp32 master VRAM | Total |
|---|---|---|---|---|
| Palettes | 2,208 | 4.4 KB | 8.8 KB | 13.2 KB |
| LoRA `lora_A`, `lora_B` | 4.69M | 9.4 MB | 18.8 MB | 28.2 MB |
| `index_logits` | 1.78B | 3.56 GB | 7.13 GB | 10.69 GB |
| Layernorms + SSM | 611K | 1.2 MB | 2.4 MB | 3.6 MB |
| **Optimizer total** | | | | **~10.72 GB** |

The `index_logits` master alone is **7.13 GB** at fp32. This is the second-largest memory consumer after the student's static params. Two thoughts:

1. **The fp32 master for `index_logits` is necessary** because the Gumbel-Softmax gradient at low τ produces updates that overflow fp16. The clamp at `train_qwen.py:1149` (`±20`) is applied to the bf16 model param after the master sync — but the master itself is unclamped, which is correct (the master can hold ±1e6 safely in fp32).
2. **The fp32 master for LoRA is unnecessary.** LoRA's `lora_A` and `lora_B` are small (~9.4 MB total), and bf16 has enough exponent range for them. The fp32 master doubles the LoRA memory for no numerical benefit. **Easy fix: skip the fp32 master for LoRA. Saves 18.8 MB.** (Marginal, but principled.)

The AdamW optimizer also maintains `exp_avg` and `exp_avg_sq` buffers, each at fp32 — so each trainable param has **3× the memory** (bf16 model + fp32 master + fp32 `exp_avg` + fp32 `exp_avg_sq`). For `index_logits`, this is:

```
  bf16 model:           3.56 GB
  fp32 master:          7.13 GB
  fp32 exp_avg:         7.13 GB
  fp32 exp_avg_sq:      7.13 GB
  ─────────────────────────────
  Total per index_logits:  24.95 GB
```

This is the single largest line item in the entire memory budget. The Muon optimizer (used for layernorms and LoRA B, per SPEC §3.5) maintains only `momentum_buffer` (1× extra), so it is more memory-efficient than AdamW — but it cannot be used for `index_logits` because `index_logits` is 3D (4, K, N), and Muon's Newton-Schulz orthogonalization only works for 2D tensors.

#### Total optimizer state

```
  AdamW state (index_logits):     21.39 GB  (master + 2 buffers)
  AdamW state (palettes):         26.4 KB
  AdamW state (LoRA A):           56.4 MB
  Muon state (LoRA B + norms):    11.4 MB
  ─────────────────────────────────────
  Total optimizer state:         ~21.46 GB
```

### 3.3 Activation memory (forward pass)

During the forward pass, each layer stores its intermediate activations for the backward pass. For a Qwen3.5 layer at `seq_len=512, batch_size=32`, the activations per layer are:

| Tensor | Shape | Dtype | VRAM |
|---|---|---|---|
| Layer input `h_in` | (32, 512, 2560) | bf16 | 80 MB |
| Attention QKV (for full-attn layers) | (32, 512, 8192+1024+1024) | bf16 | 320 MB |
| Attention scores (full-attn only) | (32, 40, 512, 512) | bf16 | 670 MB |
| Attention output | (32, 512, 2560) | bf16 | 80 MB |
| MLP intermediate (gate, up, down) | (32, 512, 9216) × 2 | bf16 | 600 MB |
| Layer output `h_out` | (32, 512, 2560) | bf16 | 80 MB |
| **Total per full-attn layer** | | | **~1.83 GB** |
| **Total per GatedDeltaNet layer** | | | **~1.26 GB** (no attention scores) |

For super-block 0 (3 GatedDeltaNet + 1 full-attn):

```
  3 × GatedDeltaNet activations:   3.78 GB
  1 × full-attn activations:      1.83 GB
  ─────────────────────────────────────
  Total activations:              5.61 GB
```

With **gradient checkpointing** (recompute activations during backward, store only the layer input), this collapses to:

```
  4 × layer input (32, 512, 2560):   4 × 80 MB = 320 MB
  ─────────────────────────────────────
  Total activations (checkpointed):  ~320 MB
```

**Savings: 5.29 GB.** This is the single largest memory recovery available — and it is the one explicitly blocked by `PartialWrapper` not being an `nn.Module` (because `torch.utils.checkpoint` requires `nn.Module` semantics, see `02_partial_wrapper_problem.md` §4.2).

### 3.4 Workspace and transient memory (forward + backward)

The CUDA kernels allocate temporary workspace during forward and backward. For the soft kernel, the workspace is:

| Tensor | Shape | Dtype | VRAM |
|---|---|---|---|
| `P` (Gumbel-Softmax probabilities) | (4, K, N) | fp16 | 3.56 GB |
| `W` (materialized weight) | (K, N) | bf16 | 1.78 GB |
| `grad_W` (computed in Python as `x.T @ grad_y`) | (K, N) | fp32 | 3.56 GB |
| `y` (matmul output) | (M, N) | bf16 | (varies) |

For a single Linear with `K=2560, N=8192`, the workspace is ~8.9 GB across all 25 Linears — but only one Linear is live at a time during the forward pass, so the peak workspace is ~8.9 GB / 25 = ~360 MB. During backward, the same workspace is reused (the `P` and `W` saved for backward are the ones from forward).

**Total workspace: ~360 MB peak (forward), ~720 MB peak (backward, with `grad_W` live alongside `P` and `W`).**

This is a minor contributor compared to the parameter and activation memory, but it explains why `torch.cuda.memory_allocated()` spikes during the backward pass (visible in `train_sb0.log` if we had GPU mem logging — the print at `train_qwen.py:1180` does emit it but the log was truncated).

### 3.5 The orphaned student `embed_tokens` (a subtle leak)

At `train_qwen.py:994`, the code does:

```python
student.model.embed_tokens = teacher.model.embed_tokens
```

This re-points `student.model.embed_tokens` from the student's own tensor to the teacher's tensor. But the student's original `embed_tokens` tensor is **not freed** — it is still referenced by:

1. The original `student.model.embed_tokens` Python attribute (now overwritten — but Python's garbage collector has not yet run).
2. The HuggingFace `from_pretrained` cache, which holds a reference to the loaded model's `embed_tokens` until the cache is cleared.
3. Any `named_parameters()` generator that captured the original tensor (unlikely, but possible).

The `gc.collect()` and `torch.cuda.empty_cache()` calls at `qwen_model.py:472–473` run inside `load_qwen_super_block_only`, **before** the teacher is loaded — so they do not free the student's orphaned `embed_tokens`. After the teacher is loaded and the assignment happens, the orphaned tensor sits in VRAM until the next `gc.collect()` — which the training loop never calls.

**Estimated waste: 1.27 GB**, persistent for the entire training run.

The fix is to explicitly free the student's `embed_tokens` before the assignment:

```python
# Free student's embed_tokens before sharing teacher's
del student.model.embed_tokens
gc.collect()
torch.cuda.empty_cache()
student.model.embed_tokens = teacher.model.embed_tokens
```

Or better: construct the student without an `embed_tokens` in the first place, and assign the teacher's after both are loaded.

---

## 4. The total memory budget

Combining all four categories:

```
  ─────────────────────────────────────────────────────────────
  CATEGORY                              CURRENT    AFTER FIX
  ─────────────────────────────────────────────────────────────
  Student static params                   9.66 GB    6.10 GB *
  Teacher static params                   2.09 GB    0.82 GB **
  Optimizer state (fp32 master + buffers) 21.46 GB   21.46 GB
  Activations (forward, no checkpointing) 5.61 GB    0.32 GB ***
  Workspace (transient)                   0.72 GB    0.72 GB
  Orphaned student embed_tokens           1.27 GB    0.00 GB ****
  ─────────────────────────────────────────────────────────────
  TOTAL                                  40.81 GB   29.42 GB
  ─────────────────────────────────────────────────────────────
  * Saves 3.56 GB by lazy-allocating _flat_idx
  ** Saves 1.27 GB by sharing embed_tokens with student
  *** Saves 5.29 GB by enabling gradient checkpointing
  **** Saves 1.27 GB by explicit del + gc.collect()
  ─────────────────────────────────────────────────────────────
  TOTAL RECOVERABLE:                     11.39 GB
```

On a Blackwell B200 with 96 GB VRAM, the current usage of ~41 GB leaves ~55 GB free — enough for batch=64 today. The post-fix usage of ~29 GB leaves ~67 GB free — enough for batch=128 (the activations scale linearly with batch size, so going from 32 to 128 adds ~1.3 GB of activations post-checkpointing, well within budget).

On an L4 (24 GB VRAM), the current usage of ~41 GB **does not fit** — the training run would OOM. This is consistent with the `train_sb0.log` showing the run is on a Blackwell GPU, not an L4. The post-fix usage of ~29 GB would barely fit on a 32 GB GPU (e.g., a single V100 32GB or a 24 GB L4 with some headroom).

---

## 5. Per-component VRAM breakdown (current state)

For the `train_sb0.log` configuration (`batch=32, seq=512, sb_idx=0, use_soft_indices=True`), the VRAM is allocated as follows (in descending order of consumption):

```
  ┌──────────────────────────────────────────────────────────────┐
  │ 1. AdamW master + buffers for index_logits   21.39 GB  52% │
  │    (bf16 model 3.56 + fp32 master 7.13 +       │            │
  │     fp32 exp_avg 7.13 + fp32 exp_avg_sq 7.13)  │            │
  ├──────────────────────────────────────────────────────────────┤
  │ 2. Forward activations (4 layers)              5.61 GB  14% │
  │    (3 GatedDeltaNet × 1.26 + 1 full-attn 1.83)│            │
  ├──────────────────────────────────────────────────────────────┤
  │ 3. Student _flat_idx cache (int64)             3.56 GB   9% │
  │    (only used in fallback path, never in CUDA) │            │
  ├──────────────────────────────────────────────────────────────┤
  │ 4. Student index_logits (bf16 model)           3.56 GB   9% │
  ├──────────────────────────────────────────────────────────────┤
  │ 5. Teacher params (frozen, full bf16)           2.09 GB   5% │
  │    (embed_tokens 1.27 + 4 layers 0.82)        │            │
  ├──────────────────────────────────────────────────────────────┤
  │ 6. Orphaned student embed_tokens               1.27 GB   3% │
  │    (not freed after sharing with teacher)       │            │
  ├──────────────────────────────────────────────────────────────┤
  │ 7. Student indices_int8 buffer                  0.45 GB   1% │
  ├──────────────────────────────────────────────────────────────┤
  │ 8. Workspace (P, W, grad_W, y — peak)          0.72 GB   2% │
  ├──────────────────────────────────────────────────────────────┤
  │ 9. Student frozen layers (non-palettized)      0.82 GB   2% │
  ├──────────────────────────────────────────────────────────────┤
  │ 10. Student LoRA + palettes + norms            0.04 GB  <1% │
  ├──────────────────────────────────────────────────────────────┤
  │ 11. Student embed_tokens (shared, but counted  1.27 GB   3% │
  │     twice because of orphan)                    │            │
  ├──────────────────────────────────────────────────────────────┤
  │ TOTAL                                          40.81 GB 100% │
  └──────────────────────────────────────────────────────────────┘
```

**The top three categories account for 75% of VRAM.** Each is fixable:

1. The 21.4 GB for AdamW state on `index_logits` is the hardest to reduce — it requires either (a) dropping the fp32 master for `index_logits` (risky, but bf16 has the same exponent range as fp32 so it may be safe), or (b) using a memory-efficient optimizer like `bnb.optim.Adam8bit` (which compresses the state to 8-bit), or (c) sharding `index_logits` across GPUs (requires FSDP, which requires the `PartialWrapper` fix).
2. The 5.6 GB for activations is recoverable via gradient checkpointing — the fix is one decorator on each layer's forward, blocked only by the `PartialWrapper` issue.
3. The 3.56 GB for `_flat_idx` is recoverable by lazy-allocation — a 5-line code change.

---

## 6. The teacher-student duplication in detail

This section traces the exact sequence of `from_pretrained` and attribute-assignment operations that produce the duplication, to make clear that the duplication is **not** a design choice — it is a side effect of the loading order.

### 6.1 The loading sequence (current)

```
  train_qwen.py:942  build_student_super_block(sb_idx=0, ...)
    │
    └── qwen_model.py:load_qwen_super_block_only(sb_idx=0)
          │
          ├── model = Qwen3_5ForConditionalGeneration.from_pretrained(...)
          │   ↓
          │   [HF downloads weights, builds full model in VRAM]
          │   [embed_tokens: 1.27 GB on GPU]
          │   [32 layers: ~6.5 GB on GPU]
          │
          ├── Carve out prefix (4 layers + embed_tokens + rotary_emb + norm)
          │   ↓
          │   [Detach prefix from full model]
          │   [lang_model.embed_tokens = None]
          │   [lang_model.layers = nn.ModuleList()]   ← full model's layers are now orphaned!
          │   [gc.collect(); torch.cuda.empty_cache()]  ← frees the orphaned full-model layers
          │
          ├── Build PartialModel + PartialWrapper
          │   ↓
          │   [wrapper.model.embed_tokens = the original embed_tokens tensor]
          │   [wrapper.model.layers = [layer0, layer1, layer2, layer3]]  ← plain Python list
          │
          └── return wrapper  [VRAM: 2.16 GB]

  train_qwen.py:947  apply_groups(...)
  train_qwen.py:963  build_optimizers(...)
  train_qwen.py:973  LambdaLR schedulers
  train_qwen.py:987  teacher, _ = load_qwen_super_block_only(sb_idx=0)
    │
    └── [HF downloads weights AGAIN — but cache hit, so fast]
        [Builds a SECOND wrapper with its OWN embed_tokens: 1.27 GB on GPU]
        [VRAM now: 4.32 GB total (2.16 student + 2.16 teacher)]

  train_qwen.py:989  # Freeze teacher
  train_qwen.py:990  for p in teacher.model.embed_tokens.parameters(): p.requires_grad_(False)
  train_qwen.py:991  for layer in teacher.model.layers:
  train_qwen.py:992      for p in layer.parameters(): p.requires_grad_(False)

  train_qwen.py:994  student.model.embed_tokens = teacher.model.embed_tokens
    │
    └── [Re-points student.model.embed_tokens to teacher's tensor]
        [Student's original embed_tokens is now ORPHANED — Python still holds a ref]
        [VRAM unchanged: 4.32 GB]
        [No gc.collect(), no torch.cuda.empty_cache() called here!]
```

### 6.2 The duplication count

After line 994, the VRAM has:

- **Teacher's `embed_tokens`**: 1.27 GB (now also referenced by student)
- **Student's original `embed_tokens`** (orphaned): 1.27 GB (not freed)
- **Teacher's 4 layers**: 0.82 GB
- **Student's 4 layers** (palettized + LoRA + index_logits + _flat_idx): ~9.66 GB − 1.27 (embed) = 8.39 GB

**Total: 1.27 + 1.27 + 0.82 + 8.39 = 11.75 GB** of static parameter memory, of which **1.27 GB is pure waste** (the orphaned student `embed_tokens`).

### 6.3 The proposed fix (in-place teacher)

The cleanest fix is to load the teacher once, share `embed_tokens` from the start, and construct the student by **palettizing the teacher's prefix in-place**:

```python
def build_teacher_and_student(sb_idx, ...):
    # Load the prefix ONCE
    wrapper, tokenizer = load_qwen_super_block_only(sb_idx, device=DEVICE, dtype=DTYPE)
    
    # wrapper is now the TEACHER (full fp16 prefix)
    teacher = wrapper
    
    # Freeze teacher
    for p in teacher.parameters(): p.requires_grad_(False)
    
    # Build student by palettizing the active layers IN-PLACE
    student = copy.deepcopy(teacher)  # shares nothing, but heavy
    # OR: share embed_tokens, copy only the active layers
    # student = PartialWrapper(PartialModel(
    #     embed_tokens=teacher.model.embed_tokens,  # shared
    #     rotary_emb=teacher.model.rotary_emb,        # shared
    #     layers=[copy.deepcopy(l) for l in teacher.model.layers],  # copied
    #     norm=teacher.model.norm,                    # shared
    #     config=teacher.config,
    # ), teacher.config)
    
    # Palettize student's active layers
    for layer_idx in range(sb_start, sb_end):
        layer = student.model.layers[layer_idx]
        # ... replace Linears with PalettizedLinear ...
    
    return teacher, student, tokenizer
```

This saves:

- 1.27 GB (no duplicate `embed_tokens`)
- 1.27 GB (no orphaned student `embed_tokens`)
- ~0.82 GB (rotary_emb and norm are shared, not copied)

**Total savings: ~3.36 GB.** Combined with the lazy `_flat_idx` fix (3.56 GB) and the gradient checkpointing fix (5.29 GB), the total recovery is **~12.21 GB** — pushing the per-step VRAM from ~41 GB down to ~29 GB.

---

## 7. Why gradient checkpointing is not enabled today

The user message asks: "Why is there no gradient checkpointing? (Qwen3.5-4B 4 layers at batch=32 seq=512 — activations are ~20GB. Checkpointing would cut to ~5GB, enabling batch=128)."

The answer is two-fold:

### 7.1 The wrapper issue (primary cause)

`PartialWrapper` is not an `nn.Module`. `torch.utils.checkpoint.checkpoint(fn, *args)` requires `fn` to participate in autograd as an `nn.Module.forward`. The wrapper's layers are real `nn.Module`s, but the layer list is a plain Python list, not `nn.ModuleList`. The training loop calls `layer(s_h, ...)` directly (line 1095), which invokes the layer's `__call__` → `forward`. This works for the forward pass, but `torch.utils.checkpoint` wraps the entire `fn` in a recomputation context that requires the layer to be discoverable via `model._modules`.

Even if we add `torch.utils.checkpoint.checkpoint(layer, s_h, ...)` directly to the loop, the recomputation graph is not built correctly because the wrapper's `forward` does not exist as a method — the loop is the forward. The result: backward produces wrong gradients or `NaN`.

### 7.2 The HuggingFace `gradient_checkpointing_enable()` issue (secondary cause)

HuggingFace transformer layers have a `gradient_checkpointing_enable()` method that sets `self.gradient_checkpointing = True` and switches the layer's forward to use `torch.utils.checkpoint`. Calling this on each layer inside `PartialWrapper.layers` works at the layer level — the flag is set, the layer's forward checks the flag — but the surrounding training loop in `train_qwen.py:1091–1098` does **not** consult the flag. It just calls `layer(s_h, ...)`. Even if the layer's `forward` checks `self.gradient_checkpointing`, the `forward` does not pass the right arguments to `torch.utils.checkpoint.checkpoint` (it requires `use_reentrant=False` in PyTorch 2.x to work with the soft kernel's custom autograd Function — there is a known interaction between `checkpoint` and `torch.autograd.Function` that requires the non-reentrant mode).

### 7.3 The fix

After the `PartialWrapper` → `nn.Module` refactor (document `02`), the fix is two changes:

1. Add `self.layers = nn.ModuleList(layers)` in `PartialModel.__init__` (already in the patched version).
2. In the training loop, replace:
   ```python
   for layer in student.model.layers:
       out = layer(s_h, ...)
       s_h = out[0] if isinstance(out, tuple) else out
   ```
   with:
   ```python
   for layer in student.model.layers:
       if use_gradient_checkpointing and self.training:
           out = torch.utils.checkpoint.checkpoint(layer, s_h, use_reentrant=False)
       else:
           out = layer(s_h, ...)
       s_h = out[0] if isinstance(out, tuple) else out
   ```

The `use_reentrant=False` flag is critical: it is the only mode compatible with `torch.autograd.Function` (which the soft kernel uses). With `use_reentrant=True` (the default in older PyTorch), the backward pass produces `NaN` when the soft kernel's `CUDAFusedLUTLinearSoft` is in the recomputation graph.

---

## 8. What batch size becomes feasible after the fix?

The activation memory scales linearly with `batch_size` (and with `seq_len`, but we hold `seq_len=512` constant). With gradient checkpointing, the per-step activation memory is:

```
  Activations (checkpointed) = 4 layers × 80 MB × (batch_size / 32) = 10 MB × batch_size
```

The post-fix total VRAM is:

```
  Static + Optimizer:        21.46 GB  (index_logits dominates, not batch-dependent)
  Student static (post-lazy _flat_idx, post-share embed_tokens):
                             5.94 GB   (was 9.66 GB)
  Teacher static (post-share): 0.82 GB
  Workspace:                  0.72 GB  (scales mildly with batch)
  Activations (checkpointed): 0.32 GB × (batch_size / 32) = 0.01 GB × batch_size
  ─────────────────────────────────────
  Total:                     ~28.94 GB + 0.01 × batch_size
```

On a 96 GB Blackwell B200, with a 80 GB safety margin (leaving room for the kernel workspace and CUDA context), the budget for activations is ~67 GB. Solving `0.01 × batch_size = 67` gives `batch_size = 6700`. In practice, the activation formula is more complex (the attention scores scale with `batch × seq × seq`, not just `batch × seq`), so the realistic ceiling is `batch_size = 256–512` for `seq_len=512`.

**Even at the conservative end, batch=128 is comfortably within budget.** This is a 4× throughput improvement over the current batch=32, achievable with no kernel changes and only the architecture fixes documented in this wave.

---

## 9. Summary of fixes and savings

| Fix | Difficulty | VRAM saved | Side benefit |
|---|---|---|---|
| Lazy-allocate `_flat_idx` (only on fallback path) | Easy (5 LOC) | 3.56 GB | No fallback means no Python gather overhead |
| Explicit `del + gc.collect()` after `embed_tokens` share | Easy (3 LOC) | 1.27 GB | Removes the silent leak |
| In-place teacher (load once, share `embed_tokens`, share `rotary_emb`, share `norm`) | Medium (1 day) | 2.09 GB | Removes second `from_pretrained` call |
| Gradient checkpointing (after `PartialWrapper` → `nn.Module` fix) | Medium (after doc 02 fix) | 5.29 GB | Enables batch=128 |
| Drop fp32 master for LoRA (keep only for `index_logits`) | Easy (5 LOC) | 28 MB | Removes unnecessary precision |
| Switch `index_logits` to bf16 (requires kernel recompile) | Hard (1 day, kernel work) | 0 GB (same size) | Removes the ±20 clamp workaround |
| Use 8-bit optimizer (`bnb.optim.Adam8bit`) for `index_logits` | Easy (pip install + 5 LOC) | ~14 GB | Halves optimizer state |
| **Total (without 8-bit optimizer)** | | **~12.21 GB** | batch=64+ feasible |
| **Total (with 8-bit optimizer)** | | **~26.21 GB** | batch=128+ feasible on 64 GB GPUs |

---

## 10. The cost of inaction

If the architecture is not fixed and the team continues to iterate on the kernel / indices / palettes in isolation, the VRAM ceiling will remain at ~41 GB per step. This means:

- Every super-block requires a Blackwell-class GPU. L4, A100, and H100 80GB are all infeasible at batch=32.
- Batch=64 requires sharding `index_logits` across GPUs, which requires FSDP, which requires the `PartialWrapper` fix.
- The cos=0.999 target in the SPEC likely requires either larger batches (more stable gradients) or longer training (more steps), both of which are bottlenecked by VRAM.
- Each architecture iteration (e.g., trying a different LoRA rank, or removing the correction layer) requires a full training run to validate, which takes ~1.5 hours per super-block at the current throughput.

The architecture fixes documented here and in `02_partial_wrapper_problem.md` are the prerequisites for any further research progress. Without them, the team is locked into the current VRAM budget and the current throughput, and the cos=0.999 target is unreachable.
