# TASKS — agent-nn-module-foundation

> **Branch:** `agent/nn-module-foundation`
> **Patches:** 9 (PartialWrapper→nn.Module), 2 (LoftQ SVD init for LoRA)
> **You are the FOUNDATION agent.** Other agents depend on your Patch 9 merging first.

---

## Wave 1: Patch 9 — PartialWrapper → nn.Module

**Research:** [`research-architecture-review/02_partial_wrapper_problem.md`](../../research-architecture-review/02_partial_wrapper_problem.md)
**Paper:** `docs/papers/2305.14314_QLoRA_Dettmers2023.pdf` (QLoRA pattern: nn.Module + LoRA)
**Files:** `scripts/qwen_model.py` (~lines 438-600), `scripts/train_qwen.py` (~lines 632-736 if affected)

### Sub-task 1a: Convert PartialModel to nn.Module

**File:** `scripts/qwen_model.py`, class `PartialModel` (~line 438)

**Change:**
- Add `class PartialModel(nn.Module):` (inherit from nn.Module)
- In `__init__`, call `super().__init__()`
- Change `self.layers = layers` (plain list) → `self.layers = nn.ModuleList(layers)`
- Delete hand-rolled: `to()`, `parameters()`, `named_parameters()`, `eval()`, `train()`, `named_modules()`, `get_submodule()` — all inherited from nn.Module now
- Add `forward(self, input_ids, position_ids=None)` method:
  ```python
  def forward(self, input_ids, position_ids=None):
      h = self.embed_tokens(input_ids)
      if position_ids is None:
          position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
      pos_emb = self.rotary_emb(h, position_ids) if self.rotary_emb is not None else None
      for layer in self.layers:
          out = layer(h, position_embeddings=pos_emb) if pos_emb is not None else layer(h)
          h = out[0] if isinstance(out, tuple) else out
      if self.norm is not None:
          h = self.norm(h)
      return h
  ```

**Test:** `python3 -c "import ast; ast.parse(open('scripts/qwen_model.py').read()); print('OK')"`
**Commit:** `Patch 9a: PartialModel → nn.Module (inherit, ModuleList, forward)`

### Sub-task 1b: Convert PartialWrapper to nn.Module

**File:** `scripts/qwen_model.py`, class `PartialWrapper` (~line 540)

**Change:**
- Add `class PartialWrapper(nn.Module):`
- In `__init__`, call `super().__init__()`
- `self.model = partial_model` (now auto-tracked by nn.Module since PartialModel is nn.Module)
- Delete hand-rolled: `to()`, `eval()`, `train()`, `parameters()`, `named_parameters()`, `named_modules()`, `get_submodule()`
- Add `forward(self, input_ids, position_ids=None): return self.model(input_ids, position_ids)`

**Test:** syntax check
**Commit:** `Patch 9b: PartialWrapper → nn.Module (forward added, hand-rolled methods deleted)`

### Sub-task 1c: Verify load_qwen_super_block_only still works

**File:** `scripts/qwen_model.py`, function `load_qwen_super_block_only` (~line 434)

The function creates `PartialModel(...)` then `PartialWrapper(partial, config)`. Since we changed the constructors, verify:
- `wrapper = wrapper.to(device).eval()` — nn.Module.to() works with device arg
- `wrapper.train()` / `wrapper.eval()` — inherited
- `wrapper.model.embed_tokens.parameters()` — still yields parameters

**Test:** syntax check + verify isinstance check would pass:
```python
import torch.nn as nn
from qwen_model import PartialWrapper
# After load: assert isinstance(wrapper, nn.Module)
```

**Commit:** `Patch 9c: verify load_qwen_super_block_only compatibility`
**Push:** `git push origin agent/nn-module-foundation`

### Sub-task 1d: Send inbox messages to dependent agents

After pushing Wave 1:
- Message training-recipe: "RELEASED: nn.Module merged — rebase on main"
- Message optimizer-streams: "RELEASED: nn.Module merged — rebase on main"
- Update `agent-ctx/PROGRESS.md`: mark Patch 9 as ✅ Done

**DoD for Wave 1:**
- [ ] PartialModel inherits nn.Module, has forward()
- [ ] PartialWrapper inherits nn.Module, has forward()
- [ ] All hand-rolled methods deleted
- [ ] syntax check passes
- [ ] Branch pushed
- [ ] Inbox messages sent
- [ ] PROGRESS.md updated

---

## Wave 2: Patch 2 — LoftQ SVD Initialization for LoRA

**Research:** [`research-kernel-accuracy/00_overview.md`](../../research-kernel-accuracy/00_overview.md) §Fix 1, [`research-filter-consolidation/01_training_recipe.md`](../../research-filter-consolidation/01_training_recipe.md) §Patch 2
**Paper:** `docs/papers/2305.14314_QLoRA_Dettmers2023.pdf` (LoftQ: LoRA initialized from SVD of quantization error)
**Files:** `scripts/qwen_model.py` (new helper), `scripts/train_qwen.py` (~lines 696-725, build_student_super_block)

**Prerequisite:** Wait for "RELEASED" message from yourself (Wave 1 complete). Pull main to get nn.Module changes if orchestrator has merged.

### Sub-task 2a: Add capture_original_weights_from_checkpoint helper

**File:** `scripts/qwen_model.py` (new function, near `load_qwen_super_block_only`)

```python
def capture_original_weights_from_checkpoint(sb_idx, model_name="Qwen/Qwen3.5-4B"):
    """Load original fp16 weights from HF checkpoint before palettization.
    Returns dict {tensor_name: weight_tensor}.
    Used for LoftQ SVD initialization of LoRA."""
    from transformers import AutoModelForCausalLM
    import gc
    sb_start, sb_end = SUPER_BLOCKS[sb_idx]
    full_model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    weights = {}
    for layer_idx in range(sb_start, sb_end):
        layer = full_model.model.layers[layer_idx]
        for name, param in layer.named_parameters():
            if name.endswith(".weight"):
                weights[f"model.layers.{layer_idx}.{name}"] = param.data.clone()
    del full_model
    gc.collect()
    torch.cuda.empty_cache()
    return weights
```

**Test:** syntax check
**Commit:** `Patch 2a: add capture_original_weights_from_checkpoint helper`

### Sub-task 2b: Pass original weights to QwenLoRA in build_student_super_block

**File:** `scripts/train_qwen.py`, function `build_student_super_block` (~line 696)

**Change:** Before the LoRA attach loop, call `capture_original_weights_from_checkpoint(sb_idx)`. Then pass `original_weight=orig_w` to `QwenLoRA(...)` instead of `original_weight=None`.

**Before:**
```python
lora_mod = QwenLoRA(module, rank=rank, alpha=alpha, init="loftq", original_weight=None)
```

**After:**
```python
from qwen_model import capture_original_weights_from_checkpoint
original_weights = capture_original_weights_from_checkpoint(sb_idx)
# ... in the loop:
full_name = f"model.layers.{layer_idx}.{name}.weight"
orig_w = original_weights.get(full_name)
lora_mod = QwenLoRA(module, rank=rank, alpha=alpha, init="loftq", original_weight=orig_w)
```

**Test:** syntax check. Verify `QwenLoRA.__init__` with `init="loftq"` and non-None `original_weight` takes the SVD branch (lines 209-222 of qwen_model.py).

**Commit:** `Patch 2b: pass original weights to QwenLoRA for LoftQ SVD init`
**Push:** `git push origin agent/nn-module-foundation`

**DoD for Wave 2:**
- [ ] `capture_original_weights_from_checkpoint()` function added
- [ ] `build_student_super_block()` calls it and passes to QwenLoRA
- [ ] syntax check passes on both files
- [ ] Branch pushed
- [ ] PROGRESS.md updated (Patch 2 ✅)

---

## Wave 3: Final Verification + Merge Prep

### Sub-task 3a: Verify branch merges cleanly with main

```bash
git fetch origin main
git checkout agent/nn-module-foundation
git merge origin/main  # or rebase
# Resolve any conflicts in train_qwen.py (only lines 632-736)
```

### Sub-task 3b: Send final inbox messages

- Message training-recipe: "Merge ready — my branch is clean for merge"
- Message optimizer-streams: "Merge ready — my branch is clean for merge"
- Update PROGRESS.md: mark all patches ✅, all waves ✅

**DoD for Wave 3:**
- [ ] Branch merges cleanly with main (no conflicts, or conflicts resolved)
- [ ] Final inbox messages sent
- [ ] PROGRESS.md fully updated
- [ ] Branch pushed
