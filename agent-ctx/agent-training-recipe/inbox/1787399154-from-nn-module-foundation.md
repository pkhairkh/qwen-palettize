# Message: LOCK — train_qwen.py:632-736 for Wave 1

**TO:** training-recipe
**FROM:** nn-module-foundation
**TIMESTAMP:** 2026-08-22T11:45:54Z
**SUBJECT:** LOCK: train_qwen.py:632-736 for Wave 1 (Patch 9 nn.Module refactor)

I am starting Wave 1 of the nn.Module refactor (Patch 9). I will be editing:

  - `scripts/qwen_model.py` — `PartialModel` and `PartialWrapper` classes (currently nested inside `load_qwen_super_block_only`, ~lines 476-571). I will move them to module level and convert them to `nn.Module` subclasses.
  - `scripts/train_qwen.py` — `build_student_super_block()` ONLY, lines 632-736 (Wave 2 will touch this for LoftQ; Wave 1 only touches qwen_model.py for the class refactor, but I am claiming the line range now to avoid collisions).

**Please do NOT edit `scripts/train_qwen.py` lines 632-736** until I send the "RELEASED" message. Your owned ranges (1034-1040, 1140-1160) are unaffected and remain yours.

**What changes for you:**
After I push and the orchestrator merges to main, `PartialWrapper` will be a real `nn.Module`. Consequences:
  - `student.named_parameters()` and `student.named_modules()` will return keys prefixed with `model.` (e.g. `model.layers.0.linear_attn.out_proj.lora_A` instead of `layers.0.linear_attn.out_proj.lora_A`). Any substring matching you do (e.g. `"index_logits" in name`) will still work.
  - `student.train()` / `student.eval()` are inherited — no behaviour change.
  - `state_dict()` / `load_state_dict()` now work natively, but the key naming differs from the legacy per-tensor `.pt` files. Checkpoint migration is a separate concern (not in scope for either of us).

I will send a "RELEASED" message once Wave 1 is pushed. You can then rebase on main.

**ACTION REQUIRED:** wait — do not edit train_qwen.py:632-736 until you receive "RELEASED: nn.Module merged".
