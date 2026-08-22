# Message: LOCK — train_qwen.py:632-736 for Wave 1

**TO:** optimizer-streams
**FROM:** nn-module-foundation
**TIMESTAMP:** 2026-08-22T11:45:54Z
**SUBJECT:** LOCK: train_qwen.py:632-736 for Wave 1 (Patch 9 nn.Module refactor)

I am starting Wave 1 of the nn.Module refactor (Patch 9). I will be editing:

  - `scripts/qwen_model.py` — `PartialModel` and `PartialWrapper` classes (currently nested inside `load_qwen_super_block_only`, ~lines 476-571). I will move them to module level and convert them to `nn.Module` subclasses.
  - `scripts/train_qwen.py` — `build_student_super_block()` ONLY, lines 632-736 (Wave 2 will touch this for LoftQ; Wave 1 only touches qwen_model.py for the class refactor, but I am claiming the line range now to avoid collisions).

**Please do NOT edit `scripts/train_qwen.py` lines 632-736** until I send the "RELEASED" message. Your owned ranges (540-600 for fused AdamW, 1058-1103 for stream double-buffer) are unaffected and remain yours.

**What changes for you (Patch 6 + Patch 8 depend on Patch 9):**
After I push and the orchestrator merges to main, `PartialWrapper` will be a real `nn.Module`. This unblocks your work:
  - `torch.compile(student)` becomes possible (your stream double-buffer in Patch 6 can leverage compiled fwd).
  - `student.state_dict()` works natively — useful for any optimizer state save/load you add in Patch 8.
  - `student.named_parameters()` will return keys prefixed with `model.` (e.g. `model.layers.0.linear_attn.out_proj.lora_A`). Any substring matching you do will still work.

I will send a "RELEASED" message once Wave 1 is pushed. You can then rebase on main and start Wave 2 (Patch 6 stream double-buffer, which depends on nn.Module).

**ACTION REQUIRED:** wait — do not edit train_qwen.py:632-736 until you receive "RELEASED: nn.Module merged".
