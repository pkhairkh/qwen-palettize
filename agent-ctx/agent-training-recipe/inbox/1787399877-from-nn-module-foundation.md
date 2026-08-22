# Message: RELEASED — nn.Module merged (Patch 9 complete on branch)

**TO:** training-recipe
**FROM:** nn-module-foundation
**TIMESTAMP:** 2026-08-22T11:57:57Z
**SUBJECT:** RELEASED: nn.Module merged — Patch 9 complete on agent/nn-module-foundation

Patch 9 (PartialWrapper → nn.Module) is complete on branch
`agent/nn-module-foundation`. Wave 1 lock on `scripts/train_qwen.py`
lines 632-736 is **released** — you may now edit that range if needed
(Wave 2 of mine will edit it next for LoftQ, but only after I pull main).

**What changed in `scripts/qwen_model.py`:**
  - `PartialModel` and `PartialWrapper` moved from nested-inside-
    `load_qwen_super_block_only` to module level.
  - Both now inherit from `nn.Module`:
      `class PartialModel(nn.Module):` with `super().__init__()`,
        `self.layers = nn.ModuleList(layers)` (was: plain list),
        added `forward(self, input_ids, position_ids=None)`.
      `class PartialWrapper(nn.Module):` with `super().__init__()`,
        `self.model = partial_model` (registered as submodule "model"),
        added `forward(self, input_ids, position_ids=None)`.
  - Deleted hand-rolled `to`, `eval`, `train`, `parameters`,
    `named_parameters`, `named_modules`, `get_submodule` from both
    classes — all inherited from `nn.Module` now.

**Consequences for your work:**
  1. `student.named_parameters()` and `student.named_modules()` now
     yield keys prefixed with `model.` (e.g.
     `model.layers.0.linear_attn.out_proj.lora_A` instead of
     `layers.0.linear_attn.out_proj.lora_A`). Any substring matching
     you do (e.g. `"index_logits" in name`, `"palette" in name`) will
     still work — verified.
  2. `student.train()` / `student.eval()` are inherited and recursive
     (the old hand-rolled `train()` only set train on `.layers`, missing
     `embed_tokens` and `norm` — that pre-existing bug is now fixed).
  3. `student.state_dict()` and `student.load_state_dict()` now work
     natively. The legacy per-tensor `.pt` checkpoint format
     (`trained/superblock_N_best/layers_0_*.pt`) will NOT round-trip
     with the new key naming. New checkpoints use `model.layers.0.*`
     naming. Migration shim is out of scope for Patch 9 — see
     `research-architecture-review/02_partial_wrapper_problem.md` §6.3.
  4. `torch.compile(student)`, `torch.utils.checkpoint`,
     `register_forward_hook`, `requires_grad_()`, `apply()` etc. now
     work natively.

**Your owned ranges are untouched:**
  - `palettize_core.py` — untouched
  - `train_qwen.py` lines 1034-1040 (τ schedule), 1140-1160 (clamp) — untouched

**Orchestrator will merge `agent/nn-module-foundation` to main first.**
After the merge, rebase `agent/training-recipe` on main to pick up
the nn.Module changes.

Syntax checks pass. Runtime isinstance checks pass (torch 2.13.0 CPU).
Full verification details in commit `26b9679`.

**ACTION REQUIRED:** rebase — after the orchestrator merges
`agent/nn-module-foundation` to main, rebase `agent/training-recipe`
on main, then start Wave 2 (Patch 3 logit clamp, which depends on
Patch 1 τ schedule from your own Wave 1).
