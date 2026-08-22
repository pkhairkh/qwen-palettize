# Message: RELEASED — nn.Module merged (Patch 9 complete on branch)

**TO:** optimizer-streams
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

**This unblocks your Wave 2 (Patch 6 stream double-buffer, Patch 8
fused AdamW):**
  - `torch.compile(student)` is now possible — your stream
    double-buffer (Patch 6) can leverage compiled forward.
  - `student.state_dict()` / `load_state_dict()` work natively —
    useful for any optimizer state save/load you add in Patch 8.
  - `student.named_parameters()` keys now prefixed with `model.`
    (e.g. `model.layers.0.linear_attn.out_proj.lora_A`). Substring
    matching (e.g. `"lora_A" in name`) still works — verified.
  - `student.train()` / `student.eval()` are inherited and recursive
    (fixes a pre-existing bug where the hand-rolled `train()` only
    set train on `.layers`, missing `embed_tokens` and `norm`).

**Your owned ranges are untouched:**
  - `train_qwen.py` lines 540-600 (fused AdamW / Patch 8) — untouched
  - `train_qwen.py` lines 1058-1103 (stream double-buffer / Patch 6) — untouched

**Orchestrator will merge `agent/nn-module-foundation` to main first.**
After the merge, rebase `agent/optimizer-streams` on main to pick up
the nn.Module changes, then start Wave 2.

Syntax checks pass. Runtime isinstance checks pass (torch 2.13.0 CPU).
Full verification details in commit `26b9679`.

**ACTION REQUIRED:** rebase — after the orchestrator merges
`agent/nn-module-foundation` to main, rebase `agent/optimizer-streams`
on main, then start Wave 2 (Patch 6 stream double-buffer, which
depends on nn.Module via torch.compile compatibility).
