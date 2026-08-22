# TASKS: nn-module-foundation

## Branch
`agent/nn-module-foundation`

## Overview
You are the FOUNDATION agent. Your Patch 11 unlocks `torch.compile` and fused Triton kernels for the layer-fusion agent. Merge FIRST.

## Patch Inventory
| # | Patch | Wave | Effort | Status |
|---|-------|------|--------|--------|
| 11 | nn.Module forward signature for fused RMSNorm | 1 | 0.5 day | ⬜ |

---

## WAVE 1: Patch 11 — nn.Module forward signature

### Sub-task 11a: Add `out_norm` parameter to PalettizedLinear.forward
**Research:** `research-architecture-review/02_partial_wrapper_problem.md`
**Paper:** `docs/papers/2305.14314_QLoRA_Dettmers2023.pdf` (QLoRA — nn.Module required for torch.compile)

**File:** `scripts/qwen_model.py:65-180` (PalettizedLinear class)

**What:** The current `PalettizedLinear.forward(self, x)` needs to accept an optional `out_norm` parameter. When provided, the fused RMSNorm is applied BEFORE the palettized matmul, inside the Triton kernel (this enables the layer-fusion agent's Patch 10). The signature should be:
```python
def forward(self, x, out_norm=None):
    # If out_norm is provided, the Triton soft/hard kernel will fuse the RMSNorm.
    # out_norm is the RMSNorm weight tensor (shape [in_features]).
    ...
```

**Commit:** `Patch 11a: add out_norm parameter to PalettizedLinear.forward`

### Sub-task 11b: Verify PartialModel delegates to nn.ModuleList
**Research:** `research-architecture-review/02_partial_wrapper_problem.md`
**File:** `scripts/qwen_model.py:438-600`

**What:** Verify that `PartialModel` properly inherits from `nn.Module` and uses `nn.ModuleList(layers)`. This was done in Round 1 (Patch 9) but verify it still works. Run: `python3 -c "import ast; ast.parse(open('scripts/qwen_model.py').read()); print('OK')"`. Also check that `model.state_dict()` works (no custom save_state needed).

**Commit:** `Patch 11b: verify PartialModel nn.Module delegation (already done in Patch 9)`

### Sub-task 11c: Verify build_student_super_block is torch.compile-compatible
**Research:** `research-architecture-review/02_partial_wrapper_problem.md`
**File:** `scripts/train_qwen.py:632-736`

**What:** Verify that `build_student_super_block` returns a proper `nn.Module` that can be passed to `torch.compile`. Check for dynamic shapes (no data-dependent control flow in forward). If any dynamic shapes exist, document them for the layer-fusion agent.

**Commit:** `Patch 11c: verify build_student torch.compile compatibility`

### Sub-task 11d: Send inbox messages + push
- Send to layer-fusion: "forward signature ready, you can implement fused RMSNorm (Patch 10). PalettizedLinear.forward now accepts optional `out_norm` parameter."
- Update `agent-ctx/PROGRESS.md` (append-only).
- Push branch.

**Commit:** `Wave 1 closeout: PROGRESS.md update + inbox msg to layer-fusion`

---

## DoD for Wave 1
- [ ] `python3 -c "import ast; ast.parse(open('scripts/qwen_model.py').read())"` passes
- [ ] `python3 -c "import ast; ast.parse(open('scripts/train_qwen.py').read())"` passes
- [ ] `PalettizedLinear.forward` accepts optional `out_norm` parameter
- [ ] `PartialModel` properly delegates to `nn.ModuleList(layers)`
- [ ] Inbox message sent to layer-fusion
- [ ] Branch pushed to origin
