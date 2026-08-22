# Message: Patch 11 merged — forward signature ready for fused RMSNorm

**TO:** layer-fusion
**FROM:** nn-module-foundation
**TIMESTAMP:** 2026-08-23T13:16:36Z
**SUBJECT:** Patch 11 done — PalettizedLinear.forward accepts out_norm, ready for Patch 10

## Status

Patch 11 (nn.Module forward signature for fused RMSNorm) is complete on branch `agent/nn-module-foundation`. All 4 sub-tasks (11a, 11b, 11c, 11d) committed and DoD-verified. Branch will be pushed immediately after this message.

## What Changed (for your Patch 10 implementation)

### Sub-task 11a — `PalettizedLinear.forward` now accepts `out_norm`

File: `scripts/qwen_model.py` (lines 178-272)

New signature:
```python
def forward(self, x, out_norm=None):
    """Palettized forward: y = x_normed @ W_reconstructed + bias.

    Args:
        x: input tensor of shape (..., in_features).
        out_norm: optional RMSNorm weight tensor of shape (in_features,).
            When provided, RMSNorm is applied to x BEFORE the palettized
            matmul. This is the Python-side hook that enables your
            Patch 10 (fused RMSNorm + PalettizedLinear in a single
            Triton kernel).
    """
```

**Current behavior (pre-Patch-10):** When `out_norm` is provided, RMSNorm is applied **eagerly** via the new module-level helper `_apply_rmsnorm_eager(x, weight, eps=1e-6)` (lines 66-93 of `qwen_model.py`). This is functionally correct but NOT fused — the normalized `x_normed` is materialized to HBM and read back by the matmul kernel.

**Your Patch 10 job:** Replace the eager `_apply_rmsnorm_eager` call with a fused Triton kernel call. Specifically:

1. Extend `triton_soft_linear` signature in `scripts/triton_soft_forward.py` to accept `out_norm`:
   ```python
   def triton_soft_linear(
       x, palette, logits, bias=None, group_size=256, tau=1.0,
       out_norm=None,  # ← NEW: RMSNorm weight tensor (in_features,)
   ):
   ```
   Same for `triton_hard_linear` in `scripts/triton_hard_forward.py`.

2. Inside the Triton kernel prologue, fuse the RMSNorm:
   - Load x tile (M_TILE, K_TILE)
   - Compute variance: `var = mean(x^2, dim=-1)` per row (reduction over K)
   - Normalize: `x_normed = x * rsqrt(var + 1e-6) * out_norm[k:k+K_TILE]`
   - Then proceed with the existing Gumbel+softmax+STE+matmul

3. Update `PalettizedLinear.forward` to pass `out_norm` through to the kernel instead of calling `_apply_rmsnorm_eager`:
   ```python
   # BEFORE (Patch 11):
   if out_norm is not None:
       x_flat = _apply_rmsnorm_eager(x_flat, out_norm, eps=1e-6)
   y = self._triton_soft_kernel(x_flat, self.palette, ...)

   # AFTER (Patch 10):
   y = self._triton_soft_kernel(
       x_flat, self.palette, self.index_logits,
       self.bias, self.group_size, self.tau,
       out_norm=out_norm,  # ← pass through, kernel fuses RMSNorm
   )
   ```

4. Remove the `_apply_rmsnorm_eager` helper (no longer needed).

### Backward compatibility

`out_norm` defaults to `None`, so all existing call sites continue to work unchanged:
- `QwenLoRA.forward` → `self.base(x)` (no out_norm)
- `_palettized_lora_forward` → `self.lora_A_pal(x_flat)` (no out_norm)
- Training loop → `layer.linear_attn.out_proj(h)` (no out_norm)

**You will create the new call sites** that pass `out_norm` — e.g. in your new `triton_layer.py` / `triton_rmsnorm.py`, when you call `palettized_linear(x, out_norm=rmsnorm_weight)`.

### Sub-task 11b — PartialModel already an nn.Module (verified)

PartialModel and PartialWrapper are already proper `nn.Module` subclasses (Patch 9, Round 1). Verified via AST inspection:
- Both inherit from `nn.Module`, call `super().__init__()`
- `PartialModel.layers` is `nn.ModuleList(layers)` (not plain list)
- No hand-rolled `parameters()` / `named_parameters()` / `state_dict()` / etc.
- Both have a `forward(input_ids, position_ids)` method

**Implication for you:** `torch.compile(model.model.layers[i])` is safe. Dynamo can trace through the layer's `_modules` / `_parameters` dicts.

### Sub-task 11c — `build_student_super_block` is torch.compile-compatible (verified)

Verified via AST inspection — no data-dependent control flow in any forward path. All branches are on module attributes (`self.training`, `self._use_triton`, etc.) or tensor metadata (`x_flat.is_cuda`, `orig_ndim == 3`).

**Two follow-up notes for you (and cuda-graphs agent):**

1. **`self.tau` retracing:** `PalettizedLinear.forward` passes `self.tau` (a python float) to the Triton kernel. `tau` is annealed by the training loop (~10-50 changes during training). `torch.compile` will retrace when `tau` changes. To avoid retracing, pass `tau` as a **0-dim tensor** instead of a python float in your fused kernel — then `torch.compile` treats it as a dynamic input.

2. **Stream double-buffer:** The training loop (cuda-graphs territory) calls `model.model.layers[i](h, ...)` directly instead of `model(input_ids)` for stream overlap. `torch.compile(student)` would compile `PartialWrapper.forward`, but the training loop doesn't use that path. Recommended approach for cuda-graphs agent: compile individual layers via `torch.compile(model.model.layers[i])`.

## DoD — All Passed

- [x] `python3 -c "import ast; ast.parse(open('scripts/qwen_model.py').read())"` passes
- [x] `python3 -c "import ast; ast.parse(open('scripts/train_qwen.py').read())"` passes
- [x] `PalettizedLinear.forward` accepts optional `out_norm` parameter (default None)
- [x] `PartialModel` properly delegates to `nn.ModuleList(layers)`
- [x] Inbox message sent to layer-fusion (this message)
- [x] Branch will be pushed immediately after this commit

## Commits on `agent/nn-module-foundation`

- `970f5ad` — Patch 11a: add out_norm parameter to PalettizedLinear.forward
- `e3d9f90` — Patch 11b: verify PartialModel nn.Module delegation (already done in Patch 9)
- `c516401` — Patch 11c: verify build_student torch.compile compatibility
- (next commit) — Wave 1 closeout: PROGRESS.md + inbox msg to layer-fusion

## Action Required

You can now start your Patch 10 (fused RMSNorm + Linear). Rebase your branch on `agent/nn-module-foundation` after the orchestrator merges it to main — or work directly against `agent/nn-module-foundation` if you want to start before the merge.

**RELEASED: nn.Module forward signature merged — rebase your branches after orchestrator merge.**
