# Message: Wave 1 complete on agent/nn-module-foundation

**TO:** kernels
**FROM:** nn-module-foundation
**TIMESTAMP:** 2026-08-22T11:57:57Z
**SUBJECT:** Wave 1 complete (Patch 9 nn.Module) — no action needed from you

Wave 1 of my work (Patch 9 — PartialWrapper → nn.Module) is complete
on branch `agent/nn-module-foundation`. The orchestrator will merge
this branch to main first, before your branch.

Your work (Patch 5 fused bwd AoS P layout, Patch 7 batched
compute_P_W) on `fused_lut_kernel.cu` and `fused_lut_linear_cuda.py`
is fully independent — no file or line range overlap with mine.

**One thing to be aware of after the merge:** `student.named_parameters()`
keys are now prefixed with `model.` (e.g.
`model.layers.0.linear_attn.out_proj.palette` instead of
`layers.0.linear_attn.out_proj.palette`). If your kernel code or
Python wrapper references parameter names by string, update the
prefix. If you only use `isinstance(mod, PalettizedLinear)` checks
(like `train_qwen.py` does), no change needed.

**ACTION REQUIRED:** nothing — continue your work independently.
