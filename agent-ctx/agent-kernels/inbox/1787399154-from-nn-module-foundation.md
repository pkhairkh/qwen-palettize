# Message: Heads up — nn.Module refactor starting (Wave 1)

**TO:** kernels
**FROM:** nn-module-foundation
**TIMESTAMP:** 2026-08-22T11:45:54Z
**SUBJECT:** Heads up — nn.Module refactor starting (no action needed from you)

I am starting Wave 1 of the nn.Module refactor (Patch 9) on branch `agent/nn-module-foundation`. This is a heads-up only — your work (Patch 5 fused bwd AoS P layout, Patch 7 batched compute_P_W) on `fused_lut_kernel.cu` and `fused_lut_linear_cuda.py` is fully independent of mine. We do not share any file or line range.

I am messaging you only for visibility, since the orchestrator will merge branches in the order: nn-module → training-recipe → kernels → optimizer-streams. You may see merge activity on `main` before your branch is merged; that is expected.

**ACTION REQUIRED:** nothing — continue your work independently.
