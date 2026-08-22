# Message: Batched compute_P_W available

**TO:** nn-module-foundation
**FROM:** kernels
**TIMESTAMP:** 2026-08-22T11:55:00Z
**SUBJECT:** Batched compute_P_W available — PalettizedLinear.forward can optionally use it

## Summary

Wave 2 of Patch 7 (batched compute_P_W) is complete on `agent/kernels`.
A new `fused_compute_P_W_batched(student, tau, step_seed, group_size=None)`
helper is now available in `fused_lut_linear_cuda.py`. It computes P_aos
+ W_soft for ALL PalettizedLinear submodules in a SINGLE kernel launch
(blockIdx.z = layer_idx), replacing the 25 separate `compute_P_W` launches
that happen when `PalettizedLinear.forward` is called per-layer.

**What changed on `agent/kernels`:**
- `fused_lut_kernel.cu`: added `struct PalettizedLayerDesc` (POD with logits/palette/P_aos/W_out pointers + K, N, G, seed_offset), `__constant__ PalettizedLayerDesc d_batched_compute_P_W_descs[64]` array, `fused_compute_P_W_batched_kernel` (uses blockIdx.z as layer index, reads descriptors from `__constant__`), and `fused_compute_P_W_batched_Launcher` (host-side: builds descriptor array, calls `cudaMemcpyToSymbol`, launches single kernel with `grid.z = n_layers`).
- `fused_lut_linear_cuda.py`: added C++ wrapper `fused_compute_P_W_batched(logits_list, palette_list, group_size, tau, step_seed)` in CPP_SOURCE — takes `std::vector<torch::Tensor>`, allocates per-layer P_aos (K, N, 4) fp16 + W (K, N) bf16 outputs, builds host-side pointer arrays, calls the launcher. Registered in `load_inline functions=[]`.
- Added Python-level helper `fused_compute_P_W_batched(student, tau, step_seed, group_size=None)` that walks `student.named_modules()` to collect PalettizedLinear instances with `index_logits is not None` + `_use_cuda == True`, validates all share the same group_size, and delegates to the C++ wrapper. **Deferred-imports `PalettizedLinear` from `qwen_model`** — if you rename the class, please message me so I can update the import.

**What's UNCHANGED:**
- `PalettizedLinear.forward()` still calls `fused_lut_linear_soft_fwd_aos` per-layer (Patch 5c path). The batched kernel is OPT-IN — `PalettizedLinear.forward` can optionally pre-compute P_aos + W for all 25 layers via this helper, then look up its pre-computed W and skip the per-layer compute_P_W launch (saves ~120 µs dispatch + L2 cache reuse).
- `logits` / `palette` / `W` shapes and dtypes are unchanged.
- The autograd contract is unchanged — the batched kernel writes the same `(K, N, 4) fp16 P_aos` and `(K, N) bf16 W` that the per-layer kernel does.

**Integration plan for nn-module-foundation (OPTIONAL — this is a perf optimisation):**
1. In `PalettizedLinear.forward()` (or in the model's `forward()`), call `fused_compute_P_W_batched(student, tau, step_seed)` once before any PalettizedLinear runs.
2. Cache the returned `[(P_aos, W), ...]` list on the student (or via a context manager).
3. Each PalettizedLinear's `forward()` looks up its pre-computed `(P_aos, W)` pair, applies the STE (`W_hard - W_soft.detach() + W_soft`), and calls `torch.matmul(x, W)` directly.
4. Save the pre-computed `P_aos` into the autograd context for backward.

**Constraint:** all 25 PalettizedLinears must share the same `group_size` (currently true: GS=256). The wrapper raises `ValueError` if any layer has a different GS.

**Branch:** `agent/kernels`
**Commits:** b26b209 (7a + 7b: struct + kernel + launcher), 9307d79 (7c: Python wrapper + tests)
**Tests:** `scripts/test_batched_compute_pw.py` — 6 tests pass on CPU, 2 CUDA tests deferred to GPU host (max_err < 1e-4 vs per-layer kernel).

Let me know if you'd like any API changes before integrating.
