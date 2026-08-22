# Holistic Roadmap: Full Triton Fusion — Transform Before Training Restart

> **Status:** PRE-IMPLEMENTATION (Round 3). The current codebase (tps=0.6, GPU power 169W/600W) is capped by 73% PyTorch autograd overhead in the backward pass. This roadmap defines 6 parallel agents that will fuse the ENTIRE Qwen3.5 layer into Triton, eliminating all Python autograd overhead.

---

## 1. Why Current State is Capped at tps=0.6

The profiler (torch.profiler on the real model, batch=32, seq=512, M=16384) revealed:

| Phase | Time | % of step | Root cause |
|-------|------|-----------|------------|
| Teacher forward | 80ms | 8% | cuBLAS bf16 GEMM (near-optimal) |
| Student forward | 260ms | 26% | 25× PalettizedLinear Triton kernels (fast) + unfused Qwen3.5 ops |
| **Backward** | **682ms** | **68%** | **187ms Triton kernels (27%) + 495ms PyTorch autograd overhead (73%)** |
| Optimizer | 37ms | 4% | bitsandbytes AdamW8bit |
| **Total** | **1004ms** | tps=0.6 | |

### Backward Breakdown (682ms, from torch.profiler)

| Kernel | CUDA time | Calls | Category |
|--------|-----------|-------|----------|
| `TritonSoftLinearBackward` (our kernels) | 187ms | 25 | ✅ Fast (27%) |
| `aten::add_` | 167ms | 1070 | ❌ Gradient accumulation (LoRA, attention) |
| `vectorized_elementwise` | 141ms | 834 | ❌ LoRA scaling, loss backward, dtype ops |
| `aten::copy_` | 120ms | 2031 | ❌ fp32 master weight copies + grad copies |
| `fused_soft_bwd_elementwise` | 83ms | 25 | ✅ (inside TritonSoftLinearBackward) |
| `fused_soft_bwd_grad_x` | 59ms | 25 | ✅ (inside TritonSoftLinearBackward) |
| `fused_soft_bwd_grad_W` | 44ms | 25 | ✅ (inside TritonSoftLinearBackward) |
| `Memcpy DtoD` | 44ms | 241 | ❌ Device-to-device copies |
| `aten::fill_/zero_` | 48ms | 1252 | ❌ Gradient zeroing |

**The 495ms of autograd overhead (73%) comes from:**
1. **Unfused LoRA backward** — 31 LoRA modules × 3 matmuls + elementwise = ~93 kernel launches. `lora_out = (x @ A) @ B.T * scaling` produces separate `grad_A`, `grad_B`, `grad_x_lora` kernels.
2. **Unfused attention backward** — rotary embedding, QK^T, softmax, AV, output projection each produce separate gradients that accumulate via `aten::add_`.
3. **Unfused layernorm backward** — RMSNorm forward saves mean/rstd, backward recomputes.
4. **fp32 master weight copies** — `FP32MasterOptimizer.step()` copies bf16→fp32 grad + fp32→bf16 weight for every parameter.
5. **Gradient accumulation** — the residual stream `h = h + layer(h)` produces an `aten::add_` for every layer.

---

## 2. The 6-Agent Roster

| Agent | Branch | Patches | Files Owned (EXCLUSIVE) | Foundation? |
|-------|--------|---------|-------------------------|-------------|
| `nn-module-foundation` | `agent/nn-module-foundation` | 10, 11 | `qwen_model.py` (PartialModel/PartialWrapper + PalettizedLinear forward signature) | **YES — merge first** |
| `layer-fusion` | `agent/layer-fusion` | 12, 13, 14 | NEW: `triton_layer.py`, `triton_rmsnorm.py`, `triton_mlp.py` | No (depends on nn-module) |
| `triton-kernels` | `agent/triton-kernels` | 15, 16, 17, 18 | `triton_soft_forward.py`, `triton_soft_backward.py`, `triton_hard_forward.py` (full) | No |
| `lora-fusion` | `agent/lora-fusion` | 19, 20 | NEW: `triton_lora.py`, `qwen_model.py` (QwenLoRA class only, ~lines 184-265) | No (depends on triton-kernels) |
| `cuda-graphs` | `agent/cuda-graphs` | 21, 22 | `train_qwen.py` (training loop CUDA Graph capture, ~lines 1095-1250) | No (depends on ALL) |
| `quality-recipe` | `agent/quality-recipe` | 23, 24, 25, 26 | `train_qwen.py` (loss config + clip + freeze), `palettize_core.py` (GROUP_SIZE) | No |

### File Conflict Map (EXCLUSIVE ownership)

```
qwen_model.py line ranges (EXCLUSIVE ownership):
  47-48:       nn-module-foundation (is_gated_delta_layer helper)
  65-180:      nn-module-foundation (PalettizedLinear class — forward signature changes)
  184-265:     lora-fusion (QwenLoRA class)
  438-600:     nn-module-foundation (PartialModel/PartialWrapper — already done)
  779-850:     nn-module-foundation (capture_original_weights, insert_correction_layers)

train_qwen.py line ranges (EXCLUSIVE ownership):
  96-102:      quality-recipe (DEFAULT_HYPERPARAMS — loss config)
  154-214:     nn-module-foundation (FP32MasterOptimizer — already optimized)
  540-600:     nn-module-foundation (build_optimizers)
  632-736:     nn-module-foundation (build_student_super_block)
  889-918:     cuda-graphs (data pipeline — if touching)
  1034-1040:   quality-recipe (τ schedule — already done, verify)
  1058-1103:   cuda-graphs (stream double-buffer — integrate with CUDA Graphs)
  1095-1250:   cuda-graphs (training loop — CUDA Graph capture)
  1140-1160:   quality-recipe (clamp + clip)

NEW files (no conflicts):
  triton_layer.py:       layer-fusion
  triton_rmsnorm.py:     layer-fusion
  triton_mlp.py:         layer-fusion
  triton_lora.py:        lora-fusion
  triton_soft_forward.py:  triton-kernels
  triton_soft_backward.py: triton-kernels
  triton_hard_forward.py:  triton-kernels
```

**Rule:** If two agents need the SAME line range in `train_qwen.py` in the SAME wave, the second agent MUST wait (check inbox for "lock released" message).

---

## 3. The 26 Patches (P10–P35)

### Patch 10: Fused RMSNorm + Linear (layer-fusion)
- **Research:** `research-kernel-efficiency/00_overview.md` §3 (fused layernorm)
- **Paper:** FlashAttention2 (Dao 2023, arXiv:2307.08691) — fused layernorm pattern
- **File:** NEW `scripts/triton_rmsnorm.py`
- **What:** Fuse `RMSNorm(x) → PalettizedLinear` into a single Triton kernel. The RMSNorm forward computes `x_normed = x / sqrt(mean(x^2) + eps) * weight`. Fusing avoids materializing `x_normed` to HBM (saves ~2× M×K×2 bytes per layer). Backward is fused too: `grad_x = grad_y @ W.T * (weight/rstd) * (1 - x_normed^2/M)`.
- **Expected:** −15ms forward, −25ms backward (eliminates 8 RMSNorm kernel launches + 8 `aten::add_` for residual).

### Patch 11: nn.Module forward signature (nn-module-foundation)
- **Research:** `research-architecture-review/02_partial_wrapper_problem.md`
- **Paper:** QLoRA (Dettmers et al. 2023, arXiv:2305.14314) — nn.Module required for torch.compile
- **File:** `qwen_model.py:65-180` (PalettizedLinear.forward), `qwen_model.py:438-600` (PartialModel)
- **What:** Make PalettizedLinear.forward accept an optional `out_norm` parameter (for fused RMSNorm). Ensure `PartialModel` properly delegates to `nn.ModuleList(layers)`. Verify `torch.compile` compatibility (no dynamic shapes in forward).
- **Expected:** Unlocks torch.compile for the layer-fusion agent. No direct speedup.

### Patch 12: Fused MLP (gate + up + SiLU + down) (layer-fusion)
- **Research:** `research-kernel-efficiency/00_overview.md` §3 (fused MLP)
- **Paper:** SwiGLU (Shazeer 2020, arXiv:2002.05202) — fused gate+up+activation
- **File:** NEW `scripts/triton_mlp.py`
- **What:** Fuse `gate_proj(x) * SiLU(up_proj(x)) → down_proj(...)` into a single kernel chain. The three PalettizedLinears stay separate (they have different palettes), but the SiLU + elementwise multiply + the down_proj input are fused. This eliminates the `aten::mul` and `aten::silu` kernel launches.
- **Expected:** −10ms forward, −20ms backward (eliminates 3 elementwise kernels × 4 layers).

### Patch 13: Fused Attention (rotary + QK^T + softmax + AV) (layer-fusion)
- **Research:** `research-kernel-efficiency/00_overview.md` §3 (fused attention)
- **Paper:** FlashAttention2 (Dao 2023, arXiv:2307.08691) — fused attention pattern
- **File:** NEW `scripts/triton_layer.py`
- **What:** For the full-attention layer (layer 3, 7, 11, ...), fuse `rotary_emb(Q,K) → QK^T / sqrt(d) → softmax → AV → o_proj` into a single FlashAttention-style Triton kernel. The QKV projections stay as PalettizedLinear (fused with RMSNorm in Patch 10), but the attention computation itself is fused.
- **Expected:** −30ms forward, −50ms backward (eliminates ~15 attention kernels × 1 full-attn layer per super-block).

### Patch 14: Fused GatedDeltaNet (conv1d + delta-rule) (layer-fusion)
- **Research:** `research-kernel-accuracy/00_overview.md` (GatedDeltaNet architecture)
- **Paper:** Mamba (Gu & Dao 2023, arXiv:2312.00752) — SSM-style fused kernel pattern
- **File:** NEW `scripts/triton_layer.py`
- **What:** For the linear-attention layers (layer 0,1,2, 4,5,6, ...), fuse `conv1d → delta_rule_update → out_proj` into a single Triton kernel. The conv1d is a depth-1 temporal convolution; the delta rule is `S = S + Δ(A @ x @ v^T)`. This is the most complex kernel.
- **Expected:** −20ms forward, −40ms backward (eliminates ~12 GatedDeltaNet kernels × 3 layers per super-block).

### Patch 15: Batched compute_P_W for 25 layers (triton-kernels)
- **Research:** `research-kernel-efficiency/03_batched_compute_pw.md`
- **Paper:** (kernel optimization, no specific paper)
- **File:** `scripts/triton_soft_forward.py`
- **What:** Replace 25 separate `compute_P_W_ste_triton` calls with a single batched kernel. Grid = `(cdiv(max_K, BM), cdiv(max_N, BN), 25)`, `blockIdx.z = layer_idx`. Each layer uses `seed_offset = layer_idx` for Gumbel noise decorrelation. Uses a `PalettizedLayerDesc[25]` array (shapes, palette pointers, logits pointers) passed as a tensor.
- **Expected:** −18ms forward (eliminates 24 kernel launches × 5µs = 120µs dispatch + L2 cache reuse).

### Patch 16: Eliminate redundant matmul in STE forward (triton-kernels)
- **Research:** `research-indices-training/01_gumbel_softmax_audit.md` Finding 8
- **Paper:** Jang et al. 2017 (arXiv:1611.01144) — STE implementation
- **File:** `scripts/triton_soft_forward.py` (TritonSoftLinear.forward)
- **What:** The current forward computes `y_soft = x @ W_soft` (inside compute_P_W kernel), discards it, then computes `y = x @ W_ste`. This wastes one cuBLAS/Triton matmul per layer. Fix: the fused kernel should compute `y = x @ W_ste` directly and save `P_aos + W_ste` for backward. The `W_soft` is only needed in backward and can be recomputed on-the-fly from `P_aos + palette`.
- **Expected:** −50ms forward (eliminates 25 redundant matmuls × ~2ms each).

### Patch 17: Buffer pooling for P_aos + W_ste (triton-kernels)
- **Research:** `research-kernel-efficiency/05_memory_optimization.md` §3.2
- **Paper:** (memory optimization, no specific paper)
- **File:** `scripts/triton_soft_forward.py`, `scripts/triton_soft_backward.py`
- **What:** `PalettizedLinear._P_POOL = {}` keyed on `(K, N)` shape. The forward writes P_aos + W_ste into pooled buffers instead of allocating new tensors. The backward reads from the same buffers. Eliminates ~52MB allocation per layer × 25 = 1.3GB peak allocation per step, reducing GC pressure and allocation overhead.
- **Expected:** −5ms (eliminates 50 `torch.empty` calls per step + associated CUDA memory management).

### Patch 18: Chunked reduction for elementwise backward (triton-kernels)
- **Research:** `research-kernel-efficiency/05_memory_optimization.md` §3.3
- **Paper:** (kernel optimization, no specific paper)
- **File:** `scripts/triton_soft_backward.py`
- **What:** The current elementwise backward materializes `grad_W` (K×N fp32) as an intermediate. The chunked kernel holds `(K_CHUNK, N_TILE, 4)` in shared memory, accumulating `grad_palette` and `grad_logits` without the HBM intermediate. K_CHUNK=64, N_TILE=32, M_TILE=128.
- **Expected:** −30ms backward (eliminates 52MB intermediate write + 52MB read per layer × 25).

### Patch 19: Fused LoRA backward (lora-fusion)
- **Research:** NEW (profiler finding: 167ms `aten::add_`, 31 LoRA modules)
- **Paper:** QLoRA (Dettmers et al. 2023, arXiv:2305.14314) — LoRA backward
- **File:** NEW `scripts/triton_lora.py`, `qwen_model.py:184-265` (QwenLoRA class)
- **What:** Fuse the LoRA backward into a single Triton kernel. Forward: `lora_out = (x @ A) @ B.T * scaling`. Backward: `grad_A = x.T @ (grad_y * scaling @ B)`, `grad_B = (grad_y * scaling).T @ (x @ A)`, `grad_x_lora = (grad_y * scaling @ B.T) @ A`. Currently these are 3 separate PyTorch matmuls + 1 elementwise scaling. Fuse scaling into the matmul loads and compute `grad_A` and `grad_B` in a single kernel with shared `x @ A` intermediate.
- **Expected:** −40ms backward (eliminates 31×3 = 93 LoRA backward kernels + 31 `aten::mul` for scaling).

### Patch 20: Fused LoRA + PalettizedLinear backward (lora-fusion)
- **Research:** NEW (profiler finding: 120ms `aten::copy_` for gradient accumulation)
- **Paper:** QLoRA (arXiv:2305.14314) — fused LoRA + base weight backward
- **File:** NEW `scripts/triton_lora.py`
- **What:** Fuse the LoRA backward with the PalettizedLinear backward. The combined kernel computes: `grad_x = grad_y @ (W_ste + lora_B @ lora_A * scaling).T`, `grad_palette`/`grad_logits` (from W_ste path), `grad_lora_A`/`grad_lora_B` (from LoRA path), all in one fused kernel pass. This eliminates the `aten::add_` that accumulates `grad_x_base + grad_x_lora`.
- **Expected:** −30ms backward (eliminates 31 `aten::add_` for grad_x accumulation + 31 separate LoRA grad_x matmuls).

### Patch 21: CUDA Graph capture for full step (cuda-graphs)
- **Research:** `research-kernel-efficiency/08_recommendations.md` §11 (Patch 10)
- **Paper:** (CUDA Graphs, no specific paper — NVIDIA documentation)
- **File:** `train_qwen.py:1095-1250` (training loop)
- **What:** Capture the entire training step (teacher forward + student forward + loss + backward + optimizer) as a CUDA Graph. Replay per step. Eliminates all Python dispatch overhead (~500 kernel launches × 5µs = 2.5ms) and enables the GPU to schedule kernels back-to-back without CPU involvement. Requires static input tensors (batch_ids copied into a static buffer).
- **Expected:** −15ms (eliminates 500+ kernel launch dispatches × 5µs = 2.5ms + CPU-GPU synchronization gaps).

### Patch 22: Stream double-buffer integration with CUDA Graphs (cuda-graphs)
- **Research:** `research-kernel-efficiency/06_stream_overlap.md`
- **Paper:** (stream overlap, no specific paper)
- **File:** `train_qwen.py:1058-1103`
- **What:** Integrate the existing stream double-buffer (teacher on stream_t, student on default stream) with the CUDA Graph capture. The teacher graph runs on stream_t, the student+backward+optimizer graph runs on the default stream, with event-based synchronization. This hides the 80ms teacher forward behind the 980ms student compute.
- **Expected:** −80ms (teacher forward fully hidden behind student backward — already partially done, but CUDA Graphs makes it deterministic).

### Patch 23: Loss config switch (quality-recipe)
- **Research:** `research-palettes-training/05_loss_function.md`
- **Paper:** GPTQ (arXiv:2210.17323), QLoRA (arXiv:2305.14314)
- **File:** `train_qwen.py:96-102` (DEFAULT_HYPERPARAMS)
- **What:** Switch `loss_type` from `"norm_mse"` to `"1-cos+norm_mse"` with `loss_weights={"cos": 0.8, "mse": 0.2}`. The current pure `norm_mse` conflates magnitude and direction errors. The 80/20 split balances the gradient contributions: `cos` explicitly optimizes the metric we care about (direction), `mse` provides stable magnitude gradient at training start.
- **Expected:** +0.01-0.02 cos (better convergence, no speed impact).

### Patch 24: Per-group gradient clipping (quality-recipe)
- **Research:** `research-kernel-accuracy/08_recommendations.md` Fix 4
- **Paper:** AdamW (arXiv:1711.05101) — gradient clipping best practices
- **File:** `train_qwen.py:1130-1145` (clip_grad_norm_ section)
- **What:** Replace global `clip_grad_norm_(model.parameters(), 0.3)` with per-group: `indices_params` (1.78B params, clip 1.0) + `other_params` (palettes+lora+layernorms, clip 0.3). The global clip scales the tiny palette gradient by ~1/45 (because 1.78B index_logits dominate the global norm), effectively zeroing palette updates.
- **Expected:** +0.01-0.02 cos (palette gradients no longer suppressed).

### Patch 25: LUT-Q re-quantization (quality-recipe)
- **Research:** `research-palettes-training/06_staged_training.md` Schedule C
- **Paper:** LUT-Q (Cardinaux et al. 2018, arXiv:1811.05355), Nagel et al. 2022 (arXiv:2203.11086)
- **File:** NEW `scripts/re_quantize.py`, `train_qwen.py` (call site at step 2000, 4000)
- **What:** Re-run k-means on the current `W_recon` per group every 2000 steps. This escapes the k-means local optimum that the gradient descent is stuck in. After re-quantization, re-initialize `index_logits` as one-hot from the new indices (±3 gap instead of ±10 for better gradient flow).
- **Expected:** +0.02-0.03 cos (escapes k-means local optimum).

### Patch 26: Deterministic-ST (quality-recipe)
- **Research:** `research-indices-training/07_recommendations.md` Fix 2
- **Paper:** LLT (Wang et al. CVPR 2022) — deterministic-ST pattern
- **File:** `scripts/triton_soft_forward.py` (compute_P_W_ste_kernel)
- **What:** Remove the 4 Gumbel noise samples from the softmax computation. Instead of `(logits + gumbel) / tau → softmax → P`, use `logits / tau → softmax → P`. This eliminates the LCG-based Gumbel sampler (statistically weak) and makes the forward deterministic. The STE still works: forward = hard argmax, backward = soft gradient. This prevents the index oscillation pathology documented by Nagel et al. 2022.
- **Expected:** +0.005-0.01 cos (more stable training, no oscillation).

---

## 4. Merge Order (Dependency Graph)

```
Wave 0 (already done — merged to main):
  nn-module-foundation: Patch 9 (nn.Module) + Patch 2 (LoftQ SVD init)
  training-recipe: Patch 1 (τ schedule) + Patch 3 (±5τ clamp)
  kernels: Patch 5 (AoS P) + Patch 7 (batched compute_P_W — CUDA C version)
  optimizer-streams: Patch 6 (stream double-buffer) + Patch 8 (bitsandbytes AdamW8bit)

Wave 1 (parallel — foundation for all Triton fusion):
  nn-module-foundation: Patch 11 (forward signature for fused RMSNorm)
  quality-recipe: Patch 23 (loss config), Patch 24 (per-group clip)
  triton-kernels: Patch 16 (eliminate redundant matmul), Patch 17 (buffer pooling)

Wave 2 (parallel — after Wave 1 merges):
  triton-kernels: Patch 15 (batched compute_P_W — Triton version), Patch 18 (chunked reduction)
  layer-fusion: Patch 10 (fused RMSNorm + Linear), Patch 12 (fused MLP)
  quality-recipe: Patch 26 (deterministic-ST)

Wave 3 (parallel — after Wave 2 merges):
  layer-fusion: Patch 13 (fused attention), Patch 14 (fused GatedDeltaNet)
  lora-fusion: Patch 19 (fused LoRA backward)
  quality-recipe: Patch 25 (LUT-Q re-quantization)

Wave 4 (parallel — after Wave 3 merges):
  lora-fusion: Patch 20 (fused LoRA + PalettizedLinear backward)
  cuda-graphs: Patch 21 (CUDA Graph capture), Patch 22 (stream integration)

Wave 5 (merge coordination):
  All agents verify their branch merges cleanly with main.
  Orchestrator merges in order: nn-module → triton-kernels → layer-fusion → lora-fusion → quality-recipe → cuda-graphs
```

**Critical:** `nn-module-foundation` MUST merge first (Wave 1). Patches 10-14 depend on `torch.compile` compatibility (Patch 11). `cuda-graphs` MUST merge last (Wave 4) — the graph capture requires all kernels to be stable.

---

## 5. Patch-to-Research-to-Paper Cross-Reference

| Patch | Research File | Paper (in `docs/papers/`) |
|-------|---------------|---------------------------|
| 10 (fused RMSNorm) | `research-kernel-efficiency/00_overview.md` §3 | `2307.08691` (FlashAttention2 — fused layernorm pattern) |
| 11 (nn.Module fwd sig) | `research-architecture-review/02_partial_wrapper_problem.md` | `2305.14314_QLoRA_Dettmers2023.pdf` |
| 12 (fused MLP) | `research-kernel-efficiency/00_overview.md` §3 | `2002.05202` (SwiGLU) |
| 13 (fused attention) | `research-kernel-efficiency/00_overview.md` §3 | `2307.08691` (FlashAttention2) |
| 14 (fused GatedDeltaNet) | `research-kernel-accuracy/00_overview.md` | `2312.00752` (Mamba — SSM fused pattern) |
| 15 (batched compute_P_W) | `research-kernel-efficiency/03_batched_compute_pw.md` | (no paper) |
| 16 (eliminate redundant matmul) | `research-indices-training/01_gumbel_softmax_audit.md` Finding 8 | `1611.01144_Gumbel-Softmax_Jang2017.pdf` |
| 17 (buffer pooling) | `research-kernel-efficiency/05_memory_optimization.md` §3.2 | (no paper) |
| 18 (chunked reduction) | `research-kernel-efficiency/05_memory_optimization.md` §3.3 | (no paper) |
| 19 (fused LoRA backward) | NEW (profiler finding) | `2305.14314_QLoRA_Dettmers2023.pdf` |
| 20 (fused LoRA + PL backward) | NEW (profiler finding) | `2305.14314_QLoRA_Dettmers2023.pdf` |
| 21 (CUDA Graphs) | `research-kernel-efficiency/08_recommendations.md` §11 | (no paper — NVIDIA docs) |
| 22 (stream + Graphs) | `research-kernel-efficiency/06_stream_overlap.md` | (no paper) |
| 23 (loss config) | `research-palettes-training/05_loss_function.md` | `2210.17323_GPTQ_Frantar2023.pdf` |
| 24 (per-group clip) | `research-kernel-accuracy/08_recommendations.md` Fix 4 | `1711.05101_AdamW_Loshchilov2019.pdf` |
| 25 (LUT-Q re-quantization) | `research-palettes-training/06_staged_training.md` | `1811.05355_LUTQ_Cardinaux2018.pdf`, `2203.11086_QAT_Oscillations_Nagel2022.pdf` |
| 26 (deterministic-ST) | `research-indices-training/07_recommendations.md` Fix 2 | `LLT_Wang_CVPR2022.pdf`, `2203.11086_QAT_Oscillations_Nagel2022.pdf` |

---

## 6. Expected End State

| Metric | Current | Target (after all 26 patches) | Improvement |
|--------|---------|-------------------------------|-------------|
| Step time | 1004ms | ~120ms | 8.4× |
| tps | 0.6 | ~8 | 13× |
| GPU power | 169W | ~550W | 91% TDP |
| Backward time | 682ms | ~80ms | 8.5× |
| Autograd overhead | 495ms (73%) | ~20ms (<20%) | 25× reduction |
| cos | 0.946 | ~0.97+ | +0.025 |
| Kernel launches/step | ~500+ | ~50 (via CUDA Graphs) | 10× |

**After all patches, the training step breakdown should be:**
- Teacher forward: 80ms (hidden behind student via stream overlap)
- Student forward: ~40ms (fused RMSNorm + Linear + MLP + attention)
- Backward: ~80ms (fused layer backward + LoRA backward + PalettizedLinear backward)
- Optimizer: ~20ms (bitsandbytes AdamW8bit, already done)
- CUDA Graph dispatch: ~2ms (single graph replay)
- **Total: ~120ms → tps ≈ 8**

---

## 7. Definition of Done (Per Agent)

Each agent is "done" when:
1. All assigned patches are implemented on their branch
2. All tests pass (`python3 -c "import ast; ast.parse(open('FILE').read())"` for syntax)
3. Branch pushes to GitHub
4. Inbox messages sent to dependent agents
5. `agent-ctx/PROGRESS.md` updated with status (append-only)

**Global DoD:** All 6 branches merged to main in dependency order. No conflicts. `train_qwen.py` runs without import errors. `tps > 3` (verified by 100-step training run).

---

## 8. Working Offline

**No training server access.** All work is offline code changes:
- Syntax check: `python3 -c "import ast; ast.parse(open('FILE').read())"`
- Import check: `python3 -c "import sys; sys.path.insert(0,'scripts'); import MODULE"`
- No GPU needed for code changes
- Triton kernel correctness tests require the server (deferred to when server is back)
- Each agent commits + pushes after each sub-task, pushes after each wave
