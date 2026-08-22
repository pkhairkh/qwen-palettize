# Global Progress Tracker

> **Updated by:** Orchestrator + each agent (append-only, never edit existing entries)

---

## Agent Status

| Agent | Branch | Wave 1 | Wave 2 | Wave 3 | Wave 4 | Merged |
|-------|--------|--------|--------|--------|--------|--------|
| nn-module-foundation | `agent/nn-module-foundation` | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ |
| triton-kernels | `agent/triton-kernels` | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ |
| layer-fusion | `agent/layer-fusion` | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ |
| lora-fusion | `agent/lora-fusion` | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ |
| cuda-graphs | `agent/cuda-graphs` | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ |
| quality-recipe | `agent/quality-recipe` | ✅ Done | 🔄 Coord Sent | ✅ Done | ⬜ Pending | ⬜ |

**Legend:** ⬜ Pending | 🔄 In Progress | ✅ Done | ❌ Blocked

---

## Merge Order

1. ⬜ nn-module-foundation (foundation — must merge first, unlocks torch.compile)
2. ⬜ triton-kernels (rebases on nn-module — optimized PalettizedLinear kernels)
3. ⬜ layer-fusion (rebases on nn-module + triton-kernels — fused RMSNorm/MLP/attention)
4. ⬜ lora-fusion (rebases on triton-kernels — fused LoRA backward)
5. ⬜ quality-recipe (rebases on nn-module — loss/clip/freeze changes)
6. ⬜ cuda-graphs (rebases on ALL — graph capture needs stable kernels)

---

## Patch Status

| # | Patch | Agent | Status | Branch | Commit |
|---|-------|-------|--------|--------|--------|
| 10 | Fused RMSNorm + Linear | layer-fusion | ⬜ | — | — |
| 11 | nn.Module forward signature | nn-module-foundation | ⬜ | — | — |
| 12 | Fused MLP (gate+up+SiLU+down) | layer-fusion | ⬜ | — | — |
| 13 | Fused Attention (FlashAttention-style) | layer-fusion | ⬜ | — | — |
| 14 | Fused GatedDeltaNet (conv1d + delta-rule) | layer-fusion | ⬜ | — | — |
| 15 | Batched compute_P_W (Triton, 25→1) | triton-kernels | ⬜ | — | — |
| 16 | Eliminate redundant matmul in STE forward | triton-kernels | ⬜ | — | — |
| 17 | Buffer pooling for P_aos + W_ste | triton-kernels | ⬜ | — | — |
| 18 | Chunked reduction for elementwise backward | triton-kernels | ⬜ | — | — |
| 19 | Fused LoRA backward | lora-fusion | ⬜ | — | — |
| 20 | Fused LoRA + PalettizedLinear backward | lora-fusion | ⬜ | — | — |
| 21 | CUDA Graph capture for full step | cuda-graphs | ⬜ | — | — |
| 22 | Stream double-buffer + CUDA Graphs integration | cuda-graphs | ⬜ | — | — |
| 23 | Loss config switch (1-cos+norm_mse, 80/20) | quality-recipe | ✅ | `agent/quality-recipe` | `d00e7f6` |
| 24 | Per-group gradient clipping | quality-recipe | ✅ | `agent/quality-recipe` | `b5828a4` |
| 25 | LUT-Q re-quantization (step 2000, 4000) | quality-recipe | ✅ | `agent/quality-recipe` | `1d84a60` |
| 26 | Deterministic-ST (remove Gumbel noise) | quality-recipe | 🔄 (coord sent) | `agent/quality-recipe` | inbox msg → triton-kernels 2026-08-23T02:00:00Z; awaiting kernel-side change |

---

## Event Log (append-only)

| Timestamp | Agent | Event |
|-----------|-------|-------|
| 2026-08-23T00:00:00Z | orchestrator | Created Round 3 multi-agent infrastructure for full Triton fusion. 6 agents, 17 patches (P10-P26), 4 waves. Previous Round 1/2 patches (P1-P9) already merged to main. Current state: tps=0.6, backward=682ms (73% autograd overhead). Target: tps>3, backward<100ms. All work OFFLINE (no server). |
| 2026-08-23T01:30:00Z | quality-recipe | Wave 1 complete: Patch 23 (loss config 1-cos+norm_mse cos=0.8 mse=0.2, commit d00e7f6) + Patch 24 (per-group clip indices=1.0 others=0.3, commit b5828a4 — comment update only; functional code already in place from prior round commit 5446edf). τ schedule (lines 1176-1192) verified to already match recommended warmup+quadratic decay+hold pattern from research-indices-training/07_recommendations.md Fix 1 — no action needed. Branch pushed. Wave 2 (Patch 26 deterministic-ST) pending: requires inbox coordination with triton-kernels to remove Gumbel noise from compute_P_W_ste_kernel in triton_soft_forward.py. |
| 2026-08-23T02:00:00Z | quality-recipe | Wave 2 Patch 26 coordination: sent inbox message to triton-kernels requesting removal of Gumbel noise from compute_P_W_ste_kernel in triton_soft_forward.py (6 specific changes listed: remove 4 Gumbel noise lines, remove step_seed from kernel sig + Python launcher, remove _gumbel_sample function, remove _next_soft_step_seed + _SOFT_STEP_SEED, update TritonSoftLinear.forward docstring). Verified train_qwen.py has NO step_seed references — no changes needed on my side (qwen_model.py already calls triton_soft_linear without step_seed; public API has never exposed it). Also flagged that triton-kernels' Patch 15 (batched compute_P_W) plan in their TASKS.md mentions base_seed for Gumbel decorrelation — needs to be dropped since there's no Gumbel anymore. Test files needing updates (test_triton_soft_forward.py, test_triton_soft_backward.py, bench_triton_kernels.py, test_batched_compute_pw.py, test_profile_kernels.py) listed for triton-kernels. Awaiting their confirmation reply in quality-recipe inbox. |
| 2026-08-23T02:30:00Z | quality-recipe | Wave 2 closeout: PROGRESS.md update. Wave 2 DoD satisfied: (a) Patch 26 coordination sent to triton-kernels with precise spec; (b) train_qwen.py verified clean of step_seed references (no caller changes needed); (c) kernel-side change deferred to triton-kernels (their file ownership). Branch pushed. Wave 3 (Patch 25 LUT-Q re-quantization) starting. |
| 2026-08-23T03:30:00Z | quality-recipe | Wave 3 complete: Patch 25 LUT-Q re-quantization (commit 1d84a60). Created NEW scripts/re_quantize.py (358 lines) with `re_quantize_indices(model, sb_idx)` function that walks all PalettizedLinear modules, reconstructs W_recon from current palette + effective indices (argmax(index_logits) in soft path, indices buffer in hard path), re-runs 1D weighted k-means per group with uniform weights, and updates palette + indices + indices_int8 + _flat_idx + index_logits (re-init as ±3 one-hot — NOT ±10, better gradient flow per Finding 12). Wired call site into train_qwen.py right after the τ schedule block: `if global_step in (2000, 4000) and use_soft_indices: re_quantize_indices(student, sb_idx)`. Also added scripts/test_re_quantize_smoke.py — 3 CPU-runnable test cases (basic soft path, hard path, mixed wrapper with non-PalettizedLinear module) — ALL PASS. Discovered pre-existing bug in palettize_pytorch.kmeans1d_weighted (line 63: argmin(dim=0) should be dim=1 — cdist returns (N,k), argmin along dim 0 returns (k,) instead of (N,), causes IndexError on downstream weights[mask]). Vendored a corrected `_kmeans1d_weighted_local` inside re_quantize.py rather than touching the un-owned palettize_pytorch.py (verified no other agent has modified it — only commit 5446edf touches it across all branches). Documented the upstream bug for the orchestrator to fix separately (one-line change). Installed CPU-only torch to verify the DoD import check: `python3 -c "import sys; sys.path.insert(0,'scripts'); import re_quantize"` PASSES. All 3 quality-recipe waves complete: Patches 23 (d00e7f6), 24 (b5828a4), 25 (1d84a60) committed; Patch 26 coordination sent to triton-kernels (55e816a) — awaiting their kernel-side change. |

---

## Inbox Summary (last message per agent)

| Agent | Last Message | From | Subject | Action Required |
|-------|--------------|------|---------|-----------------|
| nn-module-foundation | — | — | — | — |
| triton-kernels | 2026-08-23T02:00:00Z | quality-recipe | Patch 26 deterministic-ST: remove Gumbel noise from compute_P_W_ste_kernel | Apply 6 changes to triton_soft_forward.py + update test files; reply in quality-recipe inbox when done |
| layer-fusion | — | — | — | — |
| lora-fusion | — | — | — | — |
| cuda-graphs | — | — | — | — |
| quality-recipe | 2026-08-23T03:30:00Z | orchestrator (startup) | Round 3 infrastructure ready — quality patches span Waves 1-3 | All 3 waves DONE (Patches 23+24+25 committed; Patch 26 coordination sent to triton-kernels, awaiting reply) |
