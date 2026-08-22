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
| quality-recipe | `agent/quality-recipe` | ✅ Done | 🔄 In Progress | ⬜ Pending | ⬜ Pending | ⬜ |

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
| 25 | LUT-Q re-quantization (step 2000, 4000) | quality-recipe | ⬜ | — | — |
| 26 | Deterministic-ST (remove Gumbel noise) | quality-recipe | 🔄 | `agent/quality-recipe` | (pending coordination w/ triton-kernels) |

---

## Event Log (append-only)

| Timestamp | Agent | Event |
|-----------|-------|-------|
| 2026-08-23T00:00:00Z | orchestrator | Created Round 3 multi-agent infrastructure for full Triton fusion. 6 agents, 17 patches (P10-P26), 4 waves. Previous Round 1/2 patches (P1-P9) already merged to main. Current state: tps=0.6, backward=682ms (73% autograd overhead). Target: tps>3, backward<100ms. All work OFFLINE (no server). |
| 2026-08-23T01:30:00Z | quality-recipe | Wave 1 complete: Patch 23 (loss config 1-cos+norm_mse cos=0.8 mse=0.2, commit d00e7f6) + Patch 24 (per-group clip indices=1.0 others=0.3, commit b5828a4 — comment update only; functional code already in place from prior round commit 5446edf). τ schedule (lines 1176-1192) verified to already match recommended warmup+quadratic decay+hold pattern from research-indices-training/07_recommendations.md Fix 1 — no action needed. Branch pushed. Wave 2 (Patch 26 deterministic-ST) pending: requires inbox coordination with triton-kernels to remove Gumbel noise from compute_P_W_ste_kernel in triton_soft_forward.py. |

---

## Inbox Summary (last message per agent)

| Agent | Last Message | From | Subject | Action Required |
|-------|--------------|------|---------|-----------------|
| nn-module-foundation | — | — | — | — |
| triton-kernels | — | — | — | — |
| layer-fusion | — | — | — | — |
| lora-fusion | — | — | — | — |
| cuda-graphs | — | — | — | — |
| quality-recipe | 2026-08-23T01:30:00Z | orchestrator (startup) | Round 3 infrastructure ready — quality patches span Waves 1-3 | Wave 1 complete; Wave 2 starting (Patch 26 coordination w/ triton-kernels) |
