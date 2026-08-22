# Global Progress Tracker

> **Updated by:** Orchestrator + each agent (append-only, never edit existing entries)

---

## Agent Status

| Agent | Branch | Wave 1 | Wave 2 | Wave 3 | Wave 4 | Merged |
|-------|--------|--------|--------|--------|--------|--------|
| nn-module-foundation | `agent/nn-module-foundation` | ✅ Done | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ |
| triton-kernels | `agent/triton-kernels` | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ |
| layer-fusion | `agent/layer-fusion` | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ |
| lora-fusion | `agent/lora-fusion` | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ |
| cuda-graphs | `agent/cuda-graphs` | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ |
| quality-recipe | `agent/quality-recipe` | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ |

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
| 11 | nn.Module forward signature | nn-module-foundation | ✅ Done | `agent/nn-module-foundation` | 970f5ad (11a) + e3d9f90 (11b) + c516401 (11c) + Wave 1 closeout |
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
| 23 | Loss config switch (1-cos+norm_mse, 80/20) | quality-recipe | ⬜ | — | — |
| 24 | Per-group gradient clipping | quality-recipe | ⬜ | — | — |
| 25 | LUT-Q re-quantization (step 2000, 4000) | quality-recipe | ⬜ | — | — |
| 26 | Deterministic-ST (remove Gumbel noise) | quality-recipe | ⬜ | — | — |

---

## Event Log (append-only)

| Timestamp | Agent | Event |
|-----------|-------|-------|
| 2026-08-23T00:00:00Z | orchestrator | Created Round 3 multi-agent infrastructure for full Triton fusion. 6 agents, 17 patches (P10-P26), 4 waves. Previous Round 1/2 patches (P1-P9) already merged to main. Current state: tps=0.6, backward=682ms (73% autograd overhead). Target: tps>3, backward<100ms. All work OFFLINE (no server). |
| 2026-08-23T13:16:36Z | nn-module-foundation | Wave 1 complete. Patch 11 (nn.Module forward signature for fused RMSNorm) — 4 sub-tasks committed: 11a (add `out_norm` param to PalettizedLinear.forward + eager RMSNorm fallback helper `_apply_rmsnorm_eager`), 11b (verify PartialModel/PartialWrapper are nn.Module subclasses with nn.ModuleList delegation — already done in Patch 9), 11c (verify `build_student_super_block` torch.compile-compatible — no data-dependent control flow, all branches on module attrs / tensor metadata), 11d (this entry + inbox message to layer-fusion). All DoD syntax checks PASS via ast.parse. Branch `agent/nn-module-foundation` pushed. Inbox message sent to layer-fusion: "Patch 11 done — forward signature ready for fused RMSNorm (Patch 10)". |

---

## Inbox Summary (last message per agent)

| Agent | Last Message | From | Subject | Action Required |
|-------|--------------|------|---------|-----------------|
| nn-module-foundation | — | — | — | — |
| triton-kernels | — | — | — | — |
| layer-fusion | 1787433396-from-nn-module-foundation.md | nn-module-foundation | Patch 11 done — forward signature ready for fused RMSNorm (Patch 10) | Start Patch 10: extend `triton_soft_linear` / `triton_hard_linear` to accept `out_norm` and fuse RMSNorm into matmul prologue. See inbox message for exact API + integration steps. |
| lora-fusion | — | — | — | — |
| cuda-graphs | — | — | — | — |
| quality-recipe | — | — | — | — |
