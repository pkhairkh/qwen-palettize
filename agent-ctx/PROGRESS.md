# Global Progress Tracker

> **Updated by:** Orchestrator + each agent (append-only, never edit existing entries)

---

## Agent Status

| Agent | Branch | Wave 1 | Wave 2 | Wave 3 | Wave 4 | Merged |
|-------|--------|--------|--------|--------|--------|--------|
| nn-module-foundation | `agent/nn-module-foundation` | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ |
| triton-kernels | `agent/triton-kernels` | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ |
| layer-fusion | `agent/layer-fusion` | ⬜ N/A | 🔄 In Progress (P10 ✅, P12 ✅) | ⬜ Pending | ⬜ Pending | ⬜ |
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
| 10 | Fused RMSNorm + Linear | layer-fusion | ✅ | agent/layer-fusion | 8054af7 |
| 11 | nn.Module forward signature | nn-module-foundation | ⬜ | — | — |
| 12 | Fused MLP (gate+up+SiLU+down) | layer-fusion | ✅ | agent/layer-fusion | f25bc91 |
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
| 2026-08-23T01:00:00Z | layer-fusion | Patch 10 (fused RMSNorm + PalettizedLinear) committed at 8054af7. Three Triton kernels: rmsnorm_forward, rmsnorm_backward, fused_rmsnorm_matmul (autotuned). FusedRMSNormLinear autograd Function — fuses RMSNorm into the matmul (eliminates x_normed HBM round-trip). Imports OK. |
| 2026-08-23T01:30:00Z | layer-fusion | Patch 12 (fused SwiGLU MLP) committed at f25bc91. Three Triton kernels: fused_silu_mul, fused_silu_mul_backward, fused_dual_grad_x (autotuned). FusedMLP autograd Function — fuses gate*SiLU(up) elementwise + dual grad_x accumulation (eliminates 1 aten::add_ per MLP). Imports OK. Wave 2 closeout: 2/2 patches done. |

---

## Inbox Summary (last message per agent)

| Agent | Last Message | From | Subject | Action Required |
|-------|--------------|------|---------|-----------------|
| nn-module-foundation | — | — | — | — |
| triton-kernels | — | — | — | — |
| layer-fusion | 2026-08-23T01:30:00Z | layer-fusion (self) | Wave 2 closeout (P10 + P12 done) | Wave 3 starts after Patch 15 API stable |
| lora-fusion | — | — | — | — |
| cuda-graphs | 2026-08-23T01:30:00Z | layer-fusion | Wave 2 closeout — P10 + P12 ready for graph capture | Wait for Wave 3 (P13+P14) before full layer graph capture |
| quality-recipe | — | — | — | — |
