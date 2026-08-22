# Global Progress Tracker

> **Updated by:** Orchestrator + each agent (append-only, never edit existing entries)

---

## Agent Status

| Agent | Branch | Wave 1 | Wave 2 | Wave 3 | Wave 4 | Merged |
|-------|--------|--------|--------|--------|--------|--------|
| nn-module-foundation | `agent/nn-module-foundation` | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ |
| triton-kernels | `agent/triton-kernels` | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ |
| layer-fusion | `agent/layer-fusion` | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ |
| lora-fusion | `agent/lora-fusion` | ⬜ Pending | ⬜ Pending | ✅ Done | ✅ Done | ⬜ |
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
| 11 | nn.Module forward signature | nn-module-foundation | ⬜ | — | — |
| 12 | Fused MLP (gate+up+SiLU+down) | layer-fusion | ⬜ | — | — |
| 13 | Fused Attention (FlashAttention-style) | layer-fusion | ⬜ | — | — |
| 14 | Fused GatedDeltaNet (conv1d + delta-rule) | layer-fusion | ⬜ | — | — |
| 15 | Batched compute_P_W (Triton, 25→1) | triton-kernels | ⬜ | — | — |
| 16 | Eliminate redundant matmul in STE forward | triton-kernels | ⬜ | — | — |
| 17 | Buffer pooling for P_aos + W_ste | triton-kernels | ⬜ | — | — |
| 18 | Chunked reduction for elementwise backward | triton-kernels | ⬜ | — | — |
| 19 | Fused LoRA backward | lora-fusion | ✅ | agent/lora-fusion | 96f69a1 + d28c2ed |
| 20 | Fused LoRA + PalettizedLinear backward | lora-fusion | ✅ | agent/lora-fusion | 897a361 |
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
| 2026-08-23T00:30:00Z | lora-fusion | Wave 3 Patch 19 done: scripts/triton_lora.py (NEW, 722 LOC) with 6 fused Triton kernels (xA, matmul, grad_xA, grad_A, grad_B, grad_x) + TritonLoRALinear.autograd.Function. QwenLoRA.forward wired to use triton_lora.triton_lora_forward() (with PyTorch fallback). Eliminates per step: 31 aten::mul for scaling, 93 separate matmul dispatches for grad_A/grad_B/grad_x_lora, 31 autograd graph node traversals. Branch pushed (96f69a1 + d28c2ed). Inbox msg sent to cuda-graphs. NOT yet eliminated: 31 aten::add for grad_x accumulation (Patch 20). |
| 2026-08-23T01:00:00Z | lora-fusion | Wave 4 Patch 20 done: added fused_pl_lora_bwd_grad_x_kernel + FusedPLLoRALinear.autograd.Function to triton_lora.py (413 LOC extension). Combined grad_x = grad_y @ (W_ste + lora_B @ lora_A.T * scaling).T - single matmul eliminating 31 aten::add_ per step (167ms backward overhead). QwenLoRA.forward wired to use fused_pl_lora_forward when self.base is PalettizedLinear with Triton soft path + soft indices active. All Wave 4 DoD met: syntax+import OK, fused LoRA backward committed, fused PL+LoRA backward committed, QwenLoRA.forward uses Triton fused kernel. Branch pushed (897a361). Inbox msg sent to cuda-graphs (Patch 20 done, ready for CUDA Graph capture). lora-fusion agent DONE - all assigned patches (19+20) complete. |

---

## Inbox Summary (last message per agent)

| Agent | Last Message | From | Subject | Action Required |
|-------|--------------|------|---------|-----------------|
| nn-module-foundation | — | — | — | — |
| triton-kernels | — | — | — | — |
| layer-fusion | — | — | — | — |
| lora-fusion | — | — | — | — |
| cuda-graphs | 1724371400-from-lora-fusion | lora-fusion | Patch 20 done - all LoRA+PL backward fused, ready for CUDA Graph capture | None (informational) |
| quality-recipe | — | — | — | — |
