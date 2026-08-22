# Global Progress Tracker

> **Updated by:** Orchestrator + each agent (append-only, never edit existing entries)

---

## Agent Status

| Agent | Branch | Wave 1 | Wave 2 | Wave 3 | Wave 4 | Merged |
|-------|--------|--------|--------|--------|--------|--------|
| nn-module-foundation | `agent/nn-module-foundation` | 🔄 In Progress | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ |
| triton-kernels | `agent/triton-kernels` | ✅ Done | ✅ Done | ⬜ Pending | ⬜ Pending | ⬜ |
| layer-fusion | `agent/layer-fusion` | 🔄 In Progress | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ |
| lora-fusion | `agent/lora-fusion` | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ |
| cuda-graphs | `agent/cuda-graphs` | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ |
| quality-recipe | `agent/quality-recipe` | 🔄 In Progress | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ |

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
| 11 | nn.Module forward signature | nn-module-foundation | 🔄 | agent/nn-module-foundation | 970f5ad |
| 12 | Fused MLP (gate+up+SiLU+down) | layer-fusion | ⬜ | — | — |
| 13 | Fused Attention (FlashAttention-style) | layer-fusion | ⬜ | — | — |
| 14 | Fused GatedDeltaNet (conv1d + delta-rule) | layer-fusion | ⬜ | — | — |
| 15 | Batched compute_P_W (Triton, 25→1) | triton-kernels | ✅ | agent/triton-kernels | 01c0ecf |
| 16 | Eliminate redundant matmul in STE forward | triton-kernels | ✅ | agent/triton-kernels | 44530fc |
| 17 | Buffer pooling for P_aos + W_ste | triton-kernels | ✅ | agent/triton-kernels | 56ca124 |
| 18 | Chunked reduction for elementwise backward | triton-kernels | ✅ | agent/triton-kernels | cb242e4 |
| 19 | Fused LoRA backward | lora-fusion | ⬜ | — | — |
| 20 | Fused LoRA + PalettizedLinear backward | lora-fusion | ⬜ | — | — |
| 21 | CUDA Graph capture for full step | cuda-graphs | ⬜ | — | — |
| 22 | Stream double-buffer + CUDA Graphs integration | cuda-graphs | ⬜ | — | — |
| 23 | Loss config switch (1-cos+norm_mse, 80/20) | quality-recipe | ✅ | agent/quality-recipe | d00e7f6 |
| 24 | Per-group gradient clipping | quality-recipe | ✅ | agent/quality-recipe | b5828a4 |
| 25 | LUT-Q re-quantization (step 2000, 4000) | quality-recipe | ⬜ | — | — |
| 26 | Deterministic-ST (remove Gumbel noise) | quality-recipe | 🔄 | agent/quality-recipe | 55e816a |

---

## Event Log (append-only)

| Timestamp | Agent | Event |
|-----------|-------|-------|
| 2026-08-23T00:00:00Z | orchestrator | Created Round 3 multi-agent infrastructure for full Triton fusion. 6 agents, 17 patches (P10-P26), 4 waves. Previous Round 1/2 patches (P1-P9) already merged to main. Current state: tps=0.6, backward=682ms (73% autograd overhead). Target: tps>3, backward<100ms. All work OFFLINE (no server). |
| 2026-08-23T13:00:29Z | triton-kernels | Wave 1 complete: Patch 16 (commit 44530fc — eliminated redundant W_soft store in compute_P_W_ste_kernel) + Patch 17 (commit 56ca124 — buffer pooling for P_aos + W_ste + grad_W via module-level _P_POOL / _BWD_POOL dicts). Inbox messages sent to layer-fusion + lora-fusion with the new API contract (compute_P_W_ste_triton now returns 2-tuple, not 3-tuple). Syntax + import checks pass. Beginning Wave 2 (Patches 15, 18). |
| 2026-08-23T13:16:00Z | triton-kernels | Wave 2 complete: Patch 15 (commit 01c0ecf — batched compute_P_W_ste_kernel, 25 layers → 1 launch via blockIdx.z + per-layer pointer/shape arrays) + Patch 18 (commit cb242e4 — fused_soft_bwd_chunked_kernel, eliminates grad_W HBM intermediate by computing grad_W via tl.dot and consuming it immediately for grad_logits + grad_palette). TritonSoftLinear.backward now uses the chunked kernel. Old two-kernel split kept for backward compat. Inbox message sent to lora-fusion with Patch 18 API changes + recommended Patch 20 design (compute grad_W internally via tl.dot). All DoD criteria verified: syntax + import checks pass, no redundant matmul, buffer pooling implemented, batched compute_P_W implemented, chunked reduction implemented. Branch ready for merge. |

---

## Inbox Summary (last message per agent)

| Agent | Last Message | From | Subject | Action Required |
|-------|--------------|------|---------|-----------------|
| nn-module-foundation | — | — | — | — |
| triton-kernels | 2026-08-23T13:16:00Z | (self) | Wave 2 done | All 4 patches complete; branch ready for merge |
| layer-fusion | 2026-08-23T13:00:29Z | triton-kernels | Wave 1 done — API stable | Rebase before Wave 2; use 2-tuple return |
| lora-fusion | 2026-08-23T13:16:00Z | triton-kernels | Wave 2 done — Patch 18 API changes | Rebase before Wave 3; compute grad_W internally in Patch 20 |
| cuda-graphs | — | — | — | — |
| quality-recipe | — | — | — | — |
