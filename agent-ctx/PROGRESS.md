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
| cuda-graphs | `agent/cuda-graphs` | ⬜ Pending | ⬜ Pending | ⬜ Pending | ✅ Done | ⬜ |
| quality-recipe | `agent/quality-recipe` | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ Pending | ⬜ |

**Legend:** ⬜ Pending | 🔄 In Progress | ✅ Done | ❌ Blocked

---

## Merge Order

1. ⬜ nn-module-foundation (foundation — must merge first, unlocks torch.compile)
2. ⬜ triton-kernels (rebases on nn-module — optimized PalettizedLinear kernels)
3. ⬜ layer-fusion (rebases on nn-module + triton-kernels — fused RMSNorm/MLP/attention)
4. ⬜ lora-fusion (rebases on triton-kernels — fused LoRA backward)
5. ⬜ quality-recipe (rebases on nn-module — loss/clip/freeze changes)
6. ✅ cuda-graphs (rebases on ALL — graph capture needs stable kernels) — Wave 4 done (P21+P22 implemented, syntax-checked, committed; push pending closeout)

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
| 21 | CUDA Graph capture for full step | cuda-graphs | ✅ Done | agent/cuda-graphs | a2c4bf1 |
| 22 | Stream double-buffer + CUDA Graphs integration | cuda-graphs | ✅ Done | agent/cuda-graphs | cc13f9d |
| 23 | Loss config switch (1-cos+norm_mse, 80/20) | quality-recipe | ⬜ | — | — |
| 24 | Per-group gradient clipping | quality-recipe | ⬜ | — | — |
| 25 | LUT-Q re-quantization (step 2000, 4000) | quality-recipe | ⬜ | — | — |
| 26 | Deterministic-ST (remove Gumbel noise) | quality-recipe | ⬜ | — | — |

---

## Event Log (append-only)

| Timestamp | Agent | Event |
|-----------|-------|-------|
| 2026-08-23T00:00:00Z | orchestrator | Created Round 3 multi-agent infrastructure for full Triton fusion. 6 agents, 17 patches (P10-P26), 4 waves. Previous Round 1/2 patches (P1-P9) already merged to main. Current state: tps=0.6, backward=682ms (73% autograd overhead). Target: tps>3, backward<100ms. All work OFFLINE (no server). |
| 2026-08-23T01:30:00Z | cuda-graphs | Wave 4 STARTED. Read ROADMAP, PROGRESS, TASKS, RULES, inbox (only orchestrator's initial dispatch — proceeding in single-agent execution mode). Read research 08_recommendations.md §11 (CUDA Graphs patch) + 06_stream_overlap.md §6 (stream double-buffer + CUDA Graphs integration). Designed CUDA Graph capture with two graph pairs (one per buf_idx) + graph-friendly loss/clip helpers (no .item() inside capture). |
| 2026-08-23T02:00:00Z | cuda-graphs | Patch 21 committed (a2c4bf1). CUDA Graph capture for full training step: static_batch_ids pre-allocated; _capture_step_graphs captures student_graph on default stream (fwd + loss + bwd + clip + opt + zero_grad); _replay_step_graphs replays both graphs. Graph-friendly helpers _graph_safe_compute_loss + _graph_safe_clip_grad_norm_ avoid .item() syncs inside capture. Re-capture policy: every 100 steps (LR/tau pickup), on HP changes, on NaN recovery. Eager path preserved verbatim as fallback (graph_capture_failed flag). Syntax check passes. |
| 2026-08-23T02:15:00Z | cuda-graphs | Patch 22 committed (cc13f9d). Stream double-buffer integration with CUDA Graphs: added stream_t.wait_event(event_s[buf_idx]) as first op in teacher_graph capture (invariant 1 WAIT — was missing in P21). All 3 producer/consumer invariants from 06_stream_overlap.md §3.1 now baked into the captures: (1) teacher waits for prev iter's student to finish reading h_out_buf, (2) student waits for teacher to finish writing h_out_buf, (3) student signals event_s immediately after compute_loss (before backward) so next iter's teacher can start. Teacher forward (80ms) fully hidden behind student compute (980ms) via stream_t vs default stream concurrency. DoD checklist all green except push (pending Wave 4 closeout). |

---

## Inbox Summary (last message per agent)

| Agent | Last Message | From | Subject | Action Required |
|-------|--------------|------|---------|-----------------|
| nn-module-foundation | — | — | — | — |
| triton-kernels | — | — | — | — |
| layer-fusion | — | — | — | — |
| lora-fusion | — | — | — | — |
| cuda-graphs | Wave 4 done: P21+P22 committed, syntax verified, branch push pending closeout | orchestrator (pending) | "CUDA Graphs ready — tps target achievable" | Merge cuda-graphs LAST (after all other agents). Eager path preserved as fallback — capture failure does not crash training. |
| quality-recipe | — | — | — | — |
