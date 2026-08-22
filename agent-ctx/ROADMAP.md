# Holistic Roadmap: Transform Before Training Restart

> **Status:** PRE-IMPLEMENTATION. The current codebase (cos=0.9530) is capped by 6 structural deficiencies identified in 568 pages of research. This roadmap defines 4 parallel agents that will deliver 9 patches across 4 branches, then merge in dependency order.

---

## 1. Why Current State is Capped at 0.9530

| # | Deficiency | Source | Patch(es) |
|---|------------|--------|-----------|
| 1 | τ anneals to 0.1, killing Gumbel-Softmax gradients | [`research-indices-training/04_tau_schedule.md`](../research-indices-training/04_tau_schedule.md) | 1 |
| 2 | LoRA zero-init (LoftQ SVD skipped, `original_weight=None`) | [`research-kernel-accuracy/00_overview.md`](../research-kernel-accuracy/00_overview.md) | 2 |
| 3 | Logit clamp ±20 saturates softmax at low τ | [`research-filter-consolidation/01_training_recipe.md`](../research-filter-consolidation/01_training_recipe.md) §3 | 3 |
| 4 | GROUP_SIZE=256 too coarse for k-means fit | [`research-filter-consolidation/01_training_recipe.md`](../research-filter-consolidation/01_training_recipe.md) §4 | 4 |
| 5 | Fused backward kernel 10× slower (strided P access) | [`research-kernel-efficiency/02_fused_bwd_fix.md`](../research-kernel-efficiency/02_fused_bwd_fix.md) | 5 |
| 6 | No stream overlap — teacher fwd blocks student bwd | [`research-kernel-efficiency/06_stream_overlap.md`](../research-kernel-efficiency/06_stream_overlap.md) | 6 |
| 7 | 25 compute_P_W launches per forward (125µs overhead) | [`research-kernel-efficiency/03_batched_compute_pw.md`](../research-kernel-efficiency/03_batched_compute_pw.md) | 7 |
| 8 | FP32MasterAdamW = 113ms/step, 21.4GB VRAM | [`research-filter-consolidation/03_optimizer_speedup.md`](../research-filter-consolidation/03_optimizer_speedup.md) §8 | 8 |
| 9 | PartialWrapper blocks torch.compile + checkpointing | [`research-architecture-review/02_partial_wrapper_problem.md`](../research-architecture-review/02_partial_wrapper_problem.md) | 9 |

**Expected after all 9 patches:** cos 0.9530 → 0.978-0.985+, step 530ms → ~200ms, VRAM 36GB → ~80GB (batch=64 enabled).

---

## 2. Agent Roster (4 agents, 4 branches)

| Agent | Branch | Patches | Files Owned (EXCLUSIVE) | Foundation? |
|-------|--------|---------|-------------------------|-------------|
| `training-recipe` | `agent/training-recipe` | 1, 3, 4 | `palettize_core.py` (full), `train_qwen.py` (τ anneal + clamp section only, ~lines 1034-1153) | No |
| `nn-module-foundation` | `agent/nn-module-foundation` | 9, 2 | `qwen_model.py` (PartialModel/PartialWrapper + new capture helper), `train_qwen.py` (build_student_super_block section only, ~lines 632-736) | **YES — merge first** |
| `kernels` | `agent/kernels` | 5, 7 | `fused_lut_kernel.cu` (full), `fused_lut_linear_cuda.py` (full) | No (independent) |
| `optimizer-streams` | `agent/optimizer-streams` | 8, 6 | `train_qwen.py` (build_optimizers section + training loop stream section only, ~lines 540-600 + 1058-1103) | Depends on nn-module |

### File Conflict Map

```
train_qwen.py line ranges (EXCLUSIVE ownership):
  540-600:  optimizer-streams (Patch 8 — fused AdamW)
  632-736:  nn-module-foundation (Patch 2 — LoftQ + Patch 9 build_student)
  960-980:  nn-module-foundation (scheduler init, if affected by nn.Module)
  1034-1040: training-recipe (Patch 1 — τ schedule)
  1058-1103: optimizer-streams (Patch 6 — stream double-buffer)
  1140-1160: training-recipe (Patch 3 — logit clamp)

qwen_model.py: nn-module-foundation ONLY
fused_lut_kernel.cu: kernels ONLY
fused_lut_linear_cuda.py: kernels ONLY
palettize_core.py: training-recipe ONLY
```

**Rule:** If two agents need the SAME line range in `train_qwen.py` in the SAME wave, the second agent MUST wait (check inbox for "lock released" message).

---

## 3. Merge Order (Dependency Graph)

```
Wave 1 (parallel):
  nn-module-foundation  ──→  Patch 9 (PartialWrapper→nn.Module)
  kernels               ──→  Patch 5 (fused bwd AoS P layout)
  training-recipe       ──→  Patch 1 (τ schedule) — can start without nn.Module
  optimizer-streams     ──→  Patch 8 (fused AdamW) — can start without nn.Module

Wave 2 (parallel, after Wave 1 merges):
  nn-module-foundation  ──→  Patch 2 (LoftQ SVD init) — needs nn.Module
  kernels               ──→  Patch 7 (batched compute_P_W) — needs Patch 5
  training-recipe       ──→  Patch 3 (logit clamp) — needs Patch 1
  optimizer-streams     ──→  Patch 6 (stream double-buffer) — needs Patch 8

Wave 3 (merge coordination):
  All agents verify their branch merges cleanly with main
  Orchestrator merges in order: nn-module → training-recipe → kernels → optimizer-streams
```

**Critical:** `nn-module-foundation` MUST merge first. Patches 2, 6, 8 depend on `nn.Module` (for torch.compile, state_dict, gradient checkpointing).

---

## 4. Inbox Communication Protocol

Each agent has an inbox: `agent-ctx/agent-{name}/inbox/`

**Message format:** `{unix-timestamp}-from-{sender-agent-name}.md`

**Example:** `agent-ctx/agent-training-recipe/inbox/1740235678-from-nn-module-foundation.md`

**Message body template:**
```markdown
# Message: {subject}

**TO:** training-recipe
**FROM:** nn-module-foundation
**TIMESTAMP:** 2025-02-22T14:34:38Z
**SUBJECT:** nn.Module refactor merged — you can rebase

{body — what changed, what you need to do}

**ACTION REQUIRED:** {rebase / wait / coordinate / nothing}
```

### Rules

1. **Check inbox at start of EVERY wave.** Process all messages before starting work.
2. **Send messages BEFORE merging** to warn others of upcoming changes.
3. **Send "lock released" messages** after merging to main, so blocked agents can proceed.
4. **Use inbox for conflict resolution** — never modify another agent's branch directly.
5. **Orchestrator (you) reads all inboxes** to track coordination state.

### Coordination Triggers

| Event | Sender | Recipient(s) | Message Subject |
|-------|--------|--------------|-----------------|
| Start editing train_qwen.py lines 632-736 | nn-module-foundation | training-recipe, optimizer-streams | "LOCK: train_qwen.py:632-736 for Wave 1" |
| Merged Patch 9 to main | nn-module-foundation | ALL | "RELEASED: nn.Module merged — rebase your branches" |
| Need to edit train_qwen.py:1140-1160 | training-recipe | (none — exclusive) | (no message needed) |
| Fused bwd kernel changed P layout | kernels | training-recipe | "P layout changed to (K,N,4) — palettize_core.py unaffected" |

---

## 5. Patch-to-Research-to-Paper Cross-Reference

| Patch | Research File | Paper (in `docs/papers/`) |
|-------|---------------|---------------------------|
| 1 (τ schedule) | `research-indices-training/04_tau_schedule.md` | `1611.01144_Gumbel-Softmax_Jang2017.pdf`, `LLT_Wang_CVPR2022.pdf` |
| 2 (LoftQ init) | `research-kernel-accuracy/00_overview.md` | `2305.14314_QLoRA_Dettmers2023.pdf` (LoRA + quant) |
| 3 (logit clamp) | `research-filter-consolidation/01_training_recipe.md` §3 | `1602.02830_BNN_Courbariaux2016.pdf` (tight clamp) |
| 4 (group size) | `research-filter-consolidation/01_training_recipe.md` §4 | `2210.17323_GPTQ_Frantar2023.pdf` (GS=128 standard) |
| 5 (fused bwd AoS) | `research-kernel-efficiency/02_fused_bwd_fix.md` | (kernel optimization, no specific paper) |
| 6 (stream double-buffer) | `research-kernel-efficiency/06_stream_overlap.md` | (CUDA streams, no specific paper) |
| 7 (batched compute_P_W) | `research-kernel-efficiency/03_batched_compute_pw.md` | (kernel fusion, no specific paper) |
| 8 (fused AdamW) | `research-filter-consolidation/03_optimizer_speedup.md` §8 | `1412.6980_Adam_Kingma2015.pdf`, `1711.05101_AdamW_Loshchilov2019.pdf` |
| 9 (nn.Module) | `research-architecture-review/02_partial_wrapper_problem.md` | `2305.14314_QLoRA_Dettmers2023.pdf` (uses nn.Module) |

---

## 6. Definition of Done (Per Agent)

Each agent is "done" when:
1. All assigned patches are implemented on their branch
2. All tests pass (correctness: `python3 -c "import ast; ast.parse(open('FILE').read())"` for syntax)
3. Branch pushes to GitHub
4. Inbox messages sent to dependent agents
5. PROGRESS.md updated with status

**Global DoD:** All 4 branches merged to main in dependency order. No conflicts. `train_qwen.py` runs without import errors.
