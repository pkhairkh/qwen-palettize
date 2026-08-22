# RULES — agent-training-recipe

> **Branch:** `agent/training-recipe`
> **Patches:** 1 (τ schedule), 3 (logit clamp), 4 (group size)
> **Depends on:** nn-module-foundation (Patch 9) for clean merge — but can start Wave 1 in parallel.

---

## File Ownership (EXCLUSIVE)

- `scripts/palettize_core.py` — FULL ownership (Patch 4: GROUP_SIZE constant)
- `scripts/train_qwen.py` — ONLY these line ranges:
  - τ anneal logic: ~lines 1034-1040 (Patch 1)
  - CLI defaults for τ: ~lines 1241-1246 (Patch 1)
  - Logit clamp after opt_indices.step: ~line 1153 (Patch 3)

**Forbidden:** Do NOT touch lines 540-600 (optimizer-streams), 632-736 (nn-module-foundation), 1058-1103 (optimizer-streams), or any kernel/CUDA files.

---

## Branch Rules

1. Work ONLY on branch `agent/training-recipe`.
2. Commit after each sub-task. Push after each wave.
3. Before Wave 2, pull main to get nn-module-foundation's changes (Patch 9).
4. If you encounter merge conflict on train_qwen.py, message the conflicting agent via inbox.

---

## Inbox Protocol

Your inbox: `agent-ctx/agent-training-recipe/inbox/`

### Checking
- **Read your inbox at the START of every wave.**
- Look for "RELEASED" messages from nn-module-foundation before starting Wave 2.

### Sending
- To message another agent, write to THEIR inbox: `agent-ctx/agent-{recipient}/inbox/{unix-timestamp}-from-training-recipe.md`
- Use the message template (see `agent-ctx/agent-nn-module-foundation/RULES.md` §Inbox Protocol).

### Critical Messages You MUST Send

1. **Before Wave 1 (Patch 1, τ schedule):** No lock needed — you own lines 1034-1040 exclusively. Just start.

2. **Before Wave 2 (Patch 3, logit clamp):** No lock needed — you own line 1153 exclusively.

3. **After Wave 1 completes:** Message nn-module-foundation:
   - Subject: "Patch 1 (τ schedule) done on my branch"
   - Action: nothing (informational)

4. **If Patch 4 (group size, palettize_core.py) requires re-calibration:** Note in PROGRESS.md that existing checkpoint is incompatible. Orchestrator decides whether to re-calibrate or use per-tensor override.

---

## Research References

- **Patch 1 (τ schedule):** [`research-indices-training/04_tau_schedule.md`](../../research-indices-training/04_tau_schedule.md), [`research-filter-consolidation/01_training_recipe.md`](../../research-filter-consolidation/01_training_recipe.md) §Patch 1
- **Patch 3 (logit clamp):** [`research-filter-consolidation/01_training_recipe.md`](../../research-filter-consolidation/01_training_recipe.md) §Patch 3
- **Patch 4 (group size):** [`research-filter-consolidation/01_training_recipe.md`](../../research-filter-consolidation/01_training_recipe.md) §Patch 4
- **Papers:**
  - `docs/papers/1611.01144_Gumbel-Softmax_Jang2017.pdf` (τ schedule analysis)
  - `docs/papers/LLT_Wang_CVPR2022.pdf` (τ floor 0.5, gradient rescaling)
  - `docs/papers/1602.02830_BNN_Courbariaux2016.pdf` (tight logit clamp ±5τ)
  - `docs/papers/2210.17323_GPTQ_Frantar2023.pdf` (GS=128 standard)

---

## Definition of Done

- [ ] Patch 1: τ schedule is piecewise warmup (500 steps) + quadratic decay to 0.5 (6000 steps) + hold
- [ ] Patch 3: logit clamp is `±5*tau` (adaptive)
- [ ] Patch 4: GROUP_SIZE=128 in palettize_core.py (or per-tensor override documented)
- [ ] All syntax checks pass
- [ ] Branch pushed
- [ ] PROGRESS.md updated
- [ ] Inbox messages sent
