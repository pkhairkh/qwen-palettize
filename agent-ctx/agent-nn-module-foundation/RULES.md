# RULES — agent-nn-module-foundation

> **Branch:** `agent/nn-module-foundation`
> **Foundation agent:** YES — your merge unblocks training-recipe and optimizer-streams.

---

## File Ownership (EXCLUSIVE)

You own these files. No other agent will touch them:

- `scripts/qwen_model.py` — `PartialModel` class (~line 438), `PartialWrapper` class (~line 540), new `capture_original_weights_from_checkpoint()` helper
- `scripts/train_qwen.py` — `build_student_super_block()` function ONLY (~lines 632-736). Other agents own other sections — see `agent-ctx/ROADMAP.md` §2.

**Forbidden:** Do NOT modify `fused_lut_kernel.cu`, `fused_lut_linear_cuda.py`, `palettize_core.py`, or other sections of `train_qwen.py`.

---

## Branch Rules

1. Work ONLY on branch `agent/nn-module-foundation`.
2. Commit after each sub-task. Push after each wave.
3. Use `git pull origin main` only at the START of each wave to incorporate any merged changes.
4. If you encounter a merge conflict with main, message the conflicting agent via inbox.

---

## Inbox Protocol

Your inbox: `agent-ctx/agent-nn-module-foundation/inbox/`

### Checking
- **Read your inbox at the START of every wave** (before any code changes).
- Process messages in timestamp order.
- Delete processed messages after acting on them (or move to `processed/` subfolder if you want history).

### Sending
- To message another agent, write to THEIR inbox: `agent-ctx/agent-{recipient}/inbox/{unix-timestamp}-from-nn-module-foundation.md`
- Use this template:
  ```markdown
  # Message: {subject}

  **TO:** {recipient}
  **FROM:** nn-module-foundation
  **TIMESTAMP:** {ISO 8601}
  **SUBJECT:** {short subject}

  {body}

  **ACTION REQUIRED:** {rebase / wait / coordinate / nothing}
  ```

### Critical Messages You MUST Send

1. **Before Wave 1 commit:** Message training-recipe and optimizer-streams:
   - Subject: "LOCK: train_qwen.py:632-736 for Wave 1"
   - Body: "I'm editing build_student_super_block. Do not touch lines 632-736 until I send RELEASED."
   - Action: wait

2. **After Wave 1 (Patch 9 merged to your branch, pushed):** Message ALL agents:
   - Subject: "RELEASED: nn.Module merged — rebase your branches"
   - Body: "Patch 9 (PartialWrapper→nn.Module) is on branch agent/nn-module-foundation. Orchestrator will merge to main. After merge, rebase your branch on main."
   - Action: rebase

3. **Before Wave 2 (Patch 2 LoftQ):** Same lock pattern for lines 632-736.

---

## Research References

- **Patch 9 (nn.Module):** [`research-architecture-review/02_partial_wrapper_problem.md`](../../research-architecture-review/02_partial_wrapper_problem.md)
- **Patch 2 (LoftQ):** [`research-kernel-accuracy/00_overview.md`](../../research-kernel-accuracy/00_overview.md) §Fix 1, [`research-filter-consolidation/01_training_recipe.md`](../../research-filter-consolidation/01_training_recipe.md) §Patch 2
- **Paper:** `docs/papers/2305.14314_QLoRA_Dettmers2023.pdf` (QLoRA uses nn.Module + LoRA pattern)

---

## Definition of Done

- [ ] Patch 9: PartialModel + PartialWrapper inherit from nn.Module, forward() added, hand-rolled methods deleted
- [ ] Patch 2: capture_original_weights_from_checkpoint() helper added, QwenLoRA receives original_weight
- [ ] `python3 -c "import ast; ast.parse(open('scripts/qwen_model.py').read())"` passes
- [ ] `python3 -c "import ast; ast.parse(open('scripts/train_qwen.py').read())"` passes
- [ ] `isinstance(student, nn.Module)` would return True (verify class definition)
- [ ] All commits pushed to `agent/nn-module-foundation` branch
- [ ] PROGRESS.md updated
- [ ] Inbox messages sent to dependent agents
