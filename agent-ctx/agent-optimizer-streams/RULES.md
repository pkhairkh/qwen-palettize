# RULES — agent-optimizer-streams

> **Branch:** `agent/optimizer-streams`
> **Patches:** 8 (fused AdamW), 6 (stream double-buffer)
> **Depends on:** nn-module-foundation (Patch 9) for clean merge — but can start Wave 1 in parallel.

---

## File Ownership (EXCLUSIVE)

- `scripts/train_qwen.py` — ONLY these line ranges:
  - `build_optimizers()` function: ~lines 540-600 (Patch 8 — fused AdamW)
  - Training loop stream setup: ~lines 1058-1103 (Patch 6 — stream double-buffer)

**Forbidden:** Do NOT touch lines 632-736 (nn-module-foundation), 1034-1040 (training-recipe τ), 1153 (training-recipe clamp), or any kernel/CUDA files.

---

## Branch Rules

1. Work ONLY on branch `agent/optimizer-streams`.
2. Commit after each sub-task. Push after each wave.
3. Before Wave 2, pull main to get nn-module-foundation's changes (Patch 9 nn.Module).
4. If you encounter merge conflict on train_qwen.py, message the conflicting agent via inbox.

---

## Inbox Protocol

Your inbox: `agent-ctx/agent-optimizer-streams/inbox/`

### Checking
- **Read your inbox at the START of every wave.**
- Look for "RELEASED" messages from nn-module-foundation before starting Wave 2.

### Sending
- To message another agent, write to THEIR inbox.
- Critical messages you MUST send:
  1. **Before Wave 1 (Patch 8, fused AdamW):** No lock needed — you own lines 540-600 exclusively.
  2. **Before Wave 2 (Patch 6, stream double-buffer):** Lock lines 1058-1103. Message training-recipe:
     - Subject: "LOCK: train_qwen.py:1058-1103 for Wave 2"
     - Body: "I'm adding stream double-buffering to the training loop. Do not touch lines 1058-1103 until I send RELEASED."
  3. **After Wave 2:** Message training-recipe: "RELEASED: stream double-buffer done"

---

## Research References

- **Patch 8 (fused AdamW):** [`research-filter-consolidation/03_optimizer_speedup.md`](../../research-filter-consolidation/03_optimizer_speedup.md) §Patch 8, [`research-kernel-efficiency/08_recommendations.md`](../../research-kernel-efficiency/08_recommendations.md) Patch 6
- **Patch 6 (stream double-buffer):** [`research-kernel-efficiency/06_stream_overlap.md`](../../research-kernel-efficiency/06_stream_overlap.md), [`research-filter-consolidation/02_kernel_efficiency.md`](../../research-filter-consolidation/02_kernel_efficiency.md) §Patch 6
- **Papers:**
  - `docs/papers/1412.6980_Adam_Kingma2015.pdf` (Adam algorithm)
  - `docs/papers/1711.05101_AdamW_Loshchilov2019.pdf` (AdamW decoupled weight decay)
  - `docs/papers/1705.07774_DissectingAdam_Balles2017.pdf` (Adam analysis)

---

## Definition of Done

- [ ] Patch 8: `bitsandbytes.optim.AdamW8bit` replaces `FP32MasterAdamW` for opt_indices
- [ ] Patch 6: Persistent stream + double-buffered h_out_buf[2] + CUDA events
- [ ] All syntax checks pass
- [ ] Branch pushed
- [ ] PROGRESS.md updated
- [ ] Inbox messages sent
