# RULES — agent-kernels

> **Branch:** `agent/kernels`
> **Patches:** 5 (fused bwd AoS P), 7 (batched compute_P_W)
> **Independent agent:** No file conflicts with other agents. You own the CUDA + Python kernel files exclusively.

---

## File Ownership (EXCLUSIVE)

- `scripts/fused_lut_kernel.cu` — FULL ownership
- `scripts/fused_lut_linear_cuda.py` — FULL ownership (the autograd Function classes, NOT train_qwen.py)

**Forbidden:** Do NOT touch `train_qwen.py`, `qwen_model.py`, `palettize_core.py`.

---

## Branch Rules

1. Work ONLY on branch `agent/kernels`.
2. Commit after each sub-task. Push after each wave.
3. Pull main at the start of each wave (in case nn-module-foundation's changes affect your forward/backward wrappers — they shouldn't, but verify).
4. If nn-module-foundation changes `qwen_model.py` in a way that affects how `PalettizedLinear.forward` calls your kernels, message them via inbox.

---

## Inbox Protocol

Your inbox: `agent-ctx/agent-kernels/inbox/`

### Checking
- **Read your inbox at the START of every wave.**
- Look for messages from nn-module-foundation about changes to `PalettizedLinear.forward()` that might affect kernel call signatures.

### Sending
- To message another agent, write to THEIR inbox.
- Critical messages you MUST send:
  1. **After Wave 1 (Patch 5, AoS P layout):** Message training-recipe and optimizer-streams:
     - Subject: "P layout changed to (K,N,4) AoS"
     - Body: "fused_lut_kernel.cu now outputs P as (K,N,4) instead of (4,K,N). If you reference P in train_qwen.py, update the permute. PalettizedLinear.forward is unchanged (kernel handles the layout internally)."
     - Action: coordinate (check if they reference P)
  2. **After Wave 2 (Patch 7, batched compute_P_W):** Message nn-module-foundation:
     - Subject: "Batched compute_P_W added"
     - Body: "New fused_compute_P_W_batched() function in fused_lut_linear_cuda.py. PalettizedLinear.forward can optionally use it (25 launches → 1)."

---

## Research References

- **Patch 5 (fused bwd AoS):** [`research-kernel-efficiency/02_fused_bwd_fix.md`](../../research-kernel-efficiency/02_fused_bwd_fix.md), [`research-filter-consolidation/02_kernel_efficiency.md`](../../research-filter-consolidation/02_kernel_efficiency.md) §Patch 5
- **Patch 7 (batched compute_P_W):** [`research-kernel-efficiency/03_batched_compute_pw.md`](../../research-kernel-efficiency/03_batched_compute_pw.md), [`research-filter-consolidation/02_kernel_efficiency.md`](../../research-filter-consolidation/02_kernel_efficiency.md) §Patch 7
- **Papers:** No specific paper for these kernel optimizations. Reference: `docs/papers/2210.17323_GPTQ_Frantar2023.pdf` (GPTQ-Marlin kernel pattern mentioned in `research-kernel-efficiency/07_literature_comparison.md` — but that file was deleted; pattern is in the surviving efficiency files).

---

## Definition of Done

- [ ] Patch 5: P stored as (K,N,4) AoS, fused bwd kernel re-enabled, backward 260ms→~60ms (claimed)
- [ ] Patch 7: Single batched compute_P_W kernel replaces 25 launches, saves ~36ms/step (claimed)
- [ ] Correctness: `max_err < 1e-3` vs Python reference (write a small test script)
- [ ] syntax check passes on both .cu and .py files
- [ ] Branch pushed
- [ ] PROGRESS.md updated
- [ ] Inbox messages sent to dependent agents
