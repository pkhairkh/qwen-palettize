# 05 — Special Notes

> **Wave 3 deliverable.** Critical context for the implementation agent. Read this BEFORE applying any patch.

---

## 1. 1d-kmeans is the BEST Non-GPTQ Calibration (Already Tested)

**Status:** VERIFIED. Do NOT recommend GPTQ, Hessian-weighted Lloyd-Max, or any alternative calibration method.

Our calibration uses **1d-kmeans weighted by activation norm** (in `palettize_core.py`). This was tested against:
- ✅ 1d-kmeans (BEST for non-GPTQ) — **our current approach**
- ❌ Hessian-weighted Lloyd-Max — tested, **BULLSHIT** (worse than plain k-means)
- ❌ GPTQ Hessian-based calibration — sidestep (different algorithm entirely)
- ❌ AWQ activation-aware scaling — sidestep (different preprocessing)
- ❌ QuIP RHT incoherence — sidestep (different preprocessing)

**Any agent recommending GPTQ, Lloyd-Max, AWQ, or QuIP calibration is proposing a SIDESTEP.** These were all considered and rejected. We enhance k-means (e.g., group size tuning, per-tensor overrides), not replace it.

---

## 2. Hessian-Weighted Lloyd-Max is BULLSHIT (Already Tested)

**Status:** VERIFIED. Do NOT recommend.

Agent 6 (literature-review) recommended "Lloyd-Max Gaussian codebook init" (`research-literature-review/00_executive_summary.md`). This was tested previously and found to be **inferior to plain 1d-kmeans**.

The Hessian-weighted variant (SqueezeLLM pattern) was also tested and rejected. It adds complexity without improving cos.

**Any agent recommending Lloyd-Max or Hessian-weighted k-means is proposing a SIDESTEP.**

---

## 3. PartialWrapper → nn.Module is a SPEED Enhancement (KEEP)

**Status:** KEEP. This is Patch 9 in `03_optimizer_speedup.md`.

Initially I (the orchestrator) classified PartialWrapper→nn.Module as "NEVER" (architecture change). This was WRONG. It is a **speed enhancement**, not a sidestep:

- Does NOT change the training approach (Gumbel-Softmax + STE + k-means + LoRA)
- Only changes the container class (plain Python → nn.Module)
- Unlocks: torch.compile (1.5-2× speedup), gradient checkpointing (batch=128+), state_dict (faster save/load)
- Agent 5 (architecture-review) correctly identified this as a speed enhancement

**Any future agent should classify PartialWrapper→nn.Module as KEEP (speed enhancement).**

---

## 4. Training Server is OFFLINE — All Patches are Research-Only

**Status:** The Blackwell training server (RTX PRO 6000, 35.246.48.124) is currently OFFLINE.

All patches in this consolidation folder are **RESEARCH ONLY**:
- Code patches are REFERENCE (clearly marked "NOT YET APPLIED")
- No training runs, no GPU access, no server connection
- Implementation agent will apply patches when server is back online

**Do NOT attempt to:**
- SSH to the training server
- Run `train_qwen.py`
- Run `calib_qwen.py`
- Access the GPU

---

## 5. Ambiguous Recommendations (Classified as KEEP)

The following recommendations were ambiguous (could be enhancement OR sidestep). They are classified as **KEEP** and noted here for transparency:

### 5a. F.gumbel_softmax(hard=True) vs Manual STE

**Recommendation:** Replace manual STE (`W = W_hard - W_soft.detach() + W_soft`) with `F.gumbel_softmax(hard=True)`.

**Verdict:** KEEP (cleaner implementation, same approach)

**Ambiguity:** Could be seen as "replacing" our STE. But it's the SAME mathematical operation, just using PyTorch's built-in implementation. The Gumbel-Softmax approach is preserved.

**Source:** Agent 1 (kernel-accuracy) `03_ste_analysis.md`, Agent 4 (palettes-training) `08_recommendations.md`

### 5b. Exponential τ Schedule vs Polynomial

**Recommendation:** Use exponential τ decay instead of polynomial.

**Verdict:** KEEP (alternative τ schedule, preserves Gumbel)

**Ambiguity:** Could be seen as changing the τ schedule. But τ schedule tuning IS enhancement, not sidestep. We keep the polynomial as default (Patch 1) but exponential is a valid alternative.

**Source:** Agent 3 (indices-training) `01_gumbel_softmax_audit.md`, Agent 6 (literature-review) `03_codebook_methods.md`

### 5c. Hybrid FP32 v + bf16 m AdamW State

**Recommendation:** Keep v (second moment) in fp32, use bf16 for m (first moment) and master.

**Verdict:** KEEP (memory optimization, preserves AdamW)

**Ambiguity:** Could be seen as changing the optimizer. But it's a mixed-precision optimization of the SAME AdamW algorithm. Not a sidestep.

**Source:** Agent 2 (kernel-efficiency) `05_memory_optimization.md`

### 5d. Cyclic Loss Schedule

**Recommendation:** Alternate 100 steps of norm_mse with 100 steps of 1-cos.

**Verdict:** KEEP (loss schedule, preserves approach)

**Ambiguity:** Could be seen as changing the loss. But it cycles BETWEEN our two existing loss types. Not a sidestep.

**Source:** Agent 4 (palettes-training) `05_loss_function.md`

---

## 6. Dependencies Between Patches

```
Patch 1 (τ schedule) ──────────────────────→ standalone
Patch 2 (LoftQ SVD init) ──────────────────→ standalone (needs capture helper)
Patch 3 (adaptive logit clamp) ────────────→ depends on Patch 1 (needs τ in scope)
Patch 4 (group size 256→128) ──────────────→ standalone (needs re-calibration)
Patch 5 (fused bwd AoS P) ─────────────────→ standalone (kernel change)
Patch 6 (stream double-buffer) ────────────→ depends on data prefetch (Patch 7 or cache)
Patch 7 (batched compute_P_W) ─────────────→ depends on Patch 5 (same P layout)
Patch 8 (fused AdamW) ──────────────────────→ standalone (needs bitsandbytes)
Patch 9 (PartialWrapper→nn.Module) ────────→ standalone (unlocks future: compile, checkpointing)
```

**Recommended order:** 9 → 1 → 2 → 3 → 4 → 5 → 7 → 6 → 8

---

## 7. What This Consolidation Does NOT Include

This consolidation focuses on **9 core patches** that enhance our current approach. It does NOT include:

1. **Testing infrastructure** (pytest, CI/CD) — out of scope for enhancement
2. **Logging frameworks** (wandb, tensorboard) — out of scope for enhancement
3. **Modular package rewrite** (Trainer class, Config dataclass) — architecture change
4. **Radical kernel migration** (tcgen05, WGMMA, TMA) — keep mma.sync.m16n8k16
5. **Alternative quantization methods** (GPTQ, AWQ, QuIP, SqueezeLLM, AQLM) — sidesteps
6. **Vector quantization** (VQ, additive codebook, lattice) — representation change
7. **From-scratch training** (BitNet pattern) — replaces k-means init

These are documented in `04_rejected_sidesteps.md` for reference.
