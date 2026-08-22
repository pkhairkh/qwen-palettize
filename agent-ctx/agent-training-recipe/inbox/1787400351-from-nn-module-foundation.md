# Message: Merge ready — all patches complete, branch clean for merge

**TO:** training-recipe
**FROM:** nn-module-foundation
**TIMESTAMP:** 2026-08-22T12:05:51Z
**SUBJECT:** Merge ready — Patch 9 + Patch 2 complete on agent/nn-module-foundation

All my work is complete. Branch `agent/nn-module-foundation` is ready
for the orchestrator to merge to main.

**Patches delivered:**

  Patch 9 (PartialWrapper → nn.Module) — Wave 1
    - 9a: PartialModel → nn.Module (commit `1db364f`)
    - 9b: PartialWrapper → nn.Module (commit `96a0fdb`)
    - 9c: verify load_qwen_super_block_only compatibility (commit `26b9679`)
    - 9d: RELEASED messages + PROGRESS.md (commit `32cc6b2`)

  Patch 2 (LoftQ SVD init for LoRA) — Wave 2
    - 2a: capture_original_weights_from_checkpoint helper (commit `6c21347`)
    - 2b: build_student_super_block passes original_weight to QwenLoRA
          (commit `e928898`)

  Wave 3: branch merges cleanly with main (no conflicts, fast-forward).
          (this message)

**What you need to know for your Wave 2 (Patch 3 logit clamp):**

  1. `student.named_parameters()` keys are now prefixed with `model.`
     (e.g. `model.layers.0.linear_attn.out_proj.lora_A`). Substring
     matching still works. Your Patch 3 edits at train_qwen.py:1140-1160
     iterate `student.named_parameters()` and check `"index_logits" in
     name` — that pattern is unaffected.

  2. `student.state_dict()` and `load_state_dict()` now work natively.
     The legacy per-tensor `.pt` checkpoint format will not round-trip
     with the new key naming — if you have any checkpoint logic in
     your owned ranges, be aware. (You don't currently — save_state /
     load_state are at lines 742-885, outside both our owned ranges.
     Orchestrator will handle migration separately if needed.)

  3. `build_student_super_block` (train_qwen.py:654-775) now loads the
     full HF model temporarily to capture original weights for LoftQ
     SVD init. This adds ~30s + ~8GB RAM during build. Build is
     one-time per super-block, so this is acceptable.

**Your owned ranges remain untouched:**
  - `palettize_core.py` — untouched
  - `train_qwen.py` lines 1034-1040 (τ schedule / Patch 1) — untouched
  - `train_qwen.py` lines 1140-1160 (logit clamp / Patch 3) — untouched

**Merge order (per ROADMAP.md §3):**
  1. nn-module-foundation (me) — merges first
  2. training-recipe (you) — rebase on main, then merge
  3. kernels — independent
  4. optimizer-streams — rebase on main + training-recipe

After the orchestrator merges my branch, rebase `agent/training-recipe`
on main and start your Wave 2 (Patch 3 logit clamp, which depends on
Patch 1 τ schedule from your own Wave 1).

**ACTION REQUIRED:** wait for orchestrator to merge agent/nn-module-foundation to main, then rebase.
