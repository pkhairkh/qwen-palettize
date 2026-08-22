# TASKS — agent-training-recipe

> **Branch:** `agent/training-recipe`
> **Patches:** 1 (τ schedule), 3 (logit clamp), 4 (group size)

---

## Wave 1: Patch 1 — Polynomial τ Schedule with Floor at 0.5

**Research:** [`research-indices-training/04_tau_schedule.md`](../../research-indices-training/04_tau_schedule.md), [`research-filter-consolidation/01_training_recipe.md`](../../research-filter-consolidation/01_training_recipe.md) §Patch 1
**Papers:** `docs/papers/1611.01144_Gumbel-Softmax_Jang2017.pdf` (τ analysis), `docs/papers/LLT_Wang_CVPR2022.pdf` (floor 0.5)
**Files:** `scripts/train_qwen.py` (~lines 1034-1040 τ anneal, ~1241-1246 CLI defaults)

### Sub-task 1a: Update CLI defaults

**File:** `scripts/train_qwen.py`, argparse section (~line 1241)

**Change:**
```python
# BEFORE:
    ap.add_argument("--tau_init", type=float, default=0.1)
    ap.add_argument("--tau_final", type=float, default=0.01)
    ap.add_argument("--tau_anneal_steps", type=int, default=4000)

# AFTER:
    ap.add_argument("--tau_init", type=float, default=2.0,
                    help="Initial Gumbel-Softmax temperature. Default 2.0.")
    ap.add_argument("--tau_final", type=float, default=0.5,
                    help="Final Gumbel-Softmax temperature. Default 0.5 (FLOOR).")
    ap.add_argument("--tau_anneal_steps", type=int, default=6000,
                    help="Steps over which to anneal temperature.")
```

**Test:** `python3 -c "import ast; ast.parse(open('scripts/train_qwen.py').read()); print('OK')"`
**Commit:** `Patch 1a: update tau CLI defaults (2.0, 0.5, 6000)`

### Sub-task 1b: Replace linear τ anneal with piecewise polynomial

**File:** `scripts/train_qwen.py`, τ anneal logic (~line 1034)

**Change:**
```python
# BEFORE:
        if use_soft_indices:
            tau = max(tau_final, tau_init * (1.0 - global_step / tau_anneal_steps))
            for name, mod in student.named_modules():
                if hasattr(mod, 'tau'):
                    mod.tau = tau

# AFTER:
        if use_soft_indices:
            T_WARMUP = 500
            T_ANNEAL = tau_anneal_steps  # 6000
            if global_step < T_WARMUP:
                tau = tau_init  # 2.0 — warmup at high tau
            elif global_step < T_WARMUP + T_ANNEAL:
                progress = (global_step - T_WARMUP) / T_ANNEAL
                tau = max(tau_final, tau_init * (1.0 - progress) ** 2)  # alpha=2 quadratic
            else:
                tau = tau_final  # 0.5 — hold
            for name, mod in student.named_modules():
                if hasattr(mod, 'tau'):
                    mod.tau = tau
```

**Rationale (from research-indices-training/04_tau_schedule.md):** Linear schedule spends 25% of window at τ<0.5 where gradients are <9% of peak. Quadratic decay (α=2) front-loads high-τ regime, keeping gradients strong for 66% of training. **62% boost in cumulative gradient signal.**

**Test:** syntax check
**Commit:** `Patch 1b: piecewise warmup + quadratic τ decay to floor 0.5`
**Push:** `git push origin agent/training-recipe`

### Sub-task 1c: Notify nn-module-foundation

Message nn-module-foundation inbox: "Patch 1 done on agent/training-recipe branch. No file conflicts with your work."

**DoD for Wave 1:**
- [ ] τ defaults: 2.0, 0.5, 6000
- [ ] Piecewise: 500-step warmup at τ=2.0, quadratic decay to 0.5, hold
- [ ] syntax check passes
- [ ] Branch pushed
- [ ] Inbox message sent to nn-module-foundation
- [ ] PROGRESS.md: Patch 1 ✅

---

## Wave 2: Patch 3 — Adaptive Logit Clamp ±5τ

**Research:** [`research-filter-consolidation/01_training_recipe.md`](../../research-filter-consolidation/01_training_recipe.md) §Patch 3
**Paper:** `docs/papers/1602.02830_BNN_Courbariaux2016.pdf` (BNN tight clamp pattern)
**File:** `scripts/train_qwen.py` (~line 1153, after opt_indices.step())

**Prerequisite:** Check inbox for "RELEASED" message from nn-module-foundation. If present, `git pull origin main` to get nn.Module changes, then rebase your branch.

### Sub-task 2a: Change clamp from ±20 to ±5τ

**File:** `scripts/train_qwen.py`, after `opt_indices.step()` (~line 1153)

**Change:**
```python
# BEFORE:
        if opt_indices:
            with torch.no_grad():
                for name, par in student.named_parameters():
                    if "index_logits" in name:
                        par.data.clamp_(-20.0, 20.0)

# AFTER:
        if opt_indices:
            with torch.no_grad():
                for name, par in student.named_parameters():
                    if "index_logits" in name:
                        # Adaptive clamp: ±5τ (was ±20)
                        # At tau=2.0: clamp ±10 (loose, exploration)
                        # At tau=0.5: clamp ±2.5 (tight, commitment)
                        par.data.clamp_(-5.0 * tau, 5.0 * tau)
```

**Rationale (from research-filter-consolidation/01_training_recipe.md §3):** At τ=0.1, `softmax(±20/0.1) = softmax(±200)` overflows to [1,0] — gradient is exactly zero. `±5τ` gives `softmax(±5) ≈ [0.993, 0.007]` — still hard but with finite-precision gradient.

**Test:** syntax check. Verify `tau` variable is in scope at line 1153 (it should be — set at top of training loop).
**Commit:** `Patch 3: adaptive logit clamp ±5τ (was ±20)`
**Push:** `git push origin agent/training-recipe`

**DoD for Wave 2:**
- [ ] Clamp is `±5*tau` (adaptive)
- [ ] syntax check passes
- [ ] Branch pushed (rebased on main if nn.Module merged)
- [ ] PROGRESS.md: Patch 3 ✅

---

## Wave 3: Patch 4 — Group Size 256→128

**Research:** [`research-filter-consolidation/01_training_recipe.md`](../../research-filter-consolidation/01_training_recipe.md) §Patch 4
**Paper:** `docs/papers/2210.17323_GPTQ_Frantar2023.pdf` (GS=128 is GPTQ/AWQ standard)
**File:** `scripts/palettize_core.py` (line 26)

### Sub-task 4a: Change GROUP_SIZE constant

**File:** `scripts/palettize_core.py`, line 26

**Change:**
```python
# BEFORE:
GROUP_SIZE = 256

# AFTER:
GROUP_SIZE = 128  # Halved from 256 for better k-means fit (GPTQ/AWQ standard)
```

**Rationale (from research-filter-consolidation/01_training_recipe.md §4):** GS=256 is the largest in the literature. Halving halves within-group weight diversity, so the 4-entry k-means codebook fits better. Expected: mean cos 0.937 → 0.945-0.950 at calibration.

**Note:** This makes existing checkpoints (GS=256) incompatible. Document in PROGRESS.md that re-calibration is needed before training restart. Orchestrator decides whether to re-calibrate or use per-tensor override for the 5 worst Linears only.

### Sub-task 4b: (Optional) Add per-tensor override

If you want to preserve existing checkpoint compatibility, add:
```python
GROUP_SIZE_OVERRIDES = {
    "model.layers.2.linear_attn.out_proj.weight": 128,
    "model.layers.2.linear_attn.in_proj_qkv.weight": 128,
    "model.layers.1.mlp.down_proj.weight": 128,
    "model.layers.0.linear_attn.in_proj_z.weight": 128,
    "model.layers.3.self_attn.k_proj.weight": 128,
}
def get_group_size_for_tensor(tensor_name):
    return GROUP_SIZE_OVERRIDES.get(tensor_name, GROUP_SIZE)
```

**Test:** syntax check
**Commit:** `Patch 4: GROUP_SIZE 256→128 (+ optional per-tensor override)`
**Push:** `git push origin agent/training-recipe`

### Sub-task 4c: Final merge prep

```bash
git fetch origin main
git merge origin/main  # resolve conflicts in train_qwen.py if any
```

**DoD for Wave 3:**
- [ ] GROUP_SIZE=128 (or per-tensor override documented)
- [ ] syntax check passes
- [ ] Branch merges cleanly with main
- [ ] PROGRESS.md: Patch 4 ✅, note about re-calibration
- [ ] Final inbox message to orchestrator: "training-recipe branch ready for merge"
