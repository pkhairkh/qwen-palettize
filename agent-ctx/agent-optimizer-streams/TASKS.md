# TASKS — agent-optimizer-streams

> **Branch:** `agent/optimizer-streams`
> **Patches:** 8 (fused AdamW), 6 (stream double-buffer)

---

## Wave 1: Patch 8 — Fused AdamW (bitsandbytes 8-bit)

**Research:** [`research-filter-consolidation/03_optimizer_speedup.md`](../../research-filter-consolidation/03_optimizer_speedup.md) §Patch 8, [`research-kernel-efficiency/08_recommendations.md`](../../research-kernel-efficiency/08_recommendations.md) Patch 6
**Papers:** `docs/papers/1412.6980_Adam_Kingma2015.pdf` (Adam), `docs/papers/1711.05101_AdamW_Loshchilov2019.pdf` (AdamW)
**File:** `scripts/train_qwen.py` (~lines 540-600, `build_optimizers()`)

### Sub-task 8a: Add bitsandbytes dependency

**File:** `requirements.txt` (new or update)

```
bitsandbytes>=0.43.0
```

**Commit:** `Patch 8a: add bitsandbytes to requirements`

### Sub-task 8b: Replace FP32MasterAdamW with bnb.optim.AdamW8bit

**File:** `scripts/train_qwen.py`, `build_optimizers()` (~line 594)

**Change:**
```python
# BEFORE:
    opt_indices = FP32MasterAdamW(plain_adamw_groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0) if plain_adamw_groups else None

# AFTER:
    import bitsandbytes as bnb
    # AdamW8bit: 8-bit state (m, v), handles fp16/bf16 params natively
    # State: 1.78B × 2 bytes (8-bit m+v) = 3.56 GB (was 21.4 GB with fp32 master)
    # Step time: ~20ms (was 113ms with 8 passes)
    opt_indices = bnb.optim.AdamW8bit(
        plain_adamw_groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0
    ) if plain_adamw_groups else None
```

**Rationale (from research-filter-consolidation/03_optimizer_speedup.md §8):** `FP32MasterAdamW` does 8 passes × ~200 kernel launches = 113ms. `bnb.optim.AdamW8bit` fuses all passes into one kernel launch, uses 8-bit state (3.56GB vs 21.4GB). Saves 93ms/step + 14GB VRAM.

**Note:** `bitsandbytes.optim.AdamW8bit` handles fp16 params natively — no fp32 master needed. The 8-bit state is less precise but stable for Gumbel-Softmax (test for NaN).

**Test:** `python3 -c "import ast; ast.parse(open('scripts/train_qwen.py').read()); print('OK')"`
**Commit:** `Patch 8b: replace FP32MasterAdamW with bnb.optim.AdamW8bit for opt_indices`
**Push:** `git push origin agent/optimizer-streams`

### Sub-task 8c: Update scheduler init (opt_indices.opt → opt_indices)

**File:** `scripts/train_qwen.py`, scheduler init (~line 966)

The current code has `sched_indices = torch.optim.lr_scheduler.LambdaLR(opt_indices.opt, lr_lambda)` because `FP32MasterAdamW` wraps `.opt`. With `bnb.optim.AdamW8bit`, there's no wrapper — use `opt_indices` directly:

```python
# BEFORE:
    sched_indices = torch.optim.lr_scheduler.LambdaLR(opt_indices.opt, lr_lambda) if opt_indices else None

# AFTER:
    sched_indices = torch.optim.lr_scheduler.LambdaLR(opt_indices, lr_lambda) if opt_indices else None
```

Also update `update_lrs()` function (~line 586) — remove the `opt_indices.param_groups` wrapper indirection if any.

**Commit:** `Patch 8c: fix scheduler + update_lrs for bnb.optim.AdamW8bit (no .opt wrapper)`

**DoD for Wave 1:**
- [ ] bitsandbytes in requirements
- [ ] `bnb.optim.AdamW8bit` replaces `FP32MasterAdamW` for opt_indices
- [ ] Scheduler init uses `opt_indices` (not `opt_indices.opt`)
- [ ] `update_lrs()` updated for direct optimizer (no wrapper)
- [ ] syntax check passes
- [ ] Branch pushed
- [ ] PROGRESS.md: Patch 8 ✅

---

## Wave 2: Patch 6 — Stream Double-Buffering

**Research:** [`research-kernel-efficiency/06_stream_overlap.md`](../../research-kernel-efficiency/06_stream_overlap.md), [`research-filter-consolidation/02_kernel_efficiency.md`](../../research-filter-consolidation/02_kernel_efficiency.md) §Patch 6
**File:** `scripts/train_qwen.py` (~lines 1058-1103, training loop)

**Prerequisite:** Check inbox for "RELEASED" message from nn-module-foundation. If present, `git pull origin main` to get nn.Module changes, then rebase your branch.

### Sub-task 6a: Send lock message

Before editing lines 1058-1103, message training-recipe:
- Subject: "LOCK: train_qwen.py:1058-1103 for Wave 2"
- Body: "I'm adding stream double-buffering. Do not touch lines 1058-1103 until RELEASED."
- Action: wait

### Sub-task 6b: Add persistent stream + double-buffered h_out

**File:** `scripts/train_qwen.py`, training loop (~line 1058)

**Change:**

```python
# BEFORE (current — fresh stream per step):
    for batch_ids in data_stream:
        stream_t = torch.cuda.Stream()  # ← created fresh each step!
        with torch.cuda.stream(stream_t):
            with torch.no_grad():
                # ... teacher forward ...
                h_out = h.detach()
        # Student uses h_out (must wait for stream_t)

# AFTER (proposed — persistent stream + double-buffer):
    # Pre-loop: allocate persistent stream + double-buffered h_out + events
    stream_t = torch.cuda.Stream()  # ← allocated ONCE
    h_out_buf = [None, None]
    event_t = [torch.cuda.Event(), torch.cuda.Event()]
    event_s = [torch.cuda.Event(), torch.cuda.Event()]
    buf_idx = 0

    # Prologue: start teacher forward for step 0
    with torch.cuda.stream(stream_t):
        with torch.no_grad():
            # ... teacher forward for first batch ...
            h_out_buf[0] = h.detach()
    event_t[0].record(stream_t)

    for step, batch_ids in enumerate(data_stream):
        curr = buf_idx
        nxt = 1 - buf_idx

        # Wait for teacher's output for THIS step
        torch.cuda.current_stream().wait_event(event_t[curr])

        # === STUDENT FORWARD + BACKWARD (uses h_out_buf[curr]) ===
        s_h = student.model.embed_tokens(batch_ids)
        # ... student forward ...
        student_out = s_h
        loss, comps = compute_loss(student_out, h_out_buf[curr], hp)
        loss.backward()
        # ... clip + opt step ...

        # Signal: student done with h_out_buf[curr]
        event_s[curr].record()

        # === TEACHER FORWARD for NEXT step (on stream_t, overlaps) ===
        if step + 1 < max_steps:
            try:
                next_batch = next(data_stream)
            except StopIteration:
                break
            stream_t.wait_event(event_s[nxt])
            with torch.cuda.stream(stream_t):
                with torch.no_grad():
                    # ... teacher forward for next_batch ...
                    h_out_buf[nxt] = h.detach()
            event_t[nxt].record(stream_t)

        buf_idx = nxt
```

**Rationale (from research-kernel-efficiency/06_stream_overlap.md):** Current setup creates a new `torch.cuda.Stream()` every step (5µs overhead × 25 Linears = 125µs wasted). Teacher forward (68ms) is sequential with student backward (260ms). Double-buffering overlaps them: total = max(68, 260) = 260ms, saving 68ms/step.

**Commit:** `Patch 6b: stream double-buffering (persistent stream + h_out_buf[2] + events)`

### Sub-task 6c: Send release message

Message training-recipe:
- Subject: "RELEASED: stream double-buffer done"
- Body: "Lines 1058-1103 are done. You can proceed if you need them."
- Action: nothing

**Push:** `git push origin agent/optimizer-streams`

**DoD for Wave 2:**
- [ ] Persistent `stream_t` (allocated once, not per-step)
- [ ] `h_out_buf[2]` double-buffered
- [ ] `event_t[2]` + `event_s[2]` for synchronization
- [ ] Teacher forward for step N+1 overlaps with student backward for step N
- [ ] syntax check passes
- [ ] Branch pushed (rebased on main if nn.Module merged)
- [ ] PROGRESS.md: Patch 6 ✅
- [ ] Inbox messages sent (lock + release)

---

## Wave 3: Final Verification + Merge Prep

### Sub-task 9a: Verify no NaN with 8-bit AdamW

Check that `bnb.optim.AdamW8bit` doesn't produce NaN with fp16 index_logits. If it does, raise eps to 1e-6:
```python
opt_indices = bnb.optim.AdamW8bit(..., eps=1e-6)
```

### Sub-task 9b: Merge prep

```bash
git fetch origin main
git merge origin/main  # resolve conflicts in train_qwen.py if any (lines 540-600, 1058-1103)
```

### Sub-task 9c: Final inbox message

Message orchestrator (via PROGRESS.md update): "optimizer-streams branch ready for merge"

**DoD for Wave 3:**
- [ ] No NaN with 8-bit AdamW (or eps raised to 1e-6)
- [ ] Branch merges cleanly with main
- [ ] PROGRESS.md fully updated
- [ ] Branch pushed
