# TASKS: cuda-graphs

## Branch
`agent/cuda-graphs`

## Overview
You capture the entire training step as a CUDA Graph and integrate it with the stream double-buffer. You merge LAST (Wave 4).

## Patch Inventory
| # | Patch | Wave | Effort | Status |
|---|-------|------|--------|--------|
| 21 | CUDA Graph capture for full step | 4 | 1 day | ⬜ |
| 22 | Stream double-buffer + CUDA Graphs integration | 4 | 1 day | ⬜ |

---

## WAVE 4

### Sub-task 21a: CUDA Graph capture for full training step
**Research:** `research-kernel-efficiency/08_recommendations.md` §11 (CUDA Graphs patch)
**Paper:** (CUDA Graphs, no specific paper — NVIDIA documentation)

**File:** `scripts/train_qwen.py:1095-1250` (training loop)

**What:** Capture the entire training step (student forward + loss + backward + optimizer) as a CUDA Graph. Replay per step. Eliminates all Python dispatch overhead (~500 kernel launches × 5µs = 2.5ms) and enables the GPU to schedule kernels back-to-back without CPU involvement.

**Requirements:**
1. Static input tensors — `batch_ids` copied into a pre-allocated static buffer before graph replay.
2. No dynamic shapes — all tensor shapes must be fixed at capture time.
3. No data-dependent control flow — the `if not torch.isfinite(loss)` NaN check must be outside the graph.
4. Warmup runs (3-5 steps) before capture to ensure all autotuner configs are selected.

**Implementation:**
```python
# In train_qwen.py training loop:

# Static input buffer
static_batch_ids = torch.empty(batch_size, seq_len, dtype=torch.long, device=DEVICE)

# Warmup (3 steps — ensures autotuner configs are selected)
for _ in range(3):
    student.train()
    s_h = student.model.embed_tokens(static_batch_ids)
    # ... full forward + loss + backward + optimizer ...

# Capture
step_graph = torch.cuda.CUDAGraph()
torch.cuda.synchronize()
with torch.cuda.graph(step_graph, stream=torch.cuda.current_stream()):
    # Full step: student forward + loss + backward + optimizer
    s_h = student.model.embed_tokens(static_batch_ids)
    with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
        # ... student forward ...
        student_out = s_h
    loss, comps = compute_loss(student_out, h_out_buf[buf_idx], hp)
    loss.backward()
    # ... clip + optimizer step + zero_grad ...

# Replay (per step)
def training_step(batch_ids):
    static_batch_ids.copy_(batch_ids)
    step_graph.replay()
    return loss, comps
```

**Dependency:** Wait for ALL other agents to merge. The graph capture requires all kernels to be stable.

**Commit:** `Patch 21: CUDA Graph capture for full training step`

### Sub-task 22a: Stream double-buffer + CUDA Graphs integration
**Research:** `research-kernel-efficiency/06_stream_overlap.md`
**Paper:** (stream overlap, no specific paper)

**File:** `scripts/train_qwen.py:1058-1103`

**What:** Integrate the existing stream double-buffer (teacher on stream_t, student on default stream) with the CUDA Graph capture. The teacher graph runs on stream_t, the student+backward+optimizer graph runs on the default stream, with event-based synchronization.

**Implementation:**
```python
# Teacher graph (on stream_t)
teacher_graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(teacher_graph, stream=stream_t):
    # Teacher forward → h_out_buf[buf_idx]
    ...

# Student graph (on default stream)
student_graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(student_graph, stream=torch.cuda.current_stream()):
    # Wait for teacher
    torch.cuda.current_stream().wait_event(event_t[buf_idx])
    # Student forward + loss + backward + optimizer
    ...

# Replay per step
def training_step(batch_ids):
    static_batch_ids.copy_(batch_ids)
    # Teacher on stream_t
    stream_t.wait_event(event_s[buf_idx])
    teacher_graph.replay()
    event_t[buf_idx].record(stream_t)
    # Student on default stream
    student_graph.replay()
    event_s[buf_idx].record(torch.cuda.current_stream())
    buf_idx = 1 - buf_idx
```

**Commit:** `Patch 22: stream double-buffer + CUDA Graphs integration (teacher hidden behind student)`

### Sub-task 22b: Send messages + push
- Send to orchestrator: "CUDA Graphs ready — tps target achievable. All 6 agents' work merged."
- Update PROGRESS.md.
- Push.

**Commit:** `Wave 4 closeout: PROGRESS.md + final msg to orchestrator`

---

## DoD
- [ ] All syntax checks pass
- [ ] CUDA Graph capture code present in train_qwen.py
- [ ] Static input buffers (batch_ids) pre-allocated
- [ ] Stream double-buffer integrated with graph replay
- [ ] Teacher graph on stream_t, student graph on default stream
- [ ] Event-based synchronization between teacher and student graphs
- [ ] Branch pushed to origin

## Offline Constraint
CUDA Graph capture code can be WRITTEN but NOT TESTED (requires GPU). Write the code carefully — the syntax must be correct even though the graph capture itself can only be validated on the server.
