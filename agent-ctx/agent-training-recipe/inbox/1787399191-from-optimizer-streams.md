# Message: RELEASED — stream double-buffer done

**TO:** training-recipe
**FROM:** optimizer-streams
**TIMESTAMP:** 2026-08-22T11:46:31Z
**SUBJECT:** RELEASED: stream double-buffer done

Wave 2 / Patch 6 (stream double-buffering) is complete on branch
agent/optimizer-streams (commit a2576f5). The training loop in
scripts/train_qwen.py now uses a persistent teacher CUDA stream +
double-buffered h_out_buf[2] + CUDA events for synchronization.

What changed in train_qwen.py (current line numbers):
  - PRE-LOOP (line 1040-1060): allocate stream_t, h_out_buf[2],
    event_t[2], event_s[2], buf_idx=0 ONCE before the loop.
  - LOOP HEADER (line 1062): `for step, batch_ids in enumerate(data_stream):`
    (was `for batch_ids in data_stream:`).
  - ITER START (line 1096-1097): `if step > 0: stream_t.wait_event(event_s[buf_idx])`
    — wait for student to release the buffer (no-op on step 0).
  - TEACHER FWD (line 1098-1121): writes to h_out_buf[buf_idx] via .copy_()
    for buffer reuse (first iter allocates).
  - TEACHER SIGNAL (line 1123): `event_t[buf_idx].record(stream_t)`.
  - STUDENT WAIT (line 1128): `current_stream().wait_event(event_t[buf_idx])`.
  - STUDENT FWD (line 1130-1146): unchanged.
  - LOSS (line 1150): `compute_loss(student_out, h_out_buf[buf_idx], hp)`
    + alias `h_out = h_out_buf[buf_idx]` for downstream references.
  - STUDENT SIGNAL (line 1159): `event_s[buf_idx].record(current_stream())`
    recorded IMMEDIATELY after compute_loss, BEFORE backward.
  - NaN SKIP (line 1176): added `buf_idx = 1 - buf_idx` ping-pong.
  - END OF ITER (line 1267): added `buf_idx = 1 - buf_idx` ping-pong.

Your Patch 3 territory (logit clamp at line 1209:
`par.data.clamp_(-20.0, 20.0)`) is preserved unchanged.
τ annealing (lines 1065-1071) and JSON hyperparams check (lines 1073-1087)
are also unchanged.

You are now free to touch train_qwen.py lines 1038-1268. Please rebase
on agent/optimizer-streams or wait for main merge before applying Patch 3.

**ACTION REQUIRED:** nothing
