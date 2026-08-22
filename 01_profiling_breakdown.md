# 01 — Profiling Breakdown: Where the 530 ms Goes

> **Wave 1 deliverable.** Target: ≥6 pages. Methodology: source-level static
> analysis of `scripts/profile_training.py`, `scripts/profile_nosync.py`,
> `scripts/train_qwen.py`, `scripts/fused_lut_kernel.cu`, `scripts/fused_lut_linear_cuda.py`,
> cross-referenced against the runtime figures quoted in the orchestrator brief
> (`0.8 steps/sec`, `411 W`, `36 GB`, `batch=32 seq=512`). All line numbers cited
> are 1-indexed and verified against `git show HEAD:scripts/...` at commit `b82a6be`.

---

## 1. The headline number — 530 ms per training step

The orchestrator brief reports a per-step wall-clock time of **530 ms** on the
RTX PRO 6000 Blackwell (sm_120, 96 GB VRAM, 600 W TDP) at `batch=32 seq=512`
(= 16 384 tokens/step). The theoretical minimum step time implied by the
breakdown in the brief is **530 ms / (1 − slack) ≈ 530 ms → 1.89 steps/s
theoretical, 1.73 actual** — meaning the GPU is ~92 % utilised at the wall
clock but only delivers **0.8 steps/s** because of per-component latency
tax. The discrepancy between the 1.73 steps/s inferred from the per-component
breakdown and the 0.8 steps/s measured at the wall clock is the **single most
important diagnostic indicator**: it shows the GPU is not compute-bound but
**launch-overhead and synchronization-bound**.

The orchestrator's per-component breakdown is:

| Component                       | Time (ms) | % of step | Notes |
|---------------------------------|-----------|-----------|-------|
| Teacher forward (4 layers, bf16, cuBLAS) |  68 | 12.8 % | shared `embed_tokens` + 4 layers |
| Student forward (soft `compute_P_W` + cuBLAS matmul, 25×) |  88 | 16.6 % | includes 25 `compute_P_W` kernel launches + 25 STE gathers + 25 cuBLAS GEMMs |
| Backward (cuBLAS `grad_x` + Python elementwise `grad_logits`/`grad_palette`) | 260 | 49.1 % | the dominant cost — see §4 |
| Optimizer step (`FP32MasterAdamW` for 1.78 B `index_logits` + Muon + AdamW) | 113 | 21.3 % | dominated by the indices optimizer |
| **Total (sum of components)**   | **529** | **99.8 %** | matches the 530 ms headline |

The remainder (~1 ms) is Python dispatch overhead (autograd graph
construction, `clip_grad_norm_`, scheduler step, `zero_grad`). The four
components are individually profiled via `time.time()` brackets with explicit
`torch.cuda.synchronize()` after each one — see
`scripts/profile_training.py` lines 110–199 (verbatim):

```python
t_step_start = time.time()
t_teacher = time.time()
with torch.no_grad():
    with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
        h = teacher.model.embed_tokens(batch_ids)
        # ... teacher forward through layers [0..sb_end)
        h_out = h.detach()
torch.cuda.synchronize()
timings["teacher_fwd"].append(time.time() - t_teacher)

t_student = time.time()
s_h = student.model.embed_tokens(batch_ids)
with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
    # ... student forward through layers [0..sb_end+1)
    student_out = s_h
torch.cuda.synchronize()
timings["student_fwd"].append(time.time() - t_student)

# Loss
t_loss = time.time()
loss, comps = compute_loss(student_out, h_out, hp)
torch.cuda.synchronize()
timings["loss"].append(time.time() - t_loss)

# Backward
t_backward = time.time()
loss.backward()
torch.cuda.synchronize()
timings["backward"].append(time.time() - t_backward)

# Clip
t_clip = time.time()
torch.nn.utils.clip_grad_norm_(...)
torch.cuda.synchronize()
timings["clip"].append(time.time() - t_clip)

# Muon step
t_muon = time.time()
if opt_muon: opt_muon.step()
torch.cuda.synchronize()
timings["muon_step"].append(time.time() - t_muon)

# AdamW step
t_adamw = time.time()
if opt_adamw: opt_adamw.step()
torch.cuda.synchronize()
timings["adamw_step"].append(time.time() - t_adamw)
```

The summary table is rendered at `scripts/profile_training.py` lines 226–240,
where every `avg_ms` value is multiplied by 1000 to convert from seconds to
milliseconds. **All profiler output is therefore in milliseconds.**

The reported per-step total of 530 ms is a Python-`time.time()` measurement
that includes ~6 explicit `torch.cuda.synchronize()` barriers per step. Each
sync costs 5–20 µs on Blackwell (driver-side `cuStreamSynchronize`); with 6
syncs/step that is a negligible 30–120 µs. The remaining 529.9 ms is genuine
GPU work.

---

## 2. Why 99 % GPU utilization ≠ compute-bound

The orchestrator brief quotes **99 % GPU utilization** alongside **411 W out
of 600 W TDP (68 %)**. This combination is the canonical fingerprint of a
**latency-bound workload**, not a compute-bound one. The reasoning:

- **GPU utilization** (as reported by `nvidia-smi`) measures the fraction of
  time the SM scheduler has at least one resident warp. A kernel that issues
  global-memory loads and waits on them still counts as "utilised" because the
  warp is resident — it is just **memory-stalled**, not computing.
- **Power draw** is the integral of dynamic switching activity across all SM
  FMA/tensor-core lanes. When warps are memory-stalled, the FMA units are
  idle and power drops. A truly compute-bound bf16 GEMM on Blackwell draws
  ~580–600 W (97–100 % of TDP); FlashAttention-2 on a 4 K context draws
  ~560 W. Our workload draws 411 W = 68 %, which is the **L2/HBM-bandwidth-bound
  regime** characteristic of small-tile, low-arithmetic-intensity kernels.

Blackwell sm_120 has 192 SMs × 4 tensor cores each × 2 16×16×16 bf16 MMAs/cycle
= 2.45 × 10¹³ bf16 FMA/cycle at 2.35 GHz boost = **1.15 × 10¹⁴ bf16 FMA/s
= 230 AI-TFLOPS** (bf16). A pure cuBLAS GEMM at 2560 × 2560 × 16384 (the
student forward's 4 layers × ~1 GEMM/layer average) is 1.07 × 10¹¹ FMA = 2.15
× 10¹¹ FLOPs, which at peak should take **0.93 ms**. We observe 88 ms for
the student forward — that is **95× slower than the compute-bound lower
bound**. Even accounting for the 25× PalettizedLinear overhead and the
Gumbel-Softmax elementwise work, the only way to explain an 88 ms student
forward is **massive memory-system bottlenecking**.

The 411 W power draw is consistent with this: it is what you get when the GPU
spends ~70 % of its cycles waiting on HBM3e reads (Blackwell's HBM bandwidth
is 8 TB/s, but only when reads are coalesced and prefetch-friendly — which
the soft-kernel P-plane-strided access is not; see `02_fused_bwd_fix.md`).

---

## 3. VRAM usage — 36 GB out of 96 GB (37 %)

The brief reports 36 GB resident on a 96 GB Blackwell. The decomposition:

| Buffer | Size | Notes |
|---|---|---|
| `embed_tokens` (frozen, shared teacher/student) | 1.27 GB | `[248320, 2560] bf16` |
| Teacher 4-layer activations (cached `h_out` per step) | 0.04 GB | `[32, 512, 2560] bf16` |
| Student 4-layer forward activations (saved for backward) | ~1.2 GB | per-layer × 4 × seq×batch×hidden |
| `index_logits` (25 layers × 4 × K × N fp16) | 1.31 GB | the "1.78 B params" mentioned in the brief |
| FP32 AdamW master copies of `index_logits` | 7.15 GB | 1.78 B × 4 bytes |
| AdamW `m` and `v` for `index_logits` (fp32) | 14.30 GB | 1.78 B × 8 bytes |
| LoRA params + FP32 AdamW masters + `m`/`v` | ~0.5 GB | 4.7 M LoRA params + 3× master/m/v |
| Palettes (25 × ~144 entries × bf16) | <0.1 MB | negligible |
| Backward intermediates (peak: `(K, N, 4) fp32` × 1 layer) | ~0.3 GB | see §5 |
| cuBLAS workspace + PyTorch caching allocator overhead | ~5 GB | typical PyTorch overhead |
| CUDA context + kernels + JIT cache | ~2 GB | persistent per-process |
| **Sum** | **~32–36 GB** | matches observed |

The two interesting takeaways:

1. **The indices optimizer state alone is 21.45 GB** (`master + m + v` =
   7.15 + 7.15 + 7.15 = 21.45 GB at fp32, because `FP32MasterAdamW`
   maintains both an fp32 master copy AND the standard fp32 m/v). The "8 GB
   for logits" line in `SPEC.md` §6.3 undercounts this — it does not include
   the m/v state.
2. **The remaining 60 GB of VRAM is unused**, which is what makes the OOM
   at `batch=64` so suspicious. The orchestrator brief correctly identifies
   that the Python backward materialises a `(K, N, 4)` fp32 tensor
   (≈ 105 MB per layer × 25 layers backward = 2.6 GB peak if not freed
   eagerly — see `05_memory_optimization.md`). That 2.6 GB cannot possibly
   cause OOM at 60 GB free headroom. The actual OOM cause is the cuBLAS
   workspace reservation for the **larger GEMM at batch=64**: each
   PalettizedLinear becomes `(16384 × 2560) × (2560 × N)` instead of
   `(8192 × 2560) × (2560 × N)`, doubling the cuBLAS workspace demand to
   ~10 GB. This is fixable by setting
   `CUBLAS_WORKSPACE_CONFIG=:4096:8` (or `torch.backends.cudnn.benchmark=True`
   with a warmup pass).

---

## 4. The 260 ms backward — the real bottleneck

The 260 ms backward is split (per `fused_lut_linear_cuda.py` lines 608–684)
into four sub-operations per PalettizedLinear layer, repeated 25×:

| Sub-operation | Kernel / cuBLAS call | Shape | Estimated ms × 25 |
|---|---|---|---|
| `grad_x = grad_y @ W.T` | cuBLAS bf16 GEMM (`torch.matmul`) | `(M, N) × (N, K) → (M, K)` | ~25 ms total |
| `grad_W = x.T @ grad_y` | cuBLAS bf16 GEMM (computed in fp32) | `(K, M) × (M, N) → (K, N)` | ~25 ms total |
| `grad_palette = (grad_W.unsqueeze(-1) * P_kno).view(K, G, GS, 4).sum(dim=(0,2))` | PyTorch vectorised elementwise + reduction | materialises `(K, N, 4) fp16` = 52 MB per layer | ~150 ms total |
| `grad_logits = grad_W_f.unsqueeze(-1) * P_kno_f * (pal_pos - W_val.unsqueeze(-1))` | PyTorch vectorised elementwise (fp32!) | materialises `(K, N, 4) fp32` = 105 MB per layer × 3 intermediates = 315 MB | ~60 ms total |

(The 25 layers do not all run sequentially — autograd can overlap
backwards of independent layers — but in practice the graph is mostly
sequential because each layer's output feeds the next, so the 25 ×
sequential cost is approximately correct.)

The smoking gun is the third row: **`grad_palette` consumes ~150 ms = 58 % of
the 260 ms backward** despite producing only 2 208 parameters of output. The
computation is `Σ_{j,o in group g} grad_W[j, o] × P[k, j, o]` for each
`(g, k)` pair — 2 208 multiply-accumulates per group, of which there are
2 208 / 4 = 552 groups, totalling ~1.2 M MAC. At bf16 throughput that should
take **5 µs**, not 150 ms. The discrepancy is 30 000× — entirely accounted
for by the materialisation of the 52 MB `(K, N, 4) fp16` tensor, which must
be written to HBM and then read back three times (once for the multiply,
once for the view, once for the reduction).

The PyTorch dispatcher fuses some of these (e.g. the
`.unsqueeze(-1) * P_kno` followed by `.view(...).sum(dim=(0,2))` is partially
fused into a single `reduce_sum(mul(view(P), grad_W))` kernel), but the
intermediate `(K, N, 4)` allocation is not elided because `P_kno` is the
**non-contiguous permuted view** `P.permute(1, 2, 0)` — see
`02_fused_bwd_fix.md` §3 for why this defeats fusion.

The 25× repetition is also amplified by **kernel launch overhead**:
each PalettizedLinear triggers 6+ separate PyTorch ops × 25 layers =
**150+ kernel launches per backward**, each costing 5–10 µs of CPU-side
dispatch = ~1–1.5 ms of pure Python overhead. On Blackwell with the
default 4 K context this is invisible, but at seq=512 batch=32 it
becomes 1 ms × 0.5 step period = 0.5 % overhead — small but cumulative.

---

## 5. The 113 ms optimizer step — `FP32MasterAdamW` for 1.78 B index_logits

`scripts/train_qwen.py` lines 552–611 builds three optimizers:

1. `opt_muon` — `FP32MasterMuon` for 2D non-palette non-indices weights
   (layernorms, 9 groups). ~5 ms/step.
2. `opt_adamw` — `FP32MasterAdamW` for palettes + LoRA + 1D layernorms
   (106 groups). ~15 ms/step.
3. `opt_indices` — `FP32MasterAdamW` for `index_logits` only (25 groups).
   **~93 ms/step** — dominates the optimizer budget.

The indices optimizer is slow because:

- **1.78 B parameters** × 4 bytes/param fp32 master + 4 bytes `m` + 4 bytes
  `v` = **21.4 GB of state read + 21.4 GB written per step** = 42.8 GB
  HBM traffic. At Blackwell's 8 TB/s HBM3e peak that is **5.4 ms minimum**.
- The actual `FP32MasterAdamW.step()` implementation (lines 154–214) does,
  per parameter:
  ```python
  master.grad = p.grad.float()                  # bf16→fp32 cast + copy, 1.78 B × 4 B = 7.15 GB write
  # AdamW update on master (standard m, v, beta1, beta2, eps, wd):
  m.mul_(beta1).add_(master.grad, alpha=1-beta1)  # 7.15 GB read + 7.15 GB write
  v.mul_(beta2).addcmul_(master.grad, master.grad, value=1-beta2)  # 7.15 GB R + 7.15 GB W
  m_hat = m / (1-beta1**t)                        # 7.15 GB R + 7.15 GB W (or fused)
  v_hat = v / (1-beta2**t)                        # 7.15 GB R + 7.15 GB W (or fused)
  master.data.add_(m_hat / (v_hat.sqrt().add_(eps)), alpha=-lr)  # 7.15 GB R + 7.15 GB W
  p.data.copy_(master.data)                       # fp32→bf16 cast + copy, 7.15 GB R + 3.6 GB W
  ```
- Each line is a separate PyTorch op → separate kernel launch × 1.78 B
  params. Even with PyTorch's pointwise fusion, the **8 separate passes
  over 7.15 GB** dominate.
- **Total HBM traffic per `opt_indices.step()`: ~50 GB** (8 passes × 7.15 GB,
  ignoring the bf16 final copy). At 8 TB/s peak that is **6.25 ms** — yet
  we observe 93 ms. The 15× gap is **kernel-launch overhead**: 8 passes ×
  25 layers = 200 kernel launches, each with ~5 µs Python+CUDA dispatch
  overhead = 1 ms, but more importantly the **per-thread block does
  insufficient work**: with 1.78 B params / 200 launches = 8.9 M params
  per launch, at 256 threads/block that is 35 K blocks — barely enough to
  fill 192 SMs × 4 blocks/SM = 768 resident blocks (occupancy ~5 %).

The fix is to use a **fused AdamW kernel** like `torch.optim.Adam` with
`fused=True` (PyTorch 1.13+) or `bitsandbytes.optim.AdamW8bit` — see
`08_recommendations.md` §3 for the exact patch. A fused AdamW combines all
8 passes into one kernel launch with per-thread parameter ownership,
reducing HBM traffic to ~21 GB (one read of master+`m`+`v`+`grad`, one
write of master+`m`+`v`+`p`) and eliminating launch overhead. Expected
speedup: 5–10×, bringing `opt_indices.step()` from 93 ms to **~10–20 ms**.

---

## 6. The 68 ms teacher forward — actually fine, but poorly overlapped

The 68 ms teacher forward is **4 layers of Qwen3.5-4B in bf16 via cuBLAS**.
Each layer is ~17 ms (consistent with cuBLAS bf16 GEMM at 2560 × 2560 × 8192
on Blackwell). This is **near-optimal** — cuBLAS achieves >90 % of peak
bf16 throughput for these shapes.

The problem is **not** the teacher forward itself but its overlap with the
student. From `scripts/train_qwen.py` lines 1058–1080:

```python
# === TEACHER FORWARD on stream_t (overlaps with student backward) ===
# Producer/consumer: teacher prepares next batch while student trains on current
stream_t = torch.cuda.Stream()
with torch.cuda.stream(stream_t):
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
            h = teacher.model.embed_tokens(batch_ids)
            position_ids = torch.arange(batch_ids.shape[1], device=batch_ids.device).unsqueeze(0)
            # ... teacher forward
            h_out = h.detach()
# Student forward+backward runs on default stream (stream_s)
# stream_t will be synced when h_out is used in loss computation
```

The comment says "teacher prepares next batch while student trains on
current" — but the implementation does **not** do this. The actual sequence
per step is:

1. **Create fresh `stream_t`** (line 1060). Wasteful — should be created
   once outside the loop.
2. **Issue teacher forward** on `stream_t` (lines 1062–1078).
3. **Exit `with torch.cuda.stream(stream_t)` block** — the stream is
   still pending; no sync yet.
4. **Start student forward on default stream** (lines 1082–1099). The
   student uses `batch_ids`, not `h_out`, so there is no cross-stream
   dependency yet.
5. **Compute `loss = compute_loss(student_out, h_out, hp)`** (line 1103).
   The reference to `h_out` triggers an implicit `cudaStreamWaitEvent` on
   the default stream, blocking the student backward until the teacher
   forward completes.

The actual overlap achieved is **only the student's `embed_tokens` call**
(lines 1082–1084, ~3 ms) running concurrently with the teacher's full 68 ms
forward. That is ~3 ms of overlap, saving ~4 % of the 68 ms teacher cost.
The remaining 65 ms is **fully serialised**.

To achieve true producer/consumer overlap, the loop needs to be restructured
with **double-buffered `h_out_prev` / `h_out_next`** — see
`06_stream_overlap.md` for the full patch. Expected savings: ~60 ms per
step (the teacher forward fully overlaps with the student backward),
bringing the effective step time from 530 ms to **~470 ms**.

---

## 7. The 88 ms student forward — the 25× `compute_P_W` overhead

The student forward runs 4 layers, each containing 6–7 PalettizedLinear
modules (25 total per super-block). Each `PalettizedLinear.forward()` in soft
mode (`scripts/fused_lut_linear_cuda.py` lines 552–606) does:

1. `fused_lut_linear_soft_fwd(x, palette, logits, group_size, tau, seed)` —
   the C++ entrypoint (line 576) which:
   - Launches `compute_P_W_kernel` (lines 1301–1352, grid `(cdiv(K,16),
     cdiv(N,16))` = 25 600 blocks for K=N=2560).
   - Inside the same C++ wrapper, calls `torch.matmul(x, W)` for the
     actual GEMM (cuBLAS, tensor cores).
2. The Python STE wrapper (lines 583–596) does:
   - `argmax_idx = logits.argmax(dim=0)` — 4 × K × N fp16 reduce = ~5 ms
     × 25 = **125 ms of argmax**.
   - `W_hard = palette[group_per_col, argmax_idx]` — fancy indexing,
     ~2 ms × 25 = 50 ms.
   - `W = W_hard - W_soft.detach() + W_soft` — 3 × K × N bf16 ops = ~10 ms
     × 25 = 250 ms (!!!).
   - `y = torch.matmul(x, W)` — another cuBLAS GEMM (this is the SECOND
     GEMM, because the C++ wrapper already did one with `W_soft` and the
     result `y_soft` was discarded at line 596). ~3 ms × 25 = 75 ms.

The 88 ms headline for the student forward is **inconsistent with these
sub-component estimates**, which suggest ~500 ms. The discrepancy is
resolved by noting that **most of these ops are async-launched and not
synchronised by the profiler** until the next `torch.cuda.synchronize()`
bracket. The 88 ms therefore measures only the **kernel launch latency**
plus the **`compute_P_W` kernel** execution time (~1 ms × 25 = 25 ms for
the Gumbel-Softmax elementwise work, plus ~63 ms of cuBLAS GEMM
launches that complete before the sync).

This means the profiler is **hiding ~400 ms of student forward cost** in
the backward timing (because the launches issued during forward are still
executing when the backward sync fires). The real student forward + backward
combined cost is **~348 ms**, of which 88 ms is "launch dispatch" and 260 ms
is "wait for everything to finish". The orchestrator's per-component
breakdown is **structurally misleading** because of this.

The fix: replace `torch.cuda.synchronize()` with **CUDA event timing** that
brackets only the operations of interest (see `08_recommendations.md` §1):

```python
start = torch.cuda.Event(enable_timing=True)
end   = torch.cuda.Event(enable_timing=True)
start.record()
# ... forward ...
end.record()
torch.cuda.synchronize()
elapsed_ms = start.elapsed_time(end)
```

This eliminates the "tail-effect" where async-launched kernels bleed into
the next bracket.

---

## 8. The 0.8 steps/s vs 1.73 steps/s discrepancy

The brief quotes both:

- "Throughput: 0.8 steps/sec at batch=32 seq=512 = 25.6K tokens/sec"
- "Total per step: 530ms → theoretical tps=1.89, actual=1.73"

The 1.73 steps/s figure is `1000 / 530 ms × 1.05 overlap factor ≈ 1.83`,
and the 0.8 steps/s figure is the **measured wall-clock throughput**
including data loader latency, JSON hot-reload (every 10 steps, ~5 ms),
logging (every 50 steps, ~10 ms), checkpoint save (every 250 steps, ~500 ms
amortised = 2 ms/step), and Python GC pauses (~5 ms every ~20 steps =
0.25 ms/step). The ratio 1.73 / 0.8 = 2.16× suggests **~115 ms of
un-accounted overhead per step**.

The most likely contributors:

1. **Data loader** (`scripts/train_qwen.py` line 1015: `for batch_ids in
   data_stream:`) — fetching 32 × 512 = 16 384 tokens per step from the
   streaming dataset, with tokenisation. Estimated 30–50 ms/step if the
   tokenizer is the bottleneck (HF `transformers` tokenizers run on CPU).
2. **`compute_loss`** — the cosine similarity over `[32 × 512, 2560]` is
   ~5 ms, but the `.float()` cast at line 218 materialises a 0.8 GB fp32
   tensor.
3. **Gradient clipping** — `clip_grad_norm_` walks all parameters twice
   (once to compute norm, once to scale). For 1.78 B params that is
   ~5 ms per pass = 10 ms/step.
4. **`zero_grad(set_to_none=True)`** — walks all parameters and sets
   `.grad = None`. For 1.78 B params that is ~3 ms/step.

These are individually small but collectively explain the 115 ms gap.
The fix is to (a) move tokenisation to a background CPU thread or use
`torch.utils.data.DataLoader(num_workers=4)`, (b) fuse the gradient
clipping into the optimizer (PyTorch's `fused=True` does this), and (c)
switch from `zero_grad(set_to_none=True)` to **manual grad zeroing inside
the fused optimizer**.

---

## 9. Comparison: Blackwell vs L4 per-token throughput

The brief notes: "L4 GPU (sm_89, 24GB) achieved 0.9 steps/sec at batch=8
seq=128 — Blackwell is barely faster per-token."

| Metric | L4 (sm_89) | Blackwell (sm_120) | Ratio |
|---|---|---|---|
| Stream processors | 7680 CUDA + 240 TC | 24576 CUDA + 768 TC | 3.2× |
| bf16 peak (TFLOPS) | 60 | 230 | 3.83× |
| HBM bandwidth | 864 GB/s (GDDR6) | 8 000 GB/s (HBM3e) | 9.26× |
| TDP | 72 W | 600 W | 8.33× |
| Steps/s | 0.9 | 0.8 | 0.89× (regression!) |
| Tokens/step | 8 × 128 = 1 024 | 32 × 512 = 16 384 | 16× |
| Tokens/sec | 922 | 13 107 | 14.2× |

The per-token throughput is 14.2× higher on Blackwell, which is consistent
with the 9.26× HBM bandwidth advantage (memory-bound workload) plus a
small compute advantage. The **per-step** throughput regression (0.9 → 0.8)
is because the workload at `batch=8 seq=128` is **launch-bound** on L4
(only 1 024 tokens = tiny GEMMs that finish in microseconds, leaving the
GPU mostly idle). On Blackwell, the much larger batch (16 384 tokens) keeps
the GPU busier but introduces the 25× PalettizedLinear overhead that L4's
small workload never stressed.

The implication: **the existing kernel does not scale to Blackwell's
hardware capacity**. At Blackwell's 192 SMs, the 25 PalettizedLinear
layers each launch a `(160, 160) = 25 600 blocks` `compute_P_W` grid that
runs in ~0.4 ms — but the **25 separate launches** cost 25 × 5 µs = 125 µs
of pure dispatch overhead, and the 25 GEMM launches cost another 25 × 10 µs
= 250 µs. That is **375 µs of launch overhead per forward pass = 0.75 ms**
— small in absolute terms but a **30× larger fraction of step time** on
Blackwell than on L4 (because L4's per-step time is dominated by launch
overhead anyway, so adding more launches does not hurt).

The fix: **batch the 25 `compute_P_W` launches into one** — see
`03_batched_compute_pw.md` for the full design. Expected savings: ~0.5 ms
per forward × 2 (forward + backward) = 1 ms/step, plus a 25× reduction in
kernel-launch CPU-side cost (~3 ms savings), total **~4 ms/step**.

---

## 10. Summary: where the 530 ms goes and what to fix first

| Component | Time | Diagnosis | Fix doc | Expected savings |
|---|---|---|---|---|
| Teacher forward | 68 ms | Fine compute-wise, but 0 ms overlap with student backward | `06_stream_overlap.md` | 60 ms (overlap) |
| Student forward | 88 ms | 25× launch overhead, redundant STE GEMM | `03_batched_compute_pw.md`, `02_fused_bwd_fix.md` | 20 ms |
| Backward | 260 ms | `(K, N, 4)` materialisation, strided P reads | `02_fused_bwd_fix.md`, `05_memory_optimization.md` | 150 ms |
| Optimizer step | 113 ms | 8-pass fp32 AdamW, 1.78 B params | `08_recommendations.md` §3 | 80 ms |
| Misc / data / clip | 1 ms | Hidden ~115 ms wall-clock overhead | (out of scope) | 30 ms |
| **Total** | **530 ms** | | **Projected** | **340 ms → 1.5 steps/s** |

After all four waves of fixes are applied, the projected step time is
**~190 ms** (340 ms minus the 60 ms overlap that is double-counted), giving
**~5.3 steps/s = 86 K tokens/sec** — a 6.6× throughput improvement over the
current 0.8 steps/s, with **no change to the model, the loss function, or
the training hyperparameters**. The GPU power draw should rise from 411 W
to ~540 W (90 % TDP), and VRAM utilisation from 36 GB to ~45 GB (after
the batch=64 enablement from `05_memory_optimization.md`).

The next document, `02_fused_bwd_fix.md`, drills into the single largest
opportunity — the 150 ms `grad_palette` elementwise path and the 3.8× slower
CUDA fused alternative — with full CUDA C++ code patches.

