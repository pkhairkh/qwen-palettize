# 02 — Kernel Efficiency Patches (5-7)

> **Wave 2 deliverable.** Full detail for the 3 kernel efficiency enhancements. Code patches included as REFERENCE ONLY — not yet applied.

---

## Patch 5: Fused Backward Kernel with AoS P Layout

**Source:** [Agent 2 (kernel-efficiency), `02_fused_bwd_fix.md`, `08_recommendations.md` Patches 2+4; Agent 3 (indices-training), `01_gumbel_softmax_audit.md`]

### Problem

The fused CUDA backward kernel (`fused_lut_linear_soft_bwd_fused_kernel`) exists in `fused_lut_kernel.cu:1498` but is **10x slower** than the Python elementwise path (2904ms vs 260ms per step).

**Root cause (Agent 2, `02_fused_bwd_fix.md`):** The kernel reads P in `(4, K, N)` SoA layout. To access all 4 P values for a single `(j, o)` position, the kernel issues 4 strided loads spanning 78.6 MB of address space (26 MB stride between planes). This causes:
- 4 separate L2 cache line fetches per thread
- No memory coalescing
- 4× the HBM bandwidth pressure

The Python path (`fused_lut_linear_cuda.py:650-680`) materializes `(K, N, 4)` via `P.permute(1, 2, 0)` which is coalesced but allocates ~100MB intermediates.

### Fix

**Change P storage layout from `(4, K, N)` SoA to `(K, N, 4)` AoS.**

With AoS layout, the 4 P values per `(j, o)` are adjacent in memory — a single 64-bit `LDG.E.U64` load replaces 4 strided 26MB-apart loads. This:
- Eliminates strided access (4 loads → 1 load)
- Enables memory coalescing
- Makes the fused CUDA kernel faster than Python (no intermediates)

### Expected Impact

- **Backward time: 260ms → ~60ms** (saves ~200ms/step)
- **VRAM: eliminates ~7GB of Python intermediates** ((K,N,4) fp32 × 25 Linears)
- Enables batch=64 (was OOM at batch=32 with Python path)
- Re-enables the fused kernel that was disabled as "3.8x slower"

### Code Patch (REFERENCE ONLY — not yet applied)

**Files:** `scripts/fused_lut_kernel.cu`, `scripts/fused_lut_linear_cuda.py`

#### Step 1: New compute_P_W kernel with AoS output

**File:** `scripts/fused_lut_kernel.cu` (new kernel, ~50 lines)

```cuda
// NEW: compute_P_W with AoS output layout (K, N, 4) instead of (4, K, N)
__global__ void fused_lut_linear_soft_compute_P_W_aos_kernel(
    const __half*        __restrict__ logits,    // (4, K, N) fp16 — INPUT stays SoA
    const __nv_bfloat16* __restrict__ palette,   // (G, 4) bf16
    __half*              __restrict__ P_aos,     // (K, N, 4) fp16 — OUTPUT is AoS
    __nv_bfloat16*       __restrict__ W_out,     // (K, N) bf16
    int K, int N, int group_size,
    float tau, uint32_t step_seed
) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int o = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= K || o >= N) return;

    const int g = o / group_size;
    const int idx_soA = j * N + o;          // index into SoA logits (4,K,N)
    const int idx_aos = (j * N + o) * 4;    // index into AoS P (K,N,4)
    const int plane_size = K * N;

    // Load 4 logits from SoA layout
    float l0 = __half2float(logits[0 * plane_size + idx_soA]);
    float l1 = __half2float(logits[1 * plane_size + idx_soA]);
    float l2 = __half2float(logits[2 * plane_size + idx_soA]);
    float l3 = __half2float(logits[3 * plane_size + idx_soA]);

    // Gumbel + softmax (same as before)
    float inv_tau = 1.0f / tau;
    float n0 = (l0 + gumbel_sample(step_seed, idx_soA * 4 + 0)) * inv_tau;
    float n1 = (l1 + gumbel_sample(step_seed, idx_soA * 4 + 1)) * inv_tau;
    float n2 = (l2 + gumbel_sample(step_seed, idx_soA * 4 + 2)) * inv_tau;
    float n3 = (l3 + gumbel_sample(step_seed, idx_soA * 4 + 3)) * inv_tau;
    float m = fmaxf(fmaxf(n0, n1), fmaxf(n2, n3));
    float e0 = expf(n0 - m), e1 = expf(n1 - m), e2 = expf(n2 - m), e3 = expf(n3 - m);
    float s = e0 + e1 + e2 + e3;
    float p0 = e0/s, p1 = e1/s, p2 = e2/s, p3 = e3/s;

    // Write P in AoS layout — 4 adjacent fp16 values
    P_aos[idx_aos + 0] = __float2half(p0);
    P_aos[idx_aos + 1] = __float2half(p1);
    P_aos[idx_aos + 2] = __float2half(p2);
    P_aos[idx_aos + 3] = __float2half(p3);

    // Compute W = sum(P * palette)
    float c0 = __bfloat162float(palette[g * 4 + 0]);
    float c1 = __bfloat162float(palette[g * 4 + 1]);
    float c2 = __bfloat162float(palette[g * 4 + 2]);
    float c3 = __bfloat162float(palette[g * 4 + 3]);
    W_out[idx_soA] = __float2bfloat16(p0*c0 + p1*c1 + p2*c2 + p3*c3);
}
```

#### Step 2: Update fused backward kernel to read AoS P

**File:** `scripts/fused_lut_kernel.cu` (modify `fused_lut_linear_soft_bwd_fused_kernel`, ~line 1498)

```cuda
// MODIFIED: read P in AoS (K,N,4) layout — single 64-bit load per (j,o)
__global__ void fused_lut_linear_soft_bwd_fused_aos_kernel(
    const __nv_bfloat16* __restrict__ grad_y,
    const __nv_bfloat16* __restrict__ x,
    const __half*        __restrict__ P_aos,    // (K, N, 4) fp16 — AoS INPUT
    const __nv_bfloat16* __restrict__ palette,
    __half*              __restrict__ grad_logits,  // (4, K, N) fp16 — OUTPUT stays SoA
    float*               __restrict__ grad_palette,
    int M, int K, int N, int group_size
) {
    // ... same as before but P access changes:
    const int idx_aos = (j * N + o) * 4;
    
    // Single 64-bit load gets all 4 P values (coalesced!)
    // P_aos[idx_aos + 0..3] are adjacent in memory
    float p0 = __half2float(P_aos[idx_aos + 0]);
    float p1 = __half2float(P_aos[idx_aos + 1]);
    float p2 = __half2float(P_aos[idx_aos + 2]);
    float p3 = __half2float(P_aos[idx_aos + 3]);
    
    // ... rest of kernel unchanged ...
}
```

#### Step 3: Switch Python backward to use fused kernel

**File:** `scripts/fused_lut_linear_cuda.py` (backward, ~line 620)

```python
# BEFORE (current — Python elementwise, 260ms):
        if needs_grad_logits or needs_grad_palette:
            grad_W = torch.matmul(x.T, grad_y)
            P_kno = P.permute(1, 2, 0).float()  # ← materializes (K,N,4) fp32
            # ... elementwise ops ...

# AFTER (proposed — fused CUDA kernel, ~60ms):
        if needs_grad_logits or needs_grad_palette:
            mod = _get_module()
            grad_logits, grad_palette = mod.fused_lut_linear_soft_bwd_fused_aos(
                grad_y, x, P_aos, palette, GS  # P_aos is (K,N,4), no permute needed
            )
```

### Verification

After applying:
1. Run `test_fused_bwd.py` (correctness test)
2. Verify `grad_logits` matches Python reference: `max_err < 1e-3`
3. Profile backward: should drop from 260ms → ~60ms
4. Check VRAM: should drop by ~7GB (no (K,N,4) fp32 intermediates)

### Dependencies

- Requires P to be stored as `(K, N, 4)` AoS (change in compute_P_W kernel output)
- STE forward (already applied) must be updated to use `P_aos` instead of `P`
- Existing checkpoint format unchanged (P is transient, not saved)

### Risks

- AoS layout changes the memory pattern — must update ALL kernels that read P
- If fused kernel still slower than Python (unlikely after AoS), keep Python path
- Test carefully: `max_err < 1e-3` vs Python reference

---

## Patch 6: Stream Double-Buffering

**Source:** [Agent 2 (kernel-efficiency), `06_stream_overlap.md`, `08_recommendations.md` Patch 1]

### Problem

Current CUDA stream setup (in `train_qwen.py`):
- Teacher forward on `stream_t`
- Student forward+backward on default stream
- BUT: `stream_t` is created fresh each step (5µs alloc/destroy overhead)
- AND: no double-buffering — teacher must finish before student can use `h_out`

Teacher forward (68ms) is sequential with student backward (260ms). Total = 328ms. With overlap, total = max(68, 260) = 260ms — saves 68ms.

### Fix

Restructure the training loop with:
1. **Persistent stream pool** (allocate `stream_t` once at startup)
2. **Double-buffered `h_out_buf[2]`** — teacher writes to buf[0] while student reads buf[1]
3. **CUDA events** for synchronization (`event_t[2]`, `event_s[2]`)

Teacher prepares batch N+1 while student trains on batch N.

### Expected Impact

- **Saves ~69ms/step** (teacher forward fully hidden behind student backward)
- Eliminates per-step stream creation overhead (~5µs × 25 Linears = 125µs)
- GPU utilization stays >95% (no gaps between teacher/student)

### Code Patch (REFERENCE ONLY — not yet applied)

**File:** `scripts/train_qwen.py`
**Lines:** ~1058-1103 (training loop)

```python
# BEFORE (current — fresh stream per step):
    for batch_ids in data_stream:
        # === TEACHER FORWARD on stream_t ===
        stream_t = torch.cuda.Stream()  # ← created fresh each step!
        with torch.cuda.stream(stream_t):
            with torch.no_grad():
                # ... teacher forward ...
                h_out = h.detach()
        # Student forward uses h_out (must wait for stream_t)

# AFTER (proposed — persistent stream + double-buffer):
    # Pre-loop: allocate persistent stream + double-buffered h_out + events
    stream_t = torch.cuda.Stream()  # ← allocated ONCE
    h_out_buf = [None, None]  # double-buffered teacher outputs
    event_t = [torch.cuda.Event(), torch.cuda.Event()]  # teacher done events
    event_s = [torch.cuda.Event(), torch.cuda.Event()]  # student done events
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
        
        # === TEACHER FORWARD for NEXT step (on stream_t, overlaps with student) ===
        if step + 1 < max_steps:
            next_batch = next(data_stream)  # or from prefetch loader
            stream_t.wait_event(event_s[nxt])  # wait if buf[nxt] still in use
            with torch.cuda.stream(stream_t):
                with torch.no_grad():
                    # ... teacher forward for next_batch ...
                    h_out_buf[nxt] = h.detach()
            event_t[nxt].record(stream_t)
        
        buf_idx = nxt
```

### Verification

After applying:
- GPU utilization should stay >95% (no gaps)
- Power draw should stay >400W (no idle periods)
- `nvidia-smi dmon` should show stream overlap
- Total step time should drop by ~69ms (teacher hidden)

### Dependencies

- Requires `next_batch` to be available before current step finishes (need prefetch loader — see Patch 7 or data pipeline cache)
- If data loading is slow, teacher forward will stall waiting for data

### Risks

- Double-buffering adds complexity — deadlocks possible if events not managed correctly
- `next(data_stream)` may block on network I/O (FineWeb-Edu streaming) — need data cache (Patch from architecture-review)

---

## Patch 7: Batched compute_P_W (25 Launches → 1)

**Source:** [Agent 2 (kernel-efficiency), `03_batched_compute_pw.md`, `08_recommendations.md` Patch 3]

### Problem

The soft forward calls `compute_P_W_kernel` **25 times per forward** (once per PalettizedLinear). Each kernel launch has ~5µs CPU-side dispatch overhead × 25 = 125µs wasted per forward. Plus, there's no L2 cache reuse between Linears.

### Fix

Fuse all 25 `compute_P_W` launches into a **single kernel launch** using:
- A `PalettizedLayerDesc[25]` array in constant memory (or device memory)
- `blockIdx.z` as the layer index (0-24)
- Each block processes one layer's (K, N) tile

### Expected Impact

- **Saves ~18ms/step forward** (25 launches → 1 launch)
- **Saves ~18ms/step backward** (same optimization for bwd kernels)
- L2 cache reuse: palette values for adjacent layers may stay in cache
- Total: ~36ms/step saved

### Code Patch (REFERENCE ONLY — not yet applied)

**Files:** `scripts/fused_lut_kernel.cu`, `scripts/fused_lut_linear_cuda.py`

#### Step 1: Define layer descriptor struct

**File:** `scripts/fused_lut_kernel.cu` (new struct)

```cuda
struct PalettizedLayerDesc {
    const __half*        logits;      // (4, K, N) fp16
    const __nv_bfloat16* palette;     // (G, 4) bf16
    __half*              P_aos;       // (K, N, 4) fp16 — AoS output
    __nv_bfloat16*       W_out;       // (K, N) bf16
    int K, N, G;
};
```

#### Step 2: Batched compute_P_W kernel

**File:** `scripts/fused_lut_kernel.cu` (new kernel)

```cuda
__global__ void fused_compute_P_W_batched_kernel(
    const PalettizedLayerDesc* __restrict__ layers,  // device array of 25 descriptors
    int n_layers, int group_size, float tau, uint32_t step_seed
) {
    const int layer_idx = blockIdx.z;
    if (layer_idx >= n_layers) return;
    
    const PalettizedLayerDesc& desc = layers[layer_idx];
    const int K = desc.K, N = desc.N;
    
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int o = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= K || o >= N) return;
    
    // ... same compute_P_W logic as Patch 5, but using desc.logits, desc.palette, etc. ...
}
```

#### Step 3: Python wrapper to build descriptor array + launch

**File:** `scripts/fused_lut_linear_cuda.py` (new function)

```python
def fused_compute_P_W_batched(student, tau, step_seed):
    """Compute P and W for ALL 25 PalettizedLinears in one kernel launch."""
    from qwen_model import PalettizedLinear
    
    # Collect all PalettizedLinear modules
    layers = []
    for name, mod in student.named_modules():
        if isinstance(mod, PalettizedLinear) and mod.index_logits is not None:
            layers.append(mod)
    
    # Build descriptor array on device
    # (each descriptor is 6 pointers + 3 ints = 72 bytes, ×25 = 1.8KB)
    descs = torch.tensor([...], device='cuda')  # pack pointers + shapes
    
    mod = _get_module()
    mod.fused_compute_P_W_batched_launcher(descs, len(layers), group_size, tau, step_seed)
```

### Alternative: CUDA Graphs

If the descriptor-based batching is too invasive, an alternative is **CUDA Graphs** (from Agent 2, `03_batched_compute_pw.md`):

```python
# Capture the 25 compute_P_W launches as a CUDA graph
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    for mod in palettized_linears:
        mod._soft_kernel(x, palette, logits, bias, group_size, tau)
# Replay as single dispatch
graph.replay()
```

This eliminates the 25× dispatch overhead without changing the kernel code, but doesn't get L2 cache reuse benefits.

### Verification

After applying:
- Profile forward: 25 `compute_P_W` launches → 1 launch
- Check `nsys profile` for kernel launch count
- Verify output matches per-Linear kernel: `max_err < 1e-4`
- Forward time should drop by ~18ms

### Dependencies

- Requires all 25 PalettizedLinears to have the same `group_size` (currently true: GS=256)
- If Patch 4 (per-tensor GS override) is applied, batched kernel needs to handle variable GS
- Compatible with Patch 5 (AoS P layout) — both change P storage

### Risks

- Descriptor array adds complexity (pointer management)
- Variable K/N across Linears (768 vs 3072) — kernel must handle via descriptor
- If CUDA Graphs alternative is used, graph capture has ~2s startup cost (amortized over training)

---

## Summary of Kernel Efficiency Patches

| # | Patch | Expected speedup | VRAM saved | Risk | Dependencies |
|---|-------|-----------------|------------|------|---------------|
| 5 | Fused bwd with AoS P | 260ms→60ms (200ms saved) | ~7GB | Medium (layout change) | P stored as (K,N,4) |
| 6 | Stream double-buffering | 69ms saved (teacher hidden) | 0 | Medium (event sync) | Prefetch data loader |
| 7 | Batched compute_P_W | 36ms saved (25→1 launch) | 0 | Low | Same GS for all Linears |
| **Total** | | **~305ms→~200ms** | **~7GB** | | |

All 3 patches enhance existing kernels (no replacement of mma.sync.m16n8k16, no tcgen05/WGMMA/TMA migration).
