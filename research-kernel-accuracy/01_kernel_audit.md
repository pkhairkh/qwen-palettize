# 01 — Line-by-Line Audit of `fused_lut_kernel.cu`

**File under audit:** `scripts/fused_lut_kernel.cu` (1631 lines, REV-2)
**Audience:** CUDA kernel engineer, numerical-accuracy reviewer
**Scope:** Every kernel and launcher in the file; flagged defects are tagged with severity (CRITICAL / HIGH / MEDIUM / LOW) and a specific line range.

---

## 1. File header and design notes (lines 1–30)

The header (lines 1–21) declares the kernel API surface and the REV-2 design choices: tile size 32×32 / 2×2-per-thread, no `__launch_bounds__`, smem cooperative loads, fp32 accumulators. The header says **"Target: sm_89 (NVIDIA L4, Ada Lovelace). Also compiles for sm_80, sm_90, sm_86."** — but the user's runtime is **sm_120 (Blackwell, RTX PRO 6000)**. The Python build driver (`fused_lut_linear_cuda.py`, line `default_arch`) actually emits `compute_120,code=sm_120` first, so the SM target mismatch is cosmetic only.

**LOW-severity concern** — the header does not document that the `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` PTX instruction (used in the TC variant, line 481) is supported on sm_80+ but the `cp.async` family used in the prologue requires sm_80+. Blackwell adds new WGMMA instructions that would replace `mma.sync` for higher throughput, but `mma.sync` is still correct on sm_120.

**Take-away:** No correctness issues here; only a documentation gap.

---

## 2. Configuration constants (lines 31–80)

The tile sizes are tuned for occupancy, not numerical accuracy. The forward tile `FWD_BM=64, FWD_BN=64, FWD_BK=32` gives 4×4 outputs/thread = 16 fp32 accumulators (line 139). The backward grad_palette tile is `BWD_GP_BM_K=64, BWD_GP_BN_N=64, BWD_GP_MM_M=32` (lines 70–73) with shared-memory accumulator `s_acc[2][4]` of fp32 (line 917). The accumulator size is bounded by `BN ≤ group_size` so the design assumption is "one group per tile".

**MEDIUM-severity concern** (lines 122–127, 333–340): the code computes `g_first`, `g_last`, `n_groups_in_tile` and switches between two palettes via `group_local = (group == g_first) ? 0 : 1`. This is correct when a tile spans at most 2 groups (which is true since `BN=64 ≤ group_size=256`). However, the code path silently drops the second palette if `n_groups_in_tile > 1` is not handled in **all** consumers (see line 132: only the first 4 palette entries are loaded if `n_groups_in_tile > 1` is false, but the materialize-W step on line 207 indexes `spalette[group_local][idx_val]` with `group_local` ∈ {0,1} — accessing `spalette[1]` when it was never written). On the canonical configuration (BN=64, GS=256) this never triggers, but it is a latent bug if `BN` is ever increased.

**Take-away:** Safe under current configuration; brittle to tile-size changes.

---

## 3. Hard forward kernel — `fused_lut_linear_fwd_kernel` (lines 95–288)

### 3.1 Shared memory layout (lines 104–110)

```cuda
__shared__ __nv_bfloat16 sx[FWD_BM][FWD_BK];        // 4 KB
__shared__ uint8_t       sidx[FWD_BK][FWD_BN];      // 2 KB
__shared__ __nv_bfloat16 spalette[2][4];              // 16 B
__shared__ __nv_bfloat16 sW[FWD_BK][FWD_BN];        // 4 KB
```

The `sW` materialization (lines 195–209) "pre-computes" the bf16 weight tile from palette+indices so the FMA inner loop only does 1 smem load per FMA (instead of 2). This is purely a performance optimization and does NOT change numerics.

### 3.2 Cooperative loads via `cp.async` (lines 148–193)

The `cp.async.cg.shared.global [..], 16` instruction on line 158 is the **cache-global** variant (bypasses L1, writes directly to smem). For `bf16` loads of 16 bytes this is fine — the data is bit-exact. The fallback path (lines 161–167) loads element-by-element with zero-padding for out-of-bounds indices; this is also bit-exact.

**HIGH-severity concern** (line 180): the indices load uses `cp.async.ca.shared.global [..], 8` (cache-all variant, 8 bytes). The indices are `uint8_t`, and 8 bytes = 8 indices. The fallback path (lines 184–188) sets `sidx[r][c + i] = 0` for OOB, but **0 is a valid palette index** (the first palette entry). If `K` or `N` is not a multiple of 8, out-of-bounds weights will be silently reconstructed as `palette[g, 0]` instead of producing zero contribution. This biases the partial sum toward the first palette entry. The effect is small (one or two columns per tile, multiplied by ~0 on average), but it is a systematic bias. **Recommendation:** use `0xFF` as the OOB sentinel and add a guard `if (idx_val < 4)` in the materialize-W step (which the hard backward kernel already does on line 1105: `const int p = (p_raw < 4) ? (int)p_raw : 0;` — although this clamps to 0 rather than skipping, it at least documents the issue).

### 3.3 Materialize-W step (lines 195–209)

```cuda
const uint8_t idx_val = sidx[r][cc];
sW[r][cc] = spalette[group_local][idx_val];
```

The lookup is correct (`idx_val ∈ [0, 3]`, `spalette` has 4 entries). No conversion error — bf16→bf16 is bit-exact.

### 3.4 FMA inner loop (lines 236–270)

```cuda
#pragma unroll 16
for (int kk = 0; kk < FWD_BK; ++kk) {
    __nv_bfloat162 wv2[2];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int n_local_0 = tx * 4 + i * 2;
        const int n_local_1 = tx * 4 + i * 2 + 1;
        wv2[i] = __halves2bfloat162(sW[kk][n_local_0], sW[kk][n_local_1]);
    }
    __nv_bfloat162 xv2[2];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int m_local_0 = ty * 4 + i * 2;
        const int m_local_1 = ty * 4 + i * 2 + 1;
        xv2[i] = __halves2bfloat162(sx[m_local_0][kk], sx[m_local_1][kk]);
    }
    #pragma unroll
    for (int mi_pair = 0; mi_pair < 2; ++mi_pair) {
        __nv_bfloat16 x0 = __low2bfloat16(xv2[mi_pair]);
        __nv_bfloat16 x1 = __high2bfloat16(xv2[mi_pair]);
        float x0_f = __bfloat162float(x0);
        float x1_f = __bfloat162float(x1);
        #pragma unroll
        for (int ni_pair = 0; ni_pair < 2; ++ni_pair) {
            __nv_bfloat16 w0 = __low2bfloat16(wv2[ni_pair]);
            __nv_bfloat16 w1 = __high2bfloat16(wv2[ni_pair]);
            float w0_f = __bfloat162float(w0);
            float w1_f = __bfloat162float(w1);
            acc[mi_pair*2 + 0][ni_pair*2 + 0] += x0_f * w0_f;
            acc[mi_pair*2 + 0][ni_pair*2 + 1] += x0_f * w1_f;
            acc[mi_pair*2 + 1][ni_pair*2 + 0] += x1_f * w0_f;
            acc[mi_pair*2 + 1][ni_pair*2 + 1] += x1_f * w1_f;
        }
    }
}
```

**This is a SCALAR fp32 FMA path, not a true TC MMA path.** The `__nv_bfloat162` packing is purely for register pressure (loads two bf16 values per register pair), but each multiply is done as `__bfloat162float` → `float * float` → `+= float`. **Numerically this is MORE accurate than a true mma.sync.bf16.bf16.f32 instruction** because:
- The `__bfloat162float` conversion is exact (bf16 → fp32 widens mantissa and exponent, no rounding).
- The fp32 multiply produces a correctly-rounded fp32 result.
- The fp32 add accumulates with full fp32 precision (24-bit mantissa).

Whereas `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` performs the multiply as bf16×bf16 with the result rounded to fp32 *before* accumulation. The 16-bit multiply loses 8 bits of mantissa precision relative to the fp32 multiply in this scalar path.

**The TC variant on line 481 is therefore LESS numerically accurate than this scalar path.** This is a critical finding: when `USE_TC_FWD=1` is set, the forward loses ~1 bit of mantissa precision per FMA. Across a K=2560 reduction with 2560 multiply-adds, the relative error grows as `sqrt(K) * eps_bf16 * ||x|| * ||w||` ≈ 50 * 2^-8 ≈ 0.2 in the worst case (assuming uncorrelated errors). In practice errors are smaller due to bias cancellation, but on the order of 10^-3 relative error per output element.

### 3.5 Output write-back (lines 274–287)

```cuda
float v = acc[mi][ni];
if (bias != nullptr) v += __bfloat162float(bias[n_global]);
y[m_global * N + n_global] = __float2bfloat16(v);
```

The `__float2bfloat16` conversion performs round-to-nearest-even (RNE) on the fp32 value to produce a bf16 result. This is correct. The bias is added in fp32 BEFORE the round — also correct (round-after-add, not round-then-add).

**HIGH-severity concern**: There is NO clipping or saturation check. If `acc[mi][ni]` overflows fp32 range (~3.4e38), the conversion to bf16 produces `±inf`. During normal training of an internal Qwen layer this won't happen (activations are bounded by LayerNorm), but during the **first forward after a bad gradient update** (e.g., NaN propagation), this can produce silent `inf` that propagates to the next layer. The hard kernel has no NaN guard.

---

## 4. Hard TC forward kernel — `fused_lut_linear_fwd_tc_kernel` (lines 309–518)

### 4.1 Layout (lines 322–325)

```cuda
__shared__ __nv_bfloat16 sx_buf[2][FWD_TC_BM][FWD_TC_BK];        // 4 KB
__shared__ uint8_t       sidx_buf[2][FWD_TC_BK][FWD_TC_BN];      // 2 KB
__shared__ __nv_bfloat16 sW[FWD_TC_BK][FWD_TC_BN];               // 2 KB
__shared__ __nv_bfloat16 spalette[2][4];                         // 16 B
```

The double-buffered `sx_buf[2]` and `sidx_buf[2]` implement a 2-stage pipeline (line 318). The `FWD_TC_BK = 16` (line 306) is the mma K-dimension.

### 4.2 ISSUE_LOAD macro and pipeline (lines 362–430)

The macro issues `cp.async.ca.shared.global [..], 8` (4 bf16) for x and `cp.async.ca.shared.global [..], 4` (4 uint8) for indices. The wait_group semantics on line 426 (`wait_group 1`) waits for all-but-1 outstanding groups, which is correct for double-buffering.

**HIGH-severity concern** (line 396): the indices load uses 4-byte `cp.async.ca`, which copies 4 raw bytes into smem. If `gn + 4 > N`, the fallback (lines 398–403) writes zeros — same OOB-palette-0 bias issue as the scalar kernel. Additionally, the OOB bytes in the *output* of the load are NOT zeroed if the cp.async load happens to read past the end of the indices tensor (this is undefined behavior in CUDA; on Ada/Blackwell it usually returns zeros, but it is not guaranteed). Recommendation: pad the indices tensor at allocation time to a multiple of 4 in the N dimension.

### 4.3 ldmatrix and mma.sync inner loop (lines 452–487)

```cuda
asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
             : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
             : "r"(smem_addr));
// ...
asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
             : "=r"(b_frag[0]), "=r"(b_frag[1])
             : "r"(smem_addr));
// ...
asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
    : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
      "r"(b_frag[0]), "r"(b_frag[1]));
```

**The `ldmatrix.sync.aligned.m8n8.x4` instruction loads 4 8×8 matrices in row-major layout**. The `.trans` modifier on the B-fragment load (line 475) transposes the 8×8 sub-matrices so they can be used as the B operand of `mma.sync.m16n8k16.row.col` (which expects B in column-major). This is the standard pattern from the CUTLASS examples.

**Numerical analysis:**
- The mma instruction multiplies bf16×bf16 and accumulates in fp32.
- The relative error per multiply is `eps_bf16 = 2^-8 ≈ 3.9e-3` (the bf16 mantissa is 8 bits = 7 explicit + 1 implicit).
- The accumulator is fp32 with full 24-bit mantissa.
- For K=2560 with `FWD_TC_BK=16`, the kernel does 160 mma iterations × 16 K-dim per iter = 2560 multiplies total.
- Error growth: `sqrt(K) * eps_bf16 * ||x|| * ||w|| / ||y||`. If `||x|| ≈ 1` and `||w|| ≈ 1` (post-LayerNorm), error ≈ 50 * 3.9e-3 = 0.2 (worst case, fully correlated). In practice with uncorrelated rounding errors, expect ~10^-3 relative error per output element.

**This is the source of the ~5% cos gap when TC is enabled.** When `USE_TC_FWD=0` (default), the scalar fp32 FMA path is used and the per-element error is ~10^-6 (fp32 has 24-bit mantissa, sqrt(K) * eps_fp32 * ||x|| * ||w|| ≈ 50 * 6e-8 = 3e-6).

### 4.4 Output write-back (lines 492–517)

Same `__float2bfloat16` round-down as the scalar kernel. Same no-NaN-guard issue. The PTX mma C-fragment layout (lines 493–497) follows the standard `(g, t) → C[g/4*8 + 0..7][2*(g%4) + 0..1]` mapping.

**MEDIUM-severity concern** (line 510): the column index is computed as
```cuda
const int n_global = bn * FWD_TC_BN + warp_n * 32 + n_offset + col_base + c_idx;
```
where `col_base = 2 * t` (line 497). With `warp_n=1` and `mma_n=3`, `n_offset=24` and the column range is `32 + 24 + 0..1 = 56..57` (per `t`). This must fit within `FWD_TC_BN=64`, which it does. No issue.

---

## 5. Hard backward grad_x TC kernel — `fused_lut_linear_bwd_grad_x_tc_kernel` (lines 551–720)

This kernel computes `grad_x[i, j] = Σ_o grad_y[i, o] * W[j, o]`, where W is reconstructed from palette+indices.

### 5.1 Layout (lines 560–563)

```cuda
__shared__ __nv_bfloat16 sgy[BWD_GX_TC_BM][BWD_GX_TC_BN];
__shared__ uint8_t       sidx[BWD_GX_TC_BN][BWD_GX_TC_BK];
__shared__ __nv_bfloat16 sW[BWD_GX_TC_BN][BWD_GX_TC_BK];
__shared__ __nv_bfloat16 spalette[2][4];
```

Note the transposed layout: `sidx[BN][BK]` (instead of `sidx[BK][BN]` in the forward). This is because the grad_x matmul treats W as `(N, K)` (transposed). The `sW[BN][BK]` is laid out accordingly.

### 5.2 Indices load (lines 616–635)

```cuda
const int off = tid * 4;
const int r = off / BWD_GX_TC_BK;       // 0..3
const int c = off % BWD_GX_TC_BK;        // 0..60 (mult of 4)
#pragma unroll
for (int i = 0; i < 4; ++i) {
    const int rr = r;                   // N direction (row of transposed tile)
    const int cc = c + i;                // K direction (col of transposed tile)
    const int gk = bk * BWD_GX_TC_BK + cc;
    const int gn = n_chunk + rr;
    if (gk < K && gn < N) {
        sidx[rr][cc] = indices[gk * N + gn];
    } else {
        sidx[rr][cc] = 0;
    }
}
```

**HIGH-severity concern**: The indices tensor has shape `(K, N)` row-major, but this kernel loads `indices[gk * N + gn]` for `(gk, gn) = (bk*BK+cc, n_chunk+rr)`. The access pattern `indices[gk * N + gn]` is correct for row-major layout. BUT the 4-byte coalesced load (line 620 comment claims "256 threads × 4 bytes = 1024 ✓") requires the inner dimension `c = off % BWD_GX_TC_BK` to be contiguous in memory. Since the indices are stored row-major in (K, N), `indices[gk * N + gn]` advances by 1 byte per `gn+1` and by N bytes per `gk+1`. The access pattern here has `cc` (the K index) as the inner dimension, which means consecutive `cc+1` accesses jump by `N` bytes in memory — **NOT coalesced**. The 4-byte loads at line 620 are physically 4 separate byte loads, defeating the 4-byte coalesced access. This is a performance bug, not a numerical bug, but it explains why the TC bwd path is slower than expected.

### 5.3 mma.sync inner loop (lines 657–692)

Same `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` pattern as the forward. Same numerical analysis applies: bf16×bf16 multiply with fp32 accumulation, ~10^-3 relative error per element.

### 5.4 grad_x write-back (lines 696–719)

`__float2bfloat16(c[r * 2 + c_idx])` — same RNE rounding. Same no-NaN-guard issue.

---

## 6. Hard backward grad_x SIMD2 kernel — `fused_lut_linear_bwd_grad_x_kernel` (lines 726–891)

Same structure as the scalar forward kernel (section 3) but with `grad_y` instead of `x` and transposed reduction. The FMA inner loop (lines 840–875) is the same `__bfloat162float → float*float → fp32 accum` pattern — numerically MORE accurate than the TC variant.

**MEDIUM-severity concern** (lines 782–794, 802–815): the cooperative loads use `int4` for 16-byte (8 bf16) loads and `int2` for 8-byte (8 uint8) loads. These are correct, but again the OOB sentinel for indices is 0 (line 812), which is a valid palette index — same bias issue as the forward.

---

## 7. Hard backward grad_palette kernel — `fused_lut_linear_bwd_grad_palette_kernel` (lines 902–1123)

This is the most complex kernel. It computes `dW[j, o] = Σ_i x[i, j] * grad_y[i, o]` and scatters into `grad_palette[g, p] += dW[j, o]` via the index `p = indices[j, o]`.

### 7.1 Shared memory (lines 912–917)

```cuda
__shared__ __nv_bfloat16 sx_chunk[BWD_GP_MM_M][BWD_GP_BM_K];   // 4 KB
__shared__ __nv_bfloat16 sgy_chunk[BWD_GP_MM_M][BWD_GP_BN_N];   // 4 KB
__shared__ uint8_t       sidx[BWD_GP_BM_K][BWD_GP_BN_N];        // 4 KB
__shared__ __nv_bfloat16 sW[BWD_GP_BM_K][BWD_GP_BN_N];          // 8 KB
__shared__ float s_acc[2][4];
```

The `s_acc[2][4]` is a per-block accumulator for 2 groups × 4 palette entries = 8 fp32 values. The block-level accumulator is flushed to global `grad_palette` via `atomicAdd` at the end (lines 1114–1122).

### 7.2 Pre-materialize W (lines 968–987)

```cuda
const uint8_t idx_val = sidx[r][cc];
sW[r][cc] = palette[group * 4 + idx_val];
```

**Note**: this loads from global `palette` (not `spalette`), so the `group_local` indexing is not used here. The compiler will optimize this through the L1 cache. Numerically: bf16 → bf16, bit-exact.

### 7.3 dW accumulation (lines 989–1083)

The `dW[4][4]` per-thread accumulator is fp32 (line 990). The inner loop (lines 1048–1081) does:

```cuda
float x0 = __bfloat162float(__low2bfloat16(xv2[ji_pair]));
float x1 = __bfloat162float(__high2bfloat16(xv2[ji_pair]));
// ...
float g0 = __bfloat162float(__low2bfloat16(gv2[oi_pair]));
float g1 = __bfloat162float(__high2bfloat16(gv2[oi_pair]));
dW[ji_pair*2 + 0][oi_pair*2 + 0] += x0 * g0;
dW[ji_pair*2 + 0][oi_pair*2 + 1] += x0 * g1;
dW[ji_pair*2 + 1][oi_pair*2 + 0] += x1 * g0;
dW[ji_pair*2 + 1][oi_pair*2 + 1] += x1 * g1;
```

Same scalar fp32 FMA pattern as the forward. The accumulator reduces over `M` in chunks of `BWD_GP_MM_M = 32` (line 72). For M=8192 (typical), this is 256 outer iterations × 32 inner = 8192 multiplies per output element. **Numerically correct.**

### 7.4 Scatter_add into shared accumulator (lines 1091–1110)

```cuda
const uint8_t p_raw = sidx[j_local][o_local];
const int p = (p_raw < 4) ? (int)p_raw : 0;
const float val = dW[ji][oi];
atomicAdd(&s_acc[group_local][p], val);
```

The `(p_raw < 4) ? (int)p_raw : 0` clamp on line 1105 is the only defensive check against OOB indices in the entire file. It clamps to 0 instead of skipping, which is acceptable behavior (palette entry 0 is usually the centroid closest to zero, so the bias is small). **MEDIUM-severity concern**: 256 threads doing `atomicAdd` to 8 fp32 slots is ~32-way contention per slot. The `atomicAdd` on shared memory fp32 is ~10 cycles uncontended, so worst case ~320 cycles per slot. Not a numerical bug, but a performance bottleneck.

### 7.5 Global flush (lines 1113–1122)

```cuda
if (linear_tid < 8) {
    const int g_local = linear_tid / 4;
    const int p = linear_tid % 4;
    if (g_local < n_groups_in_tile) {
        const int g_global = (g_local == 0) ? g_first : g_last;
        const float val = s_acc[g_local][p];
        atomicAdd(&grad_palette[g_global * 4 + p], val);
    }
}
```

This is a global fp32 `atomicAdd`. Across many blocks, this is the final reduction. **Numerically correct** — fp32 atomicAdd on Blackwell is IEEE-754 round-to-nearest, but the order of accumulation is non-deterministic (depends on block scheduling). This means runs are NOT bit-reproducible, but the statistical expectation is correct.

---

## 8. Hard backward grad_bias kernel — `fused_lut_linear_bwd_grad_bias_kernel` (lines 1129–1147)

```cuda
float acc = 0.0f;
for (int m = 0; m < M; ++m) {
    const __nv_bfloat16 v = grad_y[m * N + n_global];
    acc += __bfloat162float(v);
}
grad_bias[n_global] = __float2bfloat16(acc);
```

**LOW-severity concern**: This is a sequential fp32 reduction over M. For large M (e.g., 8192), this is O(M) per warp and is correct but slow. Could use warp-shuffle reduction for ~32× speedup. Numerically correct (fp32 accum throughout).

**HIGH-severity concern**: The output is `__float2bfloat16(acc)` — bf16. If `acc` is large (e.g., sum of 8192 values each ~1.0, sum ~8192), the bf16 representation loses precision. With M=8192 and values of magnitude ~1, the accumulator is ~8192, and bf16's resolution at 8192 is `2^(13-8) = 32` ULP — meaning the gradient is quantized to multiples of 32. This is a significant precision loss for grad_bias. **Recommendation**: keep grad_bias in fp32 (cast in the C++ wrapper).

---

## 9. Soft forward kernel — `fused_lut_linear_soft_compute_P_W_kernel` (lines 1301–1352)

### 9.1 Gumbel sampler (lines 1274–1285)

```cuda
__device__ __forceinline__ float gumbel_sample(uint32_t seed, uint32_t idx) {
    uint32_t x = seed ^ (idx * 0x9E3779B9u);
    x ^= x >> 13;
    x = x * 1103515245u + 12345u;
    x ^= x >> 17;
    x = x * 1103515245u + 12345u;
    float u = (float)(x & 0xFFFFFFu) * (1.0f / 16777216.0f);   // [0, 1)
    u = fmaxf(u, 1e-7f);                            // avoid log(0)
    return -logf(-logf(u));                          // Gumbel(0, 1)
}
```

**MEDIUM-severity concern**: The LCG-based PRNG has poor statistical quality (the `1103515245u + 12345u` constants are from the ANSI C `rand()`). The Gumbel sample is `-log(-log(u))`, which has unbounded support. For `u = 1e-7`, `log(u) ≈ -16.1`, `-log(-log(u)) ≈ -log(16.1) ≈ -2.78`. For `u = 1 - 1e-7`, `log(u) ≈ -1e-7`, `-log(1e-7) ≈ 16.1`. So Gumbel samples span roughly [-3, +16]. Combined with logits=±10 and tau=0.1, the noisy logit `(logit + gumbel) / tau` can reach `(10 + 16) / 0.1 = 260` or `(-10 - 3) / 0.1 = -130`. **The fp32 softmax can handle these magnitudes** (max fp32 ≈ 3.4e38), but the `expf(260 - 260) = 1` and `expf(-130 - 260) = exp(-390) ≈ 0` — meaning the softmax becomes exactly one-hot.

This is the **fundamental cause of the Gumbel-Softmax gradient vanishing at low tau**. Once the softmax is one-hot, `P[k]` is 0 or 1, and the gradient formula `grad_W * P[k] * (c[k] - W)` evaluates to 0 for all k (since either `P[k] = 0` for non-argmax entries, or `(c[argmax] - W_soft) = 0` for the argmax entry because `W_soft = c[argmax]` when P is one-hot). The kernel implementation on lines 1331–1337 is mathematically correct; the issue is the mathematical vanishing gradient, not a kernel bug.

### 9.2 Softmax (lines 1330–1337)

```cuda
float m = fmaxf(fmaxf(n0, n1), fmaxf(n2, n3));
float e0 = expf(n0 - m);
// ...
float s = e0 + e1 + e2 + e3;
float p0 = e0 / s, p1 = e1 / s, p2 = e2 / s, p3 = e3 / s;
```

**Numerically stable softmax**: max-subtraction is correct. The `expf` is fp32. The `s = e0+e1+e2+e3` is fp32 sum. **No precision issue here.**

**LOW-severity concern**: At very low tau (e.g., 0.01), `n_i = logit / tau` can be ±1000. `expf(-2000) = 0` and `expf(0) = 1`, so the softmax is exactly one-hot. This is mathematically correct behavior but means **gradient flow is dead** — the kernel will write `P = [0, 0, 1, 0]` exactly, and the backward kernel will produce zero gradients.

### 9.3 P save (lines 1339–1343)

```cuda
P[0 * plane_size + idx] = __float2half(p0);
```

**HIGH-severity concern**: P is stored as **fp16** (5-bit exponent, 10-bit mantissa), not bf16 (8-bit exponent, 7-bit mantissa). This is unusual: the kernel header comments say "fp16 softmax" but the typical Gumbel-Softmax implementation uses fp32 P for backward. The fp16 storage loses precision in two ways:

1. **Denormal underflow**: fp16's smallest normal value is `2^-14 ≈ 6.1e-5`. If `p_k < 6.1e-5`, it gets stored as a denormal (which has reduced mantissa precision) or zero. At tau=0.5 with 4 logits, the smallest P can be as small as `softmax(-2/0.5) = exp(-4) / (1 + 3*exp(-4)) ≈ 0.018` — well above the denormal threshold. At tau=0.1 with logits=±10, the smallest P can be `softmax(-20/0.1) = exp(-200) ≈ 0` — already zero in fp16.

2. **Mantissa precision**: fp16 has 10-bit mantissa (≈3 decimal digits). For P values like `0.99999`, fp16 stores it as `1.0` exactly. This loses the gradient signal entirely.

**Recommendation**: store P as bf16 or fp32 if memory allows. bf16 would preserve the exponent range (no denormal issues) at the cost of 2 fewer mantissa bits — still better than fp16 for this use case.

### 9.4 W computation (lines 1345–1351)

```cuda
float c0 = __bfloat162float(palette[g * 4 + 0]);
// ...
float W_val = p0 * c0 + p1 * c1 + p2 * c2 + p3 * c3;
W_out[idx] = __float2bfloat16(W_val);
```

**Correct**: fp32 accumulation, RNE round to bf16 at output. No issue.

---

## 10. Soft backward grad_logits kernel — `fused_lut_linear_soft_bwd_grad_logits_kernel` (lines 1364–1398)

```cuda
float W_val = c0 * p0 + c1 * p1 + c2 * p2 + c3 * p3;
grad_logits[0 * plane_size + idx] = __float2half(dW * p0 * (c0 - W_val));
```

**Mathematical verification:**
- `W = Σ_k P[k] * c[k]`
- `dL/dlogit_k = dL/dW * dW/dlogit_k`
- `dW/dlogit_k = dW/dP_j * dP_j/dlogit_k = Σ_j c[j] * dP[j]/dlogit_k`
- For softmax: `dP[j]/dlogit_k = P[j] * (δ_jk - P[k])`
- So `dW/dlogit_k = Σ_j c[j] * P[j] * (δ_jk - P[k]) = c[k] * P[k] - P[k] * Σ_j c[j] * P[j] = P[k] * (c[k] - W)`
- Therefore `grad_logit_k = dW * P[k] * (c[k] - W)` ✓

The kernel implementation matches the math. **However**, the `W_val` is RECOMPUTED from `P` and `palette` instead of using the saved `W_out` from forward. This recomputation introduces an extra rounding step: `P (fp16) → fp32 → multiply by palette (bf16→fp32) → fp32 sum → fp32`. The saved `W_out` is `fp32 → bf16` (round-to-bf16). The recomputed `W_val` is fp32 (no round-down to bf16). **These differ by up to 1 bf16 ULP** — but since the backward kernel uses the fp32 recomputed value, it's actually MORE accurate than the saved bf16 W.

**HIGH-severity concern** (line 1394): `__float2half(dW * p0 * (c0 - W_val))` — the result is stored as fp16. If `dW * p0 * (c0 - W_val)` has magnitude < 6e-8 (denormal range of fp16), it underflows to zero. With `dW ~ 1e-3`, `p0 ~ 0.01`, `c0 - W_val ~ 1e-2`, the product is ~1e-7 — right at the denormal threshold. **At low tau where P is one-hot**, `c0 - W_val` is ~0 for the argmax entry, making the product ~0, and `P_k` is ~0 for non-argmax entries, also making the product ~0. So the entire gradient is essentially zero — confirming the vanishing-gradient hypothesis.

**Recommendation**: store grad_logits as fp32, not fp16.

---

## 11. Soft backward grad_palette kernel — `fused_lut_linear_soft_bwd_grad_palette_kernel` (lines 1409–1434)

```cuda
atomicAdd(&grad_palette[g * 4 + 0], dW * p0);
atomicAdd(&grad_palette[g * 4 + 1], dW * p1);
atomicAdd(&grad_palette[g * 4 + 2], dW * p2);
atomicAdd(&grad_palette[g * 4 + 3], dW * p3);
```

**Mathematically correct**: `grad_palette[g, k] = Σ_{j, o ∈ group g} dW[j, o] * P[j, o, k]`.

**MEDIUM-severity concern**: Each thread does 4 atomicAdds to fp32 global memory. For a (K=2560, N=8192) tensor with G=N/GS=32 groups, each group has K×256 = 655360 atomic adds. With 4 atomic adds per thread, that's 1.6M atomic adds to 4 fp32 slots per group. This is heavy contention on global atomic memory. The hard backward kernel (section 7.4) uses a smem accumulator first, which is faster — the soft kernel should do the same.

**Numerically correct** — fp32 atomicAdd is IEEE-754 round-to-nearest, just non-deterministic order.

---

## 12. Fused soft backward kernel — `fused_lut_linear_soft_bwd_fused_kernel` (lines 1498–1610)

This is the IX.b variant that computes `grad_W` on-the-fly instead of materializing it. The M-reduction (lines 1527–1583) accumulates `grad_W += Σ_i sx_chunk[i][tk] * sgy_chunk[i][tn]` in fp32. This is correct.

**HIGH-severity concern** (line 1577): `#pragma unroll` over `SOFT_BWD_BM_CHUNK = 128` iterations. The compiler may refuse to unroll this (too many instructions), falling back to a sequential loop. The unroll attempt inflates code size and may cause register spilling. Empirically, the developers note in `fused_lut_linear_cuda.py` that this kernel was 3.8× SLOWER than the PyTorch vectorized path (due to strided reads of P from global memory), so it's not actually used in training. The Python backward (section 13) is the one that runs.

---

## 13. Python backward (in `fused_lut_linear_cuda.py`, not the .cu file)

The Python backward in `CUDAFusedLUTLinearSoft.backward` (around lines 525-580 of `fused_lut_linear_cuda.py`) is what actually runs during training. It does:

```python
grad_W = torch.matmul(x.T, grad_y)  # (K, N) bf16
# ...
contributions = (grad_W.unsqueeze(-1) * P_kno).view(K, G, GS, 4)
grad_palette = contributions.sum(dim=(0, 2)).to(torch.bfloat16)
# ...
grad_W_f = grad_W.float()
P_kno_f = P.permute(1, 2, 0).float()
g_idx = torch.arange(N, device=x.device) // GS
pal_pos = palette[g_idx.long()].unsqueeze(0).expand(K, N, 4).float()
W_val = (P_kno_f * pal_pos).sum(dim=-1)
grad_logits = (
    grad_W_f.unsqueeze(-1) * P_kno_f * (pal_pos - W_val.unsqueeze(-1))
).to(torch.float16).permute(2, 0, 1).contiguous()
```

**CRITICAL findings:**

1. **`grad_W` is bf16** (line `grad_W = torch.matmul(x.T, grad_y)` — both x and grad_y are bf16, result is bf16 under autocast). This loses 16 bits of mantissa relative to fp32. For small gradients (early in training when loss is small), the bf16 representation has ~3 decimal digits of precision, which is insufficient for the gradient to be useful.

2. **`grad_palette` accumulated in bf16** (line `contributions = (grad_W.unsqueeze(-1) * P_kno).view(K, G, GS, 4)` — grad_W is bf16, P_kno is fp16, the product is bf16 under autocast or fp32 without. The sum `dim=(0, 2)` is a fp32 or bf16 reduction. The final `.to(torch.bfloat16)` truncates to bf16. This is a 2-step precision loss: bf16 multiply then bf16 sum then bf16 cast.

3. **`grad_logits` is fp16** (line `.to(torch.float16)`) — same fp16 underflow issue as the CUDA kernel.

4. **The skip_grad_logits env var** (`SKIP_ZERO_GRAD_LOGITS=1`) is documented in the code as "Empirically verified at tau=0.1 with logits=±10: all 25 index_logits grads are 0.0. The L4 'training' of indices was a no-op." This confirms the vanishing-gradient hypothesis empirically.

---

## 14. Host-side launchers (lines 1156–1247)

The launchers cast `c10::BFloat16*` to `__nv_bfloat16*` — these types are layout-compatible (both IEEE 754 bf16). The `USE_TC_FWD` and `USE_TC_BWD_GX` environment variables (lines 1164, 1197) select between the scalar and TC variants. Default is scalar (more accurate). **No issue.**

---

## 15. Summary of audit findings

| Section | Lines | Severity | Finding |
|---------|-------|----------|---------|
| 3.2 | 184–188 | HIGH | OOB indices use 0 (valid palette entry), causing systematic bias toward palette[0]. |
| 3.4 | 236–270 | — | Scalar fp32 FMA path is MORE accurate than TC. Default. |
| 3.5 | 285 | HIGH | No NaN/inf guard on output write. |
| 4.3 | 480–485 | HIGH | TC mma.sync loses 8 bits of mantissa per FMA vs scalar fp32. |
| 5.2 | 620 | HIGH (perf) | TC bwd grad_x indices load is non-coalesced (strided). |
| 7.4 | 1105 | MEDIUM | Hard bwd grad_palette clamps OOB indices to 0 (acceptable). |
| 8 | 1146 | HIGH | grad_bias output is bf16 — large sums lose precision. |
| 9.1 | 1274–1285 | MEDIUM | LCG PRNG is statistically weak; Gumbel samples are bounded. |
| 9.3 | 1339–1343 | HIGH | P stored as fp16 — denormal underflow + mantissa precision loss. |
| 10 | 1394 | HIGH | grad_logits stored as fp16 — underflow at low tau. |
| 11 | 1430–1433 | MEDIUM | Soft bwd grad_palette does 4 global atomicAdds per thread (slow). |
| 13 | (Python) | CRITICAL | Python backward uses bf16 grad_W, bf16 grad_palette accumulation, fp16 grad_logits. |

**The single most damaging issue is section 13**: the Python backward path uses bf16 throughout for grad_W and grad_palette, and fp16 for grad_logits. Combined with the Gumbel-Softmax vanishing gradient at low tau (section 9.1), this means the index_logits receive essentially zero useful gradient signal, and the palette gradients are quantized to bf16 precision (~3 decimal digits). The hard forward kernel (sections 3 and 4) is numerically fine — the cos gap to teacher is dominated by the SOFT path's precision losses, not the hard forward.
