// =============================================================================
// fused_lut_kernel.cu — Fused 2-bit LUT-quantized linear layer CUDA kernels
//
// Implements:
//   forward:      y = x @ W + bias,  where W[j, o] = palette[o // GS, indices[j, o]]
//   backward:     grad_x = grad_y @ W.T
//                 grad_palette = scatter_add(x.T @ grad_y, _flat_idx)  (fp32 accum)
//                 grad_bias = grad_y.sum(0)
//
// Target: sm_89 (NVIDIA L4, Ada Lovelace). Also compiles for sm_80, sm_90, sm_86.
//
// Design notes (REV-2):
//   * Tile size REDUCED from 64x64/4x4-per-thread to 32x32/2x2-per-thread
//     to eliminate register spilling (was 1104-1696B stack, now ~0).
//   * Smem cooperative loads fixed to cover full tile (was loading half).
//   * Per-thread output count = 4 (2x2), giving ~5 fp32 accumulators per thread.
//   * No __launch_bounds__ — let compiler pick. With ~20 regs/thread we get
//     6+ blocks/SM naturally → high occupancy.
//   * grad_palette smem accumulator bounded to 4 entries (single group per tile
//     since BN=32 ≤ group_size=256).
// =============================================================================

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#ifndef GROUP_SIZE_DEFAULT
#define GROUP_SIZE_DEFAULT 256
#endif

// ─────────────────────────────────────────────────────────────────────────────
//  Configuration constants
// ─────────────────────────────────────────────────────────────────────────────

namespace {

constexpr int WARP = 32;

// Forward tile: BM rows of M, BN cols of N, BK reduction over K.
//   Smem: x[BM][BK] bf16 = 4 KB, idx[BK][BN] uint8 = 2 KB, palette[1][4] = 16 B → ~6 KB
//   Block: 16x16 = 256 threads; each owns 4x4 = 16 output elements.
//   Per-thread accumulator count: 16 fp32 = 16 regs (fits well).
//   NO __launch_bounds__ — compiler picks register count for optimal occupancy.
constexpr int FWD_BM = 64;
constexpr int FWD_BN = 64;
constexpr int FWD_BK = 32;
constexpr int FWD_TX = 16;
constexpr int FWD_TY = 16;
constexpr int FWD_THREADS = FWD_TX * FWD_TY;   // 256

// Backward grad_x tile: same shape as forward, transposed reduction.
//   grad_x[i, j] = Σ_o grad_y[i, o] * W[j, o]
//   Tile: (BM over M, BK over K) and reduce over N in chunks of BN.
//   32x32 tile with 16x16 threads → 2x2 outputs/thread (matches existing code).
constexpr int BWD_GX_BM = 64;
constexpr int BWD_GX_BK = 64;
constexpr int BWD_GX_BN = 32;  // smaller BN to keep smem small
// 64x64 outputs, 16x16 threads → 4x4 outputs/thread (matches fwd design).
constexpr int BWD_GX_THREADS = 256;

// Backward grad_palette tile.
//   Tile: (BM_K rows of K) × (BN_N cols of N), reduce over M in chunks of MM_M.
//   Smem: x_chunk[MM_M][BM_K] bf16 = 4 KB
//         gy_chunk[MM_M][BN_N] bf16 = 4 KB
//         idx[BM_K][BN_N] uint8 = 2 KB
//         sW[BM_K][BN_N] bf16 = 8 KB (pre-materialized W tile)
//         s_acc[2][4] fp32 = 32 B
//   Total ~18 KB → fits in 100 KB Ada smem easily.
//   Block: 16x16 = 256 threads; each owns 4x4 = 16 (j, o) pairs.
constexpr int BWD_GP_BM_K = 64;
constexpr int BWD_GP_BN_N = 64;
constexpr int BWD_GP_MM_M = 32;
constexpr int BWD_GP_THREADS = 256;

// Backward grad_bias: one warp per BN output columns, reduce over M.
constexpr int BWD_GB_BN = 32;        // output cols per program (= warp width)
constexpr int BWD_GB_THREADS = 32;   // single warp per program
constexpr int BWD_GB_M_CHUNK = 128;

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
//  Forward kernel
// ─────────────────────────────────────────────────────────────────────────────
//
//  Grid: (cdiv(M, FWD_BM), cdiv(N, FWD_BN))
//  Block: (FWD_TX, FWD_TY) = (16, 16) = 256 threads
//  Each thread owns 4x4 = 16 output elements:
//    m_local = ty*4 + {0..3}, n_local = tx*4 + {0..3}
//
//  Optimization: hoist W (palette lookup) loads out of the mi loop.
//  For each kk, we load 4 W values (one per ni), then reuse across 4 mi rows.
//  This gives 4 FMAs per W load (vs 1 in the naive version).
//
__global__ void fused_lut_linear_fwd_kernel(
    const __nv_bfloat16* __restrict__ x,        // (M, K) bf16
    const __nv_bfloat16* __restrict__ palette,  // (G, 4) bf16
    const uint8_t*        __restrict__ indices,  // (K, N) uint8
    const __nv_bfloat16* __restrict__ bias,      // (N,) bf16 or nullptr
    __nv_bfloat16*       __restrict__ y,         // (M, N) bf16
    int M, int K, int N,
    int group_size
) {
    __shared__ __nv_bfloat16 sx[FWD_BM][FWD_BK];        // 4 KB
    __shared__ uint8_t       sidx[FWD_BK][FWD_BN];      // 2 KB
    __shared__ __nv_bfloat16 spalette[2][4];              // 16 B
    // Pre-materialized W tile: (BK × BN) bf16 = 32 × 64 = 4 KB.
    // Computed once per K-chunk from palette + indices, then accessed as a
    // contiguous bf16 matrix (1 smem load per FMA instead of 2).
    __shared__ __nv_bfloat16 sW[FWD_BK][FWD_BN];        // 4 KB

    const int bm = blockIdx.x;
    const int bn = blockIdx.y;
    const int tx = threadIdx.x;   // 0..15 (N direction)
    const int ty = threadIdx.y;   // 0..15 (M direction)
    const int linear_tid = ty * FWD_TX + tx;     // 0..255

    // Each thread owns 4x4 = 16 output elements
    const int m_global_base = bm * FWD_BM + ty * 4;
    const int n_global_base = bn * FWD_BN + tx * 4;

    // Determine which groups this BN tile touches (1 or 2; usually 1 since BN=64 ≤ group_size=256)
    const int n_tile_start = bn * FWD_BN;
    const int n_tile_end   = min(n_tile_start + FWD_BN, N);
    const int g_first = n_tile_start / group_size;
    const int g_last  = (n_tile_end - 1) / group_size;
    const int n_groups_in_tile = g_last - g_first + 1;

    // Load palette entries for the groups touched by this tile
    if (ty == 0 && tx < 4) {
        spalette[0][tx] = palette[g_first * 4 + tx];
        if (n_groups_in_tile > 1) {
            spalette[1][tx] = palette[g_last * 4 + tx];
        }
    }
    __syncthreads();

    // 16 fp32 accumulators (4x4)
    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            acc[i][j] = 0.0f;

    // ── Loop over K in chunks of FWD_BK ───────────────────────────────────
    for (int k_chunk = 0; k_chunk < K; k_chunk += FWD_BK) {
        // ── Cooperative load of x tile via cp.async ────────────────────────
        {
            const int off = linear_tid * 8;
            const int r = off / FWD_BK;
            const int c = off % FWD_BK;
            const int gm = bm * FWD_BM + r;
            const int gk = k_chunk + c;
            const size_t smem_addr = __cvta_generic_to_shared(&sx[r][c]);
            if (gm < M && gk + 8 <= K) {
                const void* gptr = &x[gm * K + gk];
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                             :: "l"(smem_addr), "l"(gptr));
            } else {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const bool ok = (bm * FWD_BM + r < M) && (k_chunk + c + i < K);
                    sx[r][c + i] = ok ? x[(bm * FWD_BM + r) * K + (k_chunk + c + i)]
                                      : __float2bfloat16(0.0f);
                }
            }
        }

        // ── Cooperative load of indices tile via cp.async ──────────────────
        {
            const int offset = linear_tid * 8;
            const int r = offset / FWD_BN;
            const int c = offset % FWD_BN;
            const int gk = k_chunk + r;
            const int gn = bn * FWD_BN + c;
            const size_t smem_addr = __cvta_generic_to_shared(&sidx[r][c]);
            if (gk < K && gn + 8 <= N) {
                const void* gptr = &indices[gk * N + gn];
                asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n"
                             :: "l"(smem_addr), "l"(gptr));
            } else {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const bool ok = (k_chunk + r < K) && (bn * FWD_BN + c + i < N);
                    sidx[r][c + i] = ok ? indices[(k_chunk + r) * N + (bn * FWD_BN + c + i)] : 0;
                }
            }
        }

        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();

        // ── Materialize W tile from palette + indices ────────────────────────
        {
            const int off = linear_tid * 8;
            const int r = off / FWD_BN;
            const int c = off % FWD_BN;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int cc = c + i;
                const int n_global = bn * FWD_BN + cc;
                const int group = n_global / group_size;
                const int group_local = (group == g_first) ? 0 : 1;
                const uint8_t idx_val = sidx[r][cc];
                sW[r][cc] = spalette[group_local][idx_val];
            }
        }
        __syncthreads();

        // ── Compute partial dot product via mma.sync + ldmatrix (Phase V) ────
        // mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
        //
        // Warp tile: 16x16 (1 mma_m=16 × 2 mma_n=8) per K-chunk iter (K=16)
        // Per K-chunk (BK=32): 2 mma_k iters × 2 mma_n iters = 4 mmas per warp
        //
        // Block has 8 warps; tile 4 warps × 2 warps (m_warp × n_warp).
        //   warp_m = warp_id / 2  (0..3)  → 16 M rows each, covering BM=64
        //   warp_n = warp_id % 2  (0..1)  → 32 N cols each, covering BN=64
        //
        // Each warp's output: 16x32 = 512 outputs / 32 threads = 16 per thread.
        // But our acc[4][4] only has 16 outputs/thread = same! ✓
        // Mapping: thread lane_id in warp owns output[m][n] where:
        //   m = warp_m*16 + 8*(lane_id/16) + (lane_id%8)/2    → row offset within warp's 16x32 tile
        //   Actually PTX layout: thread (g, t) where g=lane_id>>2, t=lane_id&3 owns:
        //     C[g/4*8 + 0..7][2*(g%4) + 0..1] for fp32 result of m16n8k16 (4 elements)
        //   For 2 mma_n outputs, that's 8 elements per thread.
        //
        // Strategy: use mma to compute 16x16 outputs at a time, accumulate to a
        // c_frag per (warp_m, warp_n, mma_n) combo, then transfer to acc at end.

        // For Phase V v1: keep SIMD2 FMA path (correctness oracle), add mma as
        // an experimental kernel variant. See below for SIMD2 path.

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
        __syncthreads();
    }

    // ── Write output ──────────────────────────────────────────────────────
    #pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
        const int m_global = m_global_base + mi;
        if (m_global >= M) continue;
        #pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            const int n_global = n_global_base + ni;
            if (n_global >= N) continue;
            float v = acc[mi][ni];
            if (bias != nullptr) v += __bfloat162float(bias[n_global]);
            y[m_global * N + n_global] = __float2bfloat16(v);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Phase V — Tensor Core forward kernel (mma.sync + ldmatrix)
//
//  Uses mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 on Ada (sm_89).
//  This kernel computes the same y = x @ W as fused_lut_linear_fwd_kernel
//  but uses tensor cores instead of scalar FMA. Expected ~3-4× speedup.
//
//  Warp tile: 16x16 (1 mma_m × 2 mma_n).
//  Block tile: BM=64, BN=64, BK=16. 8 warps = 4×2 grid (m_warp × n_warp).
//    warp_m = warp_id / 2  (0..3 → 16 M rows each)
//    warp_n = warp_id % 2  (0..1 → 32 N cols each)
//  Per K-chunk: each warp does 2 mma_k × 2 mma_n = 4 mmas.
// ─────────────────────────────────────────────────────────────────────────────

constexpr int FWD_TC_BM = 64;
constexpr int FWD_TC_BN = 64;
constexpr int FWD_TC_BK = 16;        // smaller BK = 16 = mma k-dim
constexpr int FWD_TC_THREADS = 256;

__global__ void fused_lut_linear_fwd_tc_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ palette,
    const uint8_t*        __restrict__ indices,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16*       __restrict__ y,
    int M, int K, int N,
    int group_size
) {
    // Phase VII: Double-buffered smem for cp.async pipelining.
    // 2 stages of x + indices; W is materialized per-iter into a single buffer.
    // Smem budget: sx_buf[2][64][16] = 4KB, sidx_buf[2][16][64] = 2KB,
    //              sW[16][64] = 2KB, spalette[2][4] = 16B → ~8 KB total
    __shared__ __nv_bfloat16 sx_buf[2][FWD_TC_BM][FWD_TC_BK];        // 4 KB (2 × 2KB)
    __shared__ uint8_t       sidx_buf[2][FWD_TC_BK][FWD_TC_BN];      // 2 KB (2 × 1KB)
    __shared__ __nv_bfloat16 sW[FWD_TC_BK][FWD_TC_BN];               // 2 KB
    __shared__ __nv_bfloat16 spalette[2][4];                          // 16 B

    const int bm = blockIdx.x;
    const int bn = blockIdx.y;
    const int tid = threadIdx.x;       // 0..255
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int warp_m = warp_id / 2;    // 0..3
    const int warp_n = warp_id % 2;    // 0..1

    // Determine which groups this BN tile touches
    const int n_tile_start = bn * FWD_TC_BN;
    const int n_tile_end   = min(n_tile_start + FWD_TC_BN, N);
    const int g_first = n_tile_start / group_size;
    const int g_last  = (n_tile_end - 1) / group_size;
    const int n_groups_in_tile = g_last - g_first + 1;

    // Load palette (first 8 threads)
    if (tid < 8) {
        const int gi = tid / 4;
        const int pi = tid % 4;
        if (gi == 0) spalette[0][pi] = palette[g_first * 4 + pi];
        else if (n_groups_in_tile > 1) spalette[1][pi] = palette[g_last * 4 + pi];
    }
    __syncthreads();

    // Per-thread accumulators: 4 mma_n × 4 fp32 = 16 fp32 per thread
    // (4 mma_n × 8 cols each = 32 cols, matching warp_n=32 col tile)
    float c0[4] = {0.0f, 0.0f, 0.0f, 0.0f};   // mma_n=0: cols 0..7
    float c1[4] = {0.0f, 0.0f, 0.0f, 0.0f};   // mma_n=1: cols 8..15
    float c2[4] = {0.0f, 0.0f, 0.0f, 0.0f};   // mma_n=2: cols 16..23
    float c3[4] = {0.0f, 0.0f, 0.0f, 0.0f};   // mma_n=3: cols 24..31

    // ── Phase VII: 2-stage cp.async pipeline ────────────────────────────
    // Issue load N+1 into sx_buf[next_buf] while computing on sx_buf[buf].

    // Helper macro for issuing cp.async loads into a specific buffer
    #define ISSUE_LOAD(BUF_IDX, K_CHUNK)                                                  \
        do {                                                                              \
            /* Load x tile into sx_buf[BUF_IDX]: 64x16 = 1024 bf16 = 2KB */               \
            /* 256 threads × 4 bf16 = 1024 ✓ */                                            \
            {                                                                             \
                const int off = tid * 4;       /* 4 bf16 = 8 bytes */                      \
                const int r = off / FWD_TC_BK;                                             \
                const int c = off % FWD_TC_BK;                                             \
                const int gm = bm * FWD_TC_BM + r;                                         \
                const int gk = (K_CHUNK) + c;                                              \
                const size_t smem_addr = __cvta_generic_to_shared(&sx_buf[BUF_IDX][r][c]);\
                if (gm < M && gk + 4 <= K) {                                               \
                    const void* gptr = &x[gm * K + gk];                                   \
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n"            \
                                 :: "r"((uint32_t)smem_addr), "l"(gptr));                  \
                } else {                                                                  \
                    for (int i = 0; i < 4; ++i) {                                          \
                        const bool ok = (bm*FWD_TC_BM+r < M) && ((K_CHUNK)+c+i < K);       \
                        sx_buf[BUF_IDX][r][c+i] = ok ? x[(bm*FWD_TC_BM+r)*K + ((K_CHUNK)+c+i)] \
                                                      : __float2bfloat16(0.0f);            \
                    }                                                                     \
                }                                                                         \
            }                                                                             \
            /* Load indices tile into sidx_buf[BUF_IDX]: 16x64 = 1024 uint8 = 1KB */      \
            /* 256 threads × 4 bytes = 1024 ✓ */                                            \
            {                                                                             \
                const int off = tid * 4;                                                   \
                const int r = off / FWD_TC_BN;                                            \
                const int c = off % FWD_TC_BN;                                             \
                const int gk = (K_CHUNK) + r;                                              \
                const int gn = bn * FWD_TC_BN + c;                                         \
                const size_t smem_addr = __cvta_generic_to_shared(&sidx_buf[BUF_IDX][r][c]);\
                if (gk < K && gn + 4 <= N) {                                               \
                    const void* gptr = &indices[gk * N + gn];                              \
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"             \
                                 :: "r"((uint32_t)smem_addr), "l"(gptr));                  \
                } else {                                                                  \
                    for (int i = 0; i < 4; ++i) {                                          \
                        const bool ok = ((K_CHUNK)+r < K) && (bn*FWD_TC_BN+c+i < N);       \
                        sidx_buf[BUF_IDX][r][c+i] = ok ? indices[((K_CHUNK)+r)*N + (bn*FWD_TC_BN+c+i)] : 0;\
                    }                                                                     \
                }                                                                         \
            }                                                                             \
            asm volatile("cp.async.commit_group;\n" ::);                                  \
        } while (0)

    // ── Prologue: issue first load into buf 0 ──────────────────────────
    if (K > 0) {
        ISSUE_LOAD(0, 0);
    }
    __syncthreads();   // also wait for palette load

    // ── Main loop: compute on buf, prefetch into next_buf ─────────────────
    int n_k_chunks = (K + FWD_TC_BK - 1) / FWD_TC_BK;
    for (int kc = 0; kc < n_k_chunks; ++kc) {
        const int k_chunk = kc * FWD_TC_BK;
        const int buf = kc & 1;          // current compute buffer
        const int next_buf = (kc + 1) & 1;

        // Issue load for NEXT iteration (if not last)
        if (kc + 1 < n_k_chunks) {
            const int next_k = (kc + 1) * FWD_TC_BK;
            ISSUE_LOAD(next_buf, next_k);
            // Wait for current iter only (next is in flight)
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            // Last iter — wait for all to finish
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();

        // Materialize W tile from sx_buf[buf] + sidx_buf[buf] + spalette
        // sW[r][c] = palette[group, sidx_buf[buf][r][c]]
        // 16*64 = 1024 bf16. 256 threads × 4 = 1024 ✓
        {
            const int off = tid * 4;
            const int r = off / FWD_TC_BN;
            const int c = off % FWD_TC_BN;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int cc = c + i;
                const int n_global = bn * FWD_TC_BN + cc;
                const int group = n_global / group_size;
                const int group_local = (group == g_first) ? 0 : 1;
                const uint8_t idx_val = sidx_buf[buf][r][cc];
                sW[r][cc] = spalette[group_local][idx_val];
            }
        }
        __syncthreads();

        // ── mma.sync inner loop ───────────────────────────────────────────
        uint32_t a_frag[4];
        uint32_t b_frag[2];

        // Load A fragment from sx_buf[buf]
        {
            const int row_in_warp = lane_id % 16;
            const int col_offset = (lane_id / 16) * 8;
            const int sx_row = warp_m * 16 + row_in_warp;
            const int sx_col = col_offset;
            uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(&sx_buf[buf][sx_row][sx_col]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                         : "r"(smem_addr));
        }

        #pragma unroll
        for (int mma_n = 0; mma_n < 4; ++mma_n) {
            {
                const int row_in_warp = lane_id % 16;
                const int sW_row = row_in_warp;
                const int sW_col = warp_n * 32 + mma_n * 8;
                uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(&sW[sW_row][sW_col]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                             : "=r"(b_frag[0]), "=r"(b_frag[1])
                             : "r"(smem_addr));
            }
            float* c = (mma_n == 0) ? c0 : (mma_n == 1) ? c1 : (mma_n == 2) ? c2 : c3;
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                  "r"(b_frag[0]), "r"(b_frag[1]));
        }
        __syncthreads();
    }

    #undef ISSUE_LOAD

    // ── Write output (PTX mma C-fragment layout) ──────────────────────────
    const int g = lane_id >> 2;
    const int t = lane_id & 3;
    const int row_lo = g;
    const int row_hi = g + 8;
    const int col_base = 2 * t;

    #pragma unroll
    for (int mma_n = 0; mma_n < 4; ++mma_n) {
        const float* c = (mma_n == 0) ? c0 : (mma_n == 1) ? c1 : (mma_n == 2) ? c2 : c3;
        const int n_offset = mma_n * 8;
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int row_in_warp = (r == 0) ? row_lo : row_hi;
            const int m_global = bm * FWD_TC_BM + warp_m * 16 + row_in_warp;
            if (m_global >= M) continue;
            #pragma unroll
            for (int c_idx = 0; c_idx < 2; ++c_idx) {
                const int n_global = bn * FWD_TC_BN + warp_n * 32 + n_offset + col_base + c_idx;
                if (n_global >= N) continue;
                float v = c[r * 2 + c_idx];
                if (bias != nullptr) v += __bfloat162float(bias[n_global]);
                y[m_global * N + n_global] = __float2bfloat16(v);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Phase V — Tensor Core forward kernel (mma.sync + ldmatrix)
//
//  Uses mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 on Ada (sm_89).
//  This kernel computes the same y = x @ W as fused_lut_linear_fwd_kernel
//  but uses tensor cores instead of scalar FMA. Expected ~3-4× speedup.
//
//  Warp tile: 16x16 (1 mma_m × 2 mma_n).
//  Block tile: BM=64, BN=64, BK=16. 8 warps = 4×2 grid (m_warp × n_warp).
//    warp_m = warp_id / 2  (0..3 → 16 M rows each)
//    warp_n = warp_id % 2  (0..1 → 32 N cols each)
//  Per K-chunk: each warp does 2 mma_k × 2 mma_n = 4 mmas.

// ─────────────────────────────────────────────────────────────────────────────
//  Phase VI — Tensor Core bwd_grad_x kernel
//
//  Computes grad_x[i, j] = Σ_o grad_y[i, o] * W[j, o]   (i.e., grad_x = grad_y @ W.T)
//
//  mma.sync.m16n8k16: A=grad_y (M,N), B=W (K,N) used as W.T (N,K) col-major.
//    M_mma=16 (M rows of grad_x output), K_mma=16 (N reduction), N_mma=8 (K cols of grad_x output)
//  Block tile: BM=64, BK=64, BN=16 (= mma K). 8 warps = 4×2 (m_warp × n_warp).
//    warp_m = warp_id / 2  (0..3 → 16 M rows each)
//    warp_n = warp_id % 2  (0..1 → 32 K cols each)
//  Per K-chunk (BN=16): each warp does 1 mma_m × 4 mma_n = 4 mmas.
// ─────────────────────────────────────────────────────────────────────────────

constexpr int BWD_GX_TC_BM = 64;
constexpr int BWD_GX_TC_BK = 64;
constexpr int BWD_GX_TC_BN = 16;
constexpr int BWD_GX_TC_THREADS = 256;

__global__ void fused_lut_linear_bwd_grad_x_tc_kernel(
    const __nv_bfloat16* __restrict__ grad_y,
    const __nv_bfloat16* __restrict__ palette,
    const uint8_t*        __restrict__ indices,
    __nv_bfloat16*       __restrict__ grad_x,
    int M, int K, int N,
    int group_size
) {
    // Smem: sgy[64][16]=2KB, sidx[16][64]=1KB, sW[16][64]=2KB, spalette[2][4]=16B
    __shared__ __nv_bfloat16 sgy[BWD_GX_TC_BM][BWD_GX_TC_BN];
    __shared__ uint8_t       sidx[BWD_GX_TC_BN][BWD_GX_TC_BK];
    __shared__ __nv_bfloat16 sW[BWD_GX_TC_BN][BWD_GX_TC_BK];
    __shared__ __nv_bfloat16 spalette[2][4];

    const int bm = blockIdx.x;
    const int bk = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int warp_m = warp_id / 2;    // 0..3
    const int warp_n = warp_id % 2;    // 0..1

    // Per-thread accumulators: 4 mma_n × 4 fp32 = 16 fp32 per thread
    float c0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float c1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float c2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float c3[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // ── Loop over N in chunks of BWD_GX_TC_BN=16 (= mma K dim) ──────────────
    for (int n_chunk = 0; n_chunk < N; n_chunk += BWD_GX_TC_BN) {
        const int n_end = min(n_chunk + BWD_GX_TC_BN, N);
        const int g_first = n_chunk / group_size;
        const int g_last = (n_end - 1) / group_size;
        const int n_groups_in_chunk = g_last - g_first + 1;

        // Load palette (first 8 threads)
        if (tid < 8) {
            const int gi = tid / 4;
            const int pi = tid % 4;
            if (gi == 0) spalette[0][pi] = palette[g_first * 4 + pi];
            else if (n_groups_in_chunk > 1) spalette[1][pi] = palette[g_last * 4 + pi];
        }

        // Load grad_y tile: 64x16 = 1024 bf16 = 2 KB. 256 threads × 4 bf16 = 1024 ✓
        {
            const int off = tid * 4;
            const int r = off / BWD_GX_TC_BN;       // 0..63
            const int c = off % BWD_GX_TC_BN;       // 0..12 (mult of 4)
            const int gm = bm * BWD_GX_TC_BM + r;
            const int gn = n_chunk + c;
            if (gm < M && gn + 4 <= N) {
                const int2* src = reinterpret_cast<const int2*>(&grad_y[gm * N + gn]);
                int2 v = __ldg(src);
                *reinterpret_cast<int2*>(&sgy[r][c]) = v;
            } else {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int cc = c + i;
                    const bool ok = (bm * BWD_GX_TC_BM + r < M) && (n_chunk + cc < N);
                    sgy[r][cc] = ok ? grad_y[(bm * BWD_GX_TC_BM + r) * N + (n_chunk + cc)]
                                     : __float2bfloat16(0.0f);
                }
            }
        }

        // Load indices tile: 16x64 uint8 = 1 KB. 256 threads × 4 bytes = 1024 ✓
        // indices has shape (K, N). sidx[BN=16][BK=64] is a transposed view.
        // sidx[r][c] = indices[bk*64 + c][n_chunk + r]
        {
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
        }
        __syncthreads();

        // Materialize W tile: sW[r][c] = palette[group, sidx[r][c]]
        // sW[16][64] = 1024 bf16. 256 threads × 4 = 1024 ✓
        {
            const int off = tid * 4;
            const int r = off / BWD_GX_TC_BK;       // 0..3
            const int c = off % BWD_GX_TC_BK;        // 0..60
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int rr = r;
                const int cc = c + i;
                const int n_global = n_chunk + rr;
                const int group = n_global / group_size;
                const int group_local = (group == g_first) ? 0 : 1;
                const uint8_t idx_val = sidx[rr][cc];
                sW[rr][cc] = spalette[group_local][idx_val];
            }
        }
        __syncthreads();

        // ── mma.sync inner loop ───────────────────────────────────────────
        // A fragment: 16x16 from sgy[warp_m*16..+16][0..16] (M rows × N reduction)
        // B fragment: 16x8 from sW[0..16][warp_n*32 + mma_n*8 .. +8] (N reduction × K output)
        uint32_t a_frag[4];
        uint32_t b_frag[2];

        {
            const int row_in_warp = lane_id % 16;
            const int col_offset = (lane_id / 16) * 8;
            const int sgy_row = warp_m * 16 + row_in_warp;
            const int sgy_col = col_offset;   // 0 or 8 (within BN=16)
            uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(&sgy[sgy_row][sgy_col]);
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                         : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                         : "r"(smem_addr));
        }

        #pragma unroll
        for (int mma_n = 0; mma_n < 4; ++mma_n) {
            {
                const int row_in_warp = lane_id % 16;
                const int sW_row = row_in_warp;                    // 0..15 (N direction)
                const int sW_col = warp_n * 32 + mma_n * 8;        // K output cols
                uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(&sW[sW_row][sW_col]);
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                             : "=r"(b_frag[0]), "=r"(b_frag[1])
                             : "r"(smem_addr));
            }
            float* c = (mma_n == 0) ? c0 : (mma_n == 1) ? c1 : (mma_n == 2) ? c2 : c3;
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                  "r"(b_frag[0]), "r"(b_frag[1]));
        }
        __syncthreads();
    }

    // ── Write output grad_x ──────────────────────────────────────────────
    const int g = lane_id >> 2;
    const int t = lane_id & 3;
    const int row_lo = g;
    const int row_hi = g + 8;
    const int col_base = 2 * t;

    #pragma unroll
    for (int mma_n = 0; mma_n < 4; ++mma_n) {
        const float* c = (mma_n == 0) ? c0 : (mma_n == 1) ? c1 : (mma_n == 2) ? c2 : c3;
        const int k_offset = mma_n * 8;
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int row_in_warp = (r == 0) ? row_lo : row_hi;
            const int m_global = bm * BWD_GX_TC_BM + warp_m * 16 + row_in_warp;
            if (m_global >= M) continue;
            #pragma unroll
            for (int c_idx = 0; c_idx < 2; ++c_idx) {
                const int k_global = bk * BWD_GX_TC_BK + warp_n * 32 + k_offset + col_base + c_idx;
                if (k_global >= K) continue;
                grad_x[m_global * K + k_global] = __float2bfloat16(c[r * 2 + c_idx]);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Backward grad_x kernel (SIMD2 path): grad_x[i, j] = Σ_o grad_y[i, o] * W[j, o]
//  Same structure as forward but transposed: reduce over N instead of K.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void fused_lut_linear_bwd_grad_x_kernel(
    const __nv_bfloat16* __restrict__ grad_y,    // (M, N)
    const __nv_bfloat16* __restrict__ palette,   // (G, 4)
    const uint8_t*        __restrict__ indices,   // (K, N)
    __nv_bfloat16*       __restrict__ grad_x,     // (M, K)
    int M, int K, int N,
    int group_size
) {
    __shared__ __nv_bfloat16 sgy[BWD_GX_BM][BWD_GX_BN];   // 4 KB
    __shared__ uint8_t       sidx[BWD_GX_BK][BWD_GX_BN];   // 2 KB
    __shared__ __nv_bfloat16 spalette[2][4];
    // Pre-materialized W tile: (BK × BN) bf16 = 64 × 32 = 4 KB.
    // Computed once per n_chunk, then accessed as contiguous bf16.
    __shared__ __nv_bfloat16 sW[BWD_GX_BK][BWD_GX_BN];   // 4 KB

    const int bm = blockIdx.x;     // M dimension
    const int bk = blockIdx.y;     // K dimension
    const int tx = threadIdx.x;     // 0..15 (K direction)
    const int ty = threadIdx.y;     // 0..15 (M direction)
    const int linear_tid = ty * 16 + tx;

    // Each thread owns 4x4 = 16 output elements (m_local, k_local)
    const int m_global_base = bm * BWD_GX_BM + ty * 4;
    const int k_global_base = bk * BWD_GX_BK + tx * 4;

    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            acc[i][j] = 0.0f;

    for (int n_chunk = 0; n_chunk < N; n_chunk += BWD_GX_BN) {
        // Determine which groups this chunk touches (1 or 2)
        const int n_start = n_chunk;
        const int n_end   = min(n_start + BWD_GX_BN, N);
        const int g_first = n_start / group_size;
        const int g_last  = (n_end - 1) / group_size;
        const int n_groups_in_chunk = g_last - g_first + 1;

        // Load palette for this chunk (first 4 threads)
        if (ty == 0 && tx < 4) {
            spalette[0][tx] = palette[g_first * 4 + tx];
            if (n_groups_in_chunk > 1) {
                spalette[1][tx] = palette[g_last * 4 + tx];
            }
        }

        // Load grad_y tile: BWD_GX_BM × BWD_GX_BN = 64*32 = 2048 bf16 = 4 KB
        // 4096 bytes / 256 threads = 16 bytes = 1 int4 (8 bf16). Single iter.
        {
            const int off = linear_tid * 8;
            const int r = off / BWD_GX_BN;       // 0..63
            const int c = off % BWD_GX_BN;        // 0..24 (mult of 8)
            const int gm = bm * BWD_GX_BM + r;
            const int gn = n_chunk + c;
            if (gm < M && gn + 8 <= N) {
                const int4* src = reinterpret_cast<const int4*>(&grad_y[gm * N + gn]);
                int4 v = __ldg(src);
                *reinterpret_cast<int4*>(&sgy[r][c]) = v;
            } else {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const bool ok = (bm * BWD_GX_BM + r < M) && (n_chunk + c + i < N);
                    sgy[r][c + i] = ok ? grad_y[(bm * BWD_GX_BM + r) * N + (n_chunk + c + i)]
                                       : __float2bfloat16(0.0f);
                }
            }
        }

        // Load indices tile: BWD_GX_BK × BWD_GX_BN = 64*32 = 2048 uint8 = 2 KB
        // 2048 bytes / 256 threads = 8 bytes = 1 int2 (8 uint8). Single iter.
        {
            const int off = linear_tid * 8;
            const int r = off / BWD_GX_BN;       // 0..63
            const int c = off % BWD_GX_BN;       // 0..24
            const int gk = bk * BWD_GX_BK + r;
            const int gn = n_chunk + c;
            if (gk < K && gn + 8 <= N) {
                const int2* src = reinterpret_cast<const int2*>(&indices[gk * N + gn]);
                int2 v = __ldg(src);
                *reinterpret_cast<int2*>(&sidx[r][c]) = v;
            } else {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const bool ok = (bk * BWD_GX_BK + r < K) && (n_chunk + c + i < N);
                    sidx[r][c + i] = ok ? indices[(bk * BWD_GX_BK + r) * N + (n_chunk + c + i)] : 0;
                }
            }
        }
        __syncthreads();

        // ── Materialize W tile from palette + indices ────────────────────────
        // sW[kk][nn] = palette[group_of(nn), sidx[kk][nn]]
        // BK*BN = 64*32 = 2048 bf16. 256 threads × 8 bf16/thread = 2048 → 1 iter.
        {
            const int off = linear_tid * 8;
            const int r = off / BWD_GX_BN;       // 0..63
            const int c = off % BWD_GX_BN;       // 0..24
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int cc = c + i;
                const int n_global = n_chunk + cc;
                const int group = n_global / group_size;
                const int group_local = (group == g_first) ? 0 : 1;
                const uint8_t idx_val = sidx[r][cc];
                sW[r][cc] = spalette[group_local][idx_val];
            }
        }
        __syncthreads();

        // Compute partial sums: acc[m, k] += Σ_n grad_y[m, n] * W[k, n]
        // Now W is contiguous bf16 — 1 smem load per FMA.
        // Phase II: use bf16 SIMD2 packing for 2× FMA throughput (fp32 accum).
        #pragma unroll 16
        for (int nn = 0; nn < BWD_GX_BN; ++nn) {
            // Load 4 W values for this nn (one per ki), packed as 2 bf162
            __nv_bfloat162 wv2[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int k_local_0 = tx * 4 + i * 2;
                const int k_local_1 = tx * 4 + i * 2 + 1;
                wv2[i] = __halves2bfloat162(sW[k_local_0][nn], sW[k_local_1][nn]);
            }

            // Load 4 grad_y values for this nn (one per mi), packed as 2 bf162
            __nv_bfloat162 gv2[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int m_local_0 = ty * 4 + i * 2;
                const int m_local_1 = ty * 4 + i * 2 + 1;
                gv2[i] = __halves2bfloat162(sgy[m_local_0][nn], sgy[m_local_1][nn]);
            }

            // 4 cross-product FMAs (extract, multiply, accumulate in fp32)
            #pragma unroll
            for (int mi_pair = 0; mi_pair < 2; ++mi_pair) {
                float g0 = __bfloat162float(__low2bfloat16(gv2[mi_pair]));
                float g1 = __bfloat162float(__high2bfloat16(gv2[mi_pair]));
                #pragma unroll
                for (int ki_pair = 0; ki_pair < 2; ++ki_pair) {
                    float w0 = __bfloat162float(__low2bfloat16(wv2[ki_pair]));
                    float w1 = __bfloat162float(__high2bfloat16(wv2[ki_pair]));
                    acc[mi_pair*2 + 0][ki_pair*2 + 0] += g0 * w0;
                    acc[mi_pair*2 + 0][ki_pair*2 + 1] += g0 * w1;
                    acc[mi_pair*2 + 1][ki_pair*2 + 0] += g1 * w0;
                    acc[mi_pair*2 + 1][ki_pair*2 + 1] += g1 * w1;
                }
            }
        }
        __syncthreads();
    }

    // Write grad_x
    #pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
        const int m_global = m_global_base + mi;
        if (m_global >= M) continue;
        #pragma unroll
        for (int ki = 0; ki < 4; ++ki) {
            const int k_global = k_global_base + ki;
            if (k_global >= K) continue;
            grad_x[m_global * K + k_global] = __float2bfloat16(acc[mi][ki]);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Backward grad_palette kernel — fused dW computation + scatter_add
//
//  Computes: dW[j, o] = Σ_i x[i, j] * grad_y[i, o]
//  Then:     grad_palette[group_of(o), indices[j, o]] += dW[j, o]    (fp32 atomic)
//
//  Grid: (cdiv(K, BM_K), cdiv(N, BN_N))
//  Block: (16, 16) = 256 threads; each owns 2x2 = 4 (j, o) pairs.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void fused_lut_linear_bwd_grad_palette_kernel(
    const __nv_bfloat16* __restrict__ x,        // (M, K)
    const __nv_bfloat16* __restrict__ grad_y,   // (M, N)
    const __nv_bfloat16* __restrict__ palette,  // (G, 4) — needed for pre-materialize W
    const uint8_t*        __restrict__ indices,  // (K, N)
    float*               __restrict__ grad_palette,  // (G, 4) fp32
    int M, int K, int N,
    int group_size
) {
    // Smem: x_chunk + gy_chunk + idx + sW (pre-materialized) + accumulator
    __shared__ __nv_bfloat16 sx_chunk[BWD_GP_MM_M][BWD_GP_BM_K];   // 4 KB
    __shared__ __nv_bfloat16 sgy_chunk[BWD_GP_MM_M][BWD_GP_BN_N];   // 4 KB
    __shared__ uint8_t       sidx[BWD_GP_BM_K][BWD_GP_BN_N];        // 4 KB
    __shared__ __nv_bfloat16 sW[BWD_GP_BM_K][BWD_GP_BN_N];          // 8 KB
    // Per-block smem accumulator: 2 groups × 4 palette entries = 8 fp32
    __shared__ float s_acc[2][4];

    const int bk = blockIdx.x;     // K dimension (j)
    const int bn = blockIdx.y;     // N dimension (o)
    const int tx = threadIdx.x;     // 0..15
    const int ty = threadIdx.y;     // 0..15
    const int linear_tid = ty * 16 + tx;     // 0..255

    // Each thread owns 4x4 = 16 (j, o) pairs
    const int j_global_base = bk * BWD_GP_BM_K + ty * 4;
    const int o_global_base = bn * BWD_GP_BN_N + tx * 4;

    // Determine groups touched by this BN tile (1 or 2 — usually 1 since BN=64 ≤ group_size=256)
    const int o_tile_start = bn * BWD_GP_BN_N;
    const int o_tile_end   = min(o_tile_start + BWD_GP_BN_N, N);
    const int g_first = o_tile_start / group_size;
    const int g_last  = (o_tile_end - 1) / group_size;
    const int n_groups_in_tile = g_last - g_first + 1;

    // Init smem accumulators (first 8 threads)
    if (linear_tid < 8) {
        const int g_idx = linear_tid / 4;
        const int p_idx = linear_tid % 4;
        s_acc[g_idx][p_idx] = 0.0f;
    }

    // Load indices tile ONCE: BWD_GP_BM_K × BWD_GP_BN_N = 64*64 = 4096 uint8 = 4 KB
    // 4096 bytes / 256 threads = 16 bytes = 1 int4 (16 uint8). Single iter.
    {
        const int offset = linear_tid * 16;     // 256 * 16 = 4096 ✓
        const int r = offset / BWD_GP_BN_N;     // 0..63
        const int c = offset % BWD_GP_BN_N;     // 0..48 (mult of 16)
        const int gk = bk * BWD_GP_BM_K + r;
        const int gn = bn * BWD_GP_BN_N + c;
        if (gk < K && gn + 16 <= N) {
            const int4* src = reinterpret_cast<const int4*>(&indices[gk * N + gn]);
            int4 v = __ldg(src);
            *reinterpret_cast<int4*>(&sidx[r][c]) = v;
        } else {
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                const int cc = c + i;
                if (cc < BWD_GP_BN_N) {
                    const bool ok = (bk * BWD_GP_BM_K + r < K) && (bn * BWD_GP_BN_N + cc < N);
                    sidx[r][cc] = ok ? indices[(bk * BWD_GP_BM_K + r) * N + (bn * BWD_GP_BN_N + cc)] : 0;
                }
            }
        }
    }
    __syncthreads();

    // ── Pre-materialize W tile from palette + indices ────────────────────────
    // sW[kk][nn] = palette[group_of(nn), sidx[kk][nn]]
    // 4096 bf16 / 256 threads = 16 bf16/thread = 1 int4 (8 bf16) × 2 iters
    #pragma unroll
    for (int iter = 0; iter < 2; ++iter) {
        const int off = (linear_tid + iter * 128) * 8;     // 256*8 = 2048 per iter, 2 iters = 4096 ✓
        const int r = off / BWD_GP_BN_N;                    // 0..63
        const int c = off % BWD_GP_BN_N;                    // 0..56 (mult of 8)
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int cc = c + i;
            const int n_global = bn * BWD_GP_BN_N + cc;
            const int group = n_global / group_size;
            const int group_local = (group == g_first) ? 0 : 1;
            const uint8_t idx_val = sidx[r][cc];
            // palette has shape (G, 4), so palette[group * 4 + idx_val] gives the W value
            sW[r][cc] = palette[group * 4 + idx_val];
        }
    }
    __syncthreads();

    // dW accumulator: 4x4 fp32 per thread (16 values)
    float dW[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            dW[i][j] = 0.0f;

    // Chunk over M
    for (int m_chunk = 0; m_chunk < M; m_chunk += BWD_GP_MM_M) {
        // ── Load x_chunk: (MM_M, BM_K) bf16 = 32*64 = 2048 bf16 = 4 KB ───────
        // 4096 bytes / 256 threads = 16 bytes = 1 int4 (8 bf16). Single iter.
        {
            const int off = linear_tid * 8;
            const int r = off / BWD_GP_BM_K;     // 0..31
            const int c = off % BWD_GP_BM_K;     // 0..56 (mult of 8)
            const int gm = m_chunk + r;
            const int gk = bk * BWD_GP_BM_K + c;
            if (gm < M && gk + 8 <= K) {
                const int4* src = reinterpret_cast<const int4*>(&x[gm * K + gk]);
                int4 v = __ldg(src);
                *reinterpret_cast<int4*>(&sx_chunk[r][c]) = v;
            } else {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int cc = c + i;
                    const bool ok = (m_chunk + r < M) && (bk * BWD_GP_BM_K + cc < K);
                    sx_chunk[r][cc] = ok ? x[(m_chunk + r) * K + (bk * BWD_GP_BM_K + cc)]
                                          : __float2bfloat16(0.0f);
                }
            }
        }

        // ── Load gy_chunk: (MM_M, BN_N) bf16 = 32*64 = 2048 bf16 = 4 KB ──────
        {
            const int off = linear_tid * 8;
            const int r = off / BWD_GP_BN_N;     // 0..31
            const int c = off % BWD_GP_BN_N;     // 0..56 (mult of 8)
            const int gm = m_chunk + r;
            const int gn = bn * BWD_GP_BN_N + c;
            if (gm < M && gn + 8 <= N) {
                const int4* src = reinterpret_cast<const int4*>(&grad_y[gm * N + gn]);
                int4 v = __ldg(src);
                *reinterpret_cast<int4*>(&sgy_chunk[r][c]) = v;
            } else {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int cc = c + i;
                    const bool ok = (m_chunk + r < M) && (bn * BWD_GP_BN_N + cc < N);
                    sgy_chunk[r][cc] = ok ? grad_y[(m_chunk + r) * N + (bn * BWD_GP_BN_N + cc)]
                                            : __float2bfloat16(0.0f);
                }
            }
        }
        __syncthreads();

        // ── Compute dW[j, o] += Σ_i x[i, j] * grad_y[i, o]  ──────────────────
        // Each thread owns 4x4 = 16 (j, o) pairs.
        // Phase II: use bf16 SIMD2 packing for 2× FMA throughput.
        #pragma unroll 8
        for (int mm = 0; mm < BWD_GP_MM_M; ++mm) {
            // Load 4 x values for this mm (one per ji), packed as 2 bf162
            __nv_bfloat162 xv2[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int j_local_0 = ty * 4 + i * 2;
                const int j_local_1 = ty * 4 + i * 2 + 1;
                xv2[i] = __halves2bfloat162(sx_chunk[mm][j_local_0], sx_chunk[mm][j_local_1]);
            }
            // Load 4 grad_y values for this mm (one per oi), packed as 2 bf162
            __nv_bfloat162 gv2[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int o_local_0 = tx * 4 + i * 2;
                const int o_local_1 = tx * 4 + i * 2 + 1;
                gv2[i] = __halves2bfloat162(sgy_chunk[mm][o_local_0], sgy_chunk[mm][o_local_1]);
            }
            // 4 cross-product FMAs (extract, multiply, accumulate in fp32)
            #pragma unroll
            for (int ji_pair = 0; ji_pair < 2; ++ji_pair) {
                float x0 = __bfloat162float(__low2bfloat16(xv2[ji_pair]));
                float x1 = __bfloat162float(__high2bfloat16(xv2[ji_pair]));
                #pragma unroll
                for (int oi_pair = 0; oi_pair < 2; ++oi_pair) {
                    float g0 = __bfloat162float(__low2bfloat16(gv2[oi_pair]));
                    float g1 = __bfloat162float(__high2bfloat16(gv2[oi_pair]));
                    dW[ji_pair*2 + 0][oi_pair*2 + 0] += x0 * g0;
                    dW[ji_pair*2 + 0][oi_pair*2 + 1] += x0 * g1;
                    dW[ji_pair*2 + 1][oi_pair*2 + 0] += x1 * g0;
                    dW[ji_pair*2 + 1][oi_pair*2 + 1] += x1 * g1;
                }
            }
        }
        __syncthreads();
    }

    // ── Scatter_add dW into grad_palette via shared-mem accumulator ──────────
    // For each (j_local, o_local), compute (g, p) and atomically add to s_acc.
    // Note: smem atomicAdd is ~10 cycles uncontended; with 256 threads targeting
    // 8 slots, contention is ~32-way → ~320 cycles total per slot. That's fast.
    // (We tried __match_any_sync warp reduction but the bookkeeping was buggy.
    //  Phase II will replace this with mma.sync + ldmatrix for true TC acceleration.)
    #pragma unroll
    for (int ji = 0; ji < 4; ++ji) {
        const int j_local = ty * 4 + ji;
        #pragma unroll
        for (int oi = 0; oi < 4; ++oi) {
            const int o_local = tx * 4 + oi;
            const int o_global = o_global_base + oi;
            if (o_global >= N) continue;
            const int j_global = j_global_base + ji;
            if (j_global >= K) continue;

            const int group = o_global / group_size;
            const int group_local = (group == g_first) ? 0 : 1;
            const uint8_t p_raw = sidx[j_local][o_local];
            const int p = (p_raw < 4) ? (int)p_raw : 0;
            const float val = dW[ji][oi];

            atomicAdd(&s_acc[group_local][p], val);
        }
    }
    __syncthreads();

    // ── Flush smem accumulators to global grad_palette via atomicAdd ────────
    if (linear_tid < 8) {
        const int g_local = linear_tid / 4;
        const int p = linear_tid % 4;
        if (g_local < n_groups_in_tile) {
            const int g_global = (g_local == 0) ? g_first : g_last;
            const float val = s_acc[g_local][p];
            atomicAdd(&grad_palette[g_global * 4 + p], val);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Backward grad_bias kernel: grad_bias[o] = Σ_i grad_y[i, o]
//  One warp per BWD_GB_BN output columns, reduce over M.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void fused_lut_linear_bwd_grad_bias_kernel(
    const __nv_bfloat16* __restrict__ grad_y,    // (M, N)
    __nv_bfloat16*       __restrict__ grad_bias,  // (N,)
    int M, int N
) {
    const int bn = blockIdx.x;
    const int tx = threadIdx.x;     // 0..31

    const int n_global = bn * BWD_GB_BN + tx;
    if (n_global >= N) return;

    float acc = 0.0f;
    for (int m = 0; m < M; ++m) {
        const __nv_bfloat16 v = grad_y[m * N + n_global];
        acc += __bfloat162float(v);
    }

    grad_bias[n_global] = __float2bfloat16(acc);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Host-side launchers
//  Accept c10::BFloat16* (host-side PyTorch type) and cast to __nv_bfloat16* —
//  the two types are layout-compatible (both IEEE 754 bf16).
// ─────────────────────────────────────────────────────────────────────────────
#include <c10/util/BFloat16.h>

void fused_lut_linear_fwdLauncher(
    const c10::BFloat16* x, const c10::BFloat16* palette, const uint8_t* indices,
    const c10::BFloat16* bias, c10::BFloat16* y,
    int M, int K, int N, int group_size
) {
    // Phase V: pick TC variant if USE_TC_FWD env var is set
    static int use_tc = -1;
    if (use_tc == -1) {
        const char* env = getenv("USE_TC_FWD");
        use_tc = (env && atoi(env) == 1) ? 1 : 0;
    }
    if (use_tc) {
        dim3 grid((M + FWD_TC_BM - 1) / FWD_TC_BM, (N + FWD_TC_BN - 1) / FWD_TC_BN);
        dim3 block(FWD_TC_THREADS);   // 256 threads (1D for simpler warp mapping)
        fused_lut_linear_fwd_tc_kernel<<<grid, block, 0, 0>>>(
            reinterpret_cast<const __nv_bfloat16*>(x),
            reinterpret_cast<const __nv_bfloat16*>(palette),
            indices,
            reinterpret_cast<const __nv_bfloat16*>(bias),
            reinterpret_cast<__nv_bfloat16*>(y),
            M, K, N, group_size);
    } else {
        dim3 grid((M + FWD_BM - 1) / FWD_BM, (N + FWD_BN - 1) / FWD_BN);
        dim3 block(FWD_TX, FWD_TY);   // (16, 16) = 256 threads
        fused_lut_linear_fwd_kernel<<<grid, block, 0, 0>>>(
            reinterpret_cast<const __nv_bfloat16*>(x),
            reinterpret_cast<const __nv_bfloat16*>(palette),
            indices,
            reinterpret_cast<const __nv_bfloat16*>(bias),
            reinterpret_cast<__nv_bfloat16*>(y),
            M, K, N, group_size);
    }
}

void fused_lut_linear_bwd_grad_xLauncher(
    const c10::BFloat16* grad_y, const c10::BFloat16* palette, const uint8_t* indices,
    c10::BFloat16* grad_x, int M, int K, int N, int group_size
) {
    // Phase VI: pick TC variant if USE_TC_BWD_GX env var is set
    static int use_tc = -1;
    if (use_tc == -1) {
        const char* env = getenv("USE_TC_BWD_GX");
        use_tc = (env && atoi(env) == 1) ? 1 : 0;
    }
    if (use_tc) {
        dim3 grid((M + BWD_GX_TC_BM - 1) / BWD_GX_TC_BM, (K + BWD_GX_TC_BK - 1) / BWD_GX_TC_BK);
        dim3 block(BWD_GX_TC_THREADS);   // 256 threads
        fused_lut_linear_bwd_grad_x_tc_kernel<<<grid, block, 0, 0>>>(
            reinterpret_cast<const __nv_bfloat16*>(grad_y),
            reinterpret_cast<const __nv_bfloat16*>(palette),
            indices,
            reinterpret_cast<__nv_bfloat16*>(grad_x),
            M, K, N, group_size);
    } else {
        dim3 grid((M + BWD_GX_BM - 1) / BWD_GX_BM, (K + BWD_GX_BK - 1) / BWD_GX_BK);
        dim3 block(16, 16);
        fused_lut_linear_bwd_grad_x_kernel<<<grid, block, 0, 0>>>(
            reinterpret_cast<const __nv_bfloat16*>(grad_y),
            reinterpret_cast<const __nv_bfloat16*>(palette),
            indices,
            reinterpret_cast<__nv_bfloat16*>(grad_x),
            M, K, N, group_size);
    }
}

void fused_lut_linear_bwd_grad_paletteLauncher(
    const c10::BFloat16* x, const c10::BFloat16* grad_y,
    const c10::BFloat16* palette,  // ADDED in Phase I
    const uint8_t* indices,
    float* grad_palette, int M, int K, int N, int group_size
) {
    dim3 grid((K + BWD_GP_BM_K - 1) / BWD_GP_BM_K, (N + BWD_GP_BN_N - 1) / BWD_GP_BN_N);
    dim3 block(16, 16);
    fused_lut_linear_bwd_grad_palette_kernel<<<grid, block, 0, 0>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(grad_y),
        reinterpret_cast<const __nv_bfloat16*>(palette),  // ADDED
        indices,
        grad_palette,
        M, K, N, group_size);
}

void fused_lut_linear_bwd_grad_biasLauncher(
    const c10::BFloat16* grad_y, c10::BFloat16* grad_bias, int M, int N
) {
    dim3 grid((N + BWD_GB_BN - 1) / BWD_GB_BN);
    dim3 block(BWD_GB_THREADS);   // single warp = 32 threads
    fused_lut_linear_bwd_grad_bias_kernel<<<grid, block, 0, 0>>>(
        reinterpret_cast<const __nv_bfloat16*>(grad_y),
        reinterpret_cast<__nv_bfloat16*>(grad_bias),
        M, N);
}

// ═══════════════════════════════════════════════════════════════════════════
//  PHASE IX — Gumbel-Softmax Soft Forward + Backward Kernels
//
//  Strategy: keep the GEMM portions in PyTorch (cuBLAS via torch.matmul),
//  write only the soft-specific elementwise/scatter kernels in CUDA.
//
//  Soft forward:
//    1. compute_P_W_kernel:  logits (4,K,N) + palette (G,4) + tau + seed
//                            → P (4,K,N) fp16 + W (K,N) bf16
//    2. y = torch.matmul(x, W) + bias   (cuBLAS, in Python)
//
//  Soft backward:
//    1. grad_x = torch.matmul(grad_y, W.T)  (cuBLAS)
//    2. grad_W = torch.matmul(x.T, grad_y).float()  (cuBLAS, fp32 accum)
//    3. grad_logits  = bwd_grad_logits_kernel (NEW, CUDA, elementwise)
//    4. grad_palette = bwd_grad_palette_soft_kernel (NEW, CUDA, weighted scatter)
//    5. grad_bias = grad_y.sum(dim=0)  (PyTorch)
// ═══════════════════════════════════════════════════════════════════════════

#include <cuda_fp16.h>
#include <c10/util/Half.h>

namespace {

// LCG-based Gumbel sampler — deterministic per (seed, idx)
__device__ __forceinline__ float gumbel_sample(uint32_t seed, uint32_t idx) {
    // Mix seed + idx to get a unique state per (j, o, k)
    uint32_t x = seed ^ (idx * 0x9E3779B9u);
    x ^= x >> 13;
    x = x * 1103515245u + 12345u;
    x ^= x >> 17;
    x = x * 1103515245u + 12345u;
    // Convert to float in (0, 1) — take low 24 bits for mantissa
    float u = (float)(x & 0xFFFFFFu) * (1.0f / 16777216.0f);   // [0, 1)
    u = fmaxf(u, 1e-7f);                            // avoid log(0)
    return -logf(-logf(u));                          // Gumbel(0, 1)
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
//  Soft compute_P_W kernel
//
//  For each (j, o):
//    - Sample 4 Gumbel noises
//    - Compute noisy_logits = (logits + gumbel) / tau
//    - Softmax → P (4,) probabilities
//    - W[j, o] = Σ_k P[k] * palette[g, k]
//
//  Block: (16, 16) = 256 threads. Each thread handles one (j, o) element.
//  Grid:  (cdiv(K, 16), cdiv(N, 16)).
// ─────────────────────────────────────────────────────────────────────────────
__global__ void fused_lut_linear_soft_compute_P_W_kernel(
    const __half*        __restrict__ logits,    // (4, K, N) fp16
    const __nv_bfloat16* __restrict__ palette,   // (G, 4) bf16
    __half*              __restrict__ P,         // (4, K, N) fp16 — output
    __nv_bfloat16*       __restrict__ W_out,     // (K, N) bf16 — output
    int K, int N, int group_size,
    float tau, uint32_t step_seed
) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int o = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= K || o >= N) return;

    const int g = o / group_size;
    const int idx = j * N + o;
    const int plane_size = K * N;

    // Load 4 logits
    float l0 = __half2float(logits[0 * plane_size + idx]);
    float l1 = __half2float(logits[1 * plane_size + idx]);
    float l2 = __half2float(logits[2 * plane_size + idx]);
    float l3 = __half2float(logits[3 * plane_size + idx]);

    // Add Gumbel noise + divide by tau
    float inv_tau = 1.0f / tau;
    float n0 = (l0 + gumbel_sample(step_seed, idx * 4 + 0)) * inv_tau;
    float n1 = (l1 + gumbel_sample(step_seed, idx * 4 + 1)) * inv_tau;
    float n2 = (l2 + gumbel_sample(step_seed, idx * 4 + 2)) * inv_tau;
    float n3 = (l3 + gumbel_sample(step_seed, idx * 4 + 3)) * inv_tau;

    // Softmax (numerically stable: subtract max)
    float m = fmaxf(fmaxf(n0, n1), fmaxf(n2, n3));
    float e0 = expf(n0 - m);
    float e1 = expf(n1 - m);
    float e2 = expf(n2 - m);
    float e3 = expf(n3 - m);
    float s = e0 + e1 + e2 + e3;
    float p0 = e0 / s, p1 = e1 / s, p2 = e2 / s, p3 = e3 / s;

    // Save P (4 planes)
    P[0 * plane_size + idx] = __float2half(p0);
    P[1 * plane_size + idx] = __float2half(p1);
    P[2 * plane_size + idx] = __float2half(p2);
    P[3 * plane_size + idx] = __float2half(p3);

    // Compute W = Σ_k P[k] * palette[g, k]
    float c0 = __bfloat162float(palette[g * 4 + 0]);
    float c1 = __bfloat162float(palette[g * 4 + 1]);
    float c2 = __bfloat162float(palette[g * 4 + 2]);
    float c3 = __bfloat162float(palette[g * 4 + 3]);
    float W_val = p0 * c0 + p1 * c1 + p2 * c2 + p3 * c3;
    W_out[idx] = __float2bfloat16(W_val);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Soft compute_P_W kernel — AoS P output (K, N, 4)
//
//  Patch 5a — same math as `fused_lut_linear_soft_compute_P_W_kernel`
//  above, but writes P in (K, N, 4) Array-of-Structures layout so that the
//  4 probabilities for one (j, o) are ADJACENT in memory. This lets the
//  backward kernel replace 4 strided 26-MB-apart loads with a single
//  coalesced 64-bit LDG (4 × fp16 = 8 bytes per (j, o)).
//
//  INPUTS (unchanged from SoA variant):
//    logits:   (4, K, N) fp16 — SoA layout (training parameter, optimizer
//              state expects this; we do NOT change the logits layout)
//    palette:  (G, 4) bf16
//
//  OUTPUTS:
//    P_aos:    (K, N, 4) fp16 — AoS layout, contiguous last-dim
//    W_out:    (K, N) bf16
//
//  Block: (16, 16) = 256 threads. Each thread handles one (j, o) element.
//  Grid:  (cdiv(K, 16), cdiv(N, 16)).
// ─────────────────────────────────────────────────────────────────────────────
__global__ void fused_lut_linear_soft_compute_P_W_aos_kernel(
    const __half*        __restrict__ logits,    // (4, K, N) fp16 — INPUT stays SoA
    const __nv_bfloat16* __restrict__ palette,   // (G, 4)    bf16
    __half*              __restrict__ P_aos,     // (K, N, 4) fp16 — OUTPUT is AoS
    __nv_bfloat16*       __restrict__ W_out,     // (K, N)    bf16
    int K, int N, int group_size,
    float tau, uint32_t step_seed
) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int o = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= K || o >= N) return;

    const int g = o / group_size;
    const int idx      = j * N + o;          // index into a (K, N) plane
    const int idx_aos  = idx * 4;            // index into AoS P (K, N, 4)
    const int plane_size = K * N;

    // ── Load 4 logits from SoA layout (4 reads, ONE per element — bandwidth-light)
    float l0 = __half2float(logits[0 * plane_size + idx]);
    float l1 = __half2float(logits[1 * plane_size + idx]);
    float l2 = __half2float(logits[2 * plane_size + idx]);
    float l3 = __half2float(logits[3 * plane_size + idx]);

    // ── Add Gumbel noise + divide by tau (same LCG as the SoA variant)
    float inv_tau = 1.0f / tau;
    float n0 = (l0 + gumbel_sample(step_seed, idx * 4 + 0)) * inv_tau;
    float n1 = (l1 + gumbel_sample(step_seed, idx * 4 + 1)) * inv_tau;
    float n2 = (l2 + gumbel_sample(step_seed, idx * 4 + 2)) * inv_tau;
    float n3 = (l3 + gumbel_sample(step_seed, idx * 4 + 3)) * inv_tau;

    // ── Softmax (numerically stable)
    float m = fmaxf(fmaxf(n0, n1), fmaxf(n2, n3));
    float e0 = expf(n0 - m);
    float e1 = expf(n1 - m);
    float e2 = expf(n2 - m);
    float e3 = expf(n3 - m);
    float s = e0 + e1 + e2 + e3;
    float p0 = e0 / s, p1 = e1 / s, p2 = e2 / s, p3 = e3 / s;

    // ── Write P_aos (K, N, 4) — 4 adjacent fp16 values
    // Adjacent threads write adjacent (j, o) elements, each 8 bytes — coalesced
    // 64-bit STG across the warp. Equivalent to one __half4 store.
    P_aos[idx_aos + 0] = __float2half(p0);
    P_aos[idx_aos + 1] = __float2half(p1);
    P_aos[idx_aos + 2] = __float2half(p2);
    P_aos[idx_aos + 3] = __float2half(p3);

    // ── Compute W = Σ_k P[k] * palette[g, k]
    float c0 = __bfloat162float(palette[g * 4 + 0]);
    float c1 = __bfloat162float(palette[g * 4 + 1]);
    float c2 = __bfloat162float(palette[g * 4 + 2]);
    float c3 = __bfloat162float(palette[g * 4 + 3]);
    float W_val = p0 * c0 + p1 * c1 + p2 * c2 + p3 * c3;
    W_out[idx] = __float2bfloat16(W_val);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Soft backward — grad_logits kernel
//
//  grad_logits[j, o, k] = grad_W[j, o] * P[j, o, k] * (palette[g, k] - W[j, o])
//
//  where W[j, o] = Σ_k P[k] * palette[g, k]  (reconstructed from P + palette)
//
//  Memory-bound, embarrassingly parallel.
//  Block: (16, 16) = 256 threads. Each thread handles one (j, o) element.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void fused_lut_linear_soft_bwd_grad_logits_kernel(
    const float*         __restrict__ grad_W,      // (K, N) fp32
    const __half*        __restrict__ P,            // (4, K, N) fp16
    const __nv_bfloat16* __restrict__ palette,     // (G, 4) bf16
    __half*              __restrict__ grad_logits, // (4, K, N) fp16 — output
    int K, int N, int group_size
) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int o = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= K || o >= N) return;

    const int g = o / group_size;
    const int idx = j * N + o;
    const int plane_size = K * N;

    float dW = grad_W[idx];
    float p0 = __half2float(P[0 * plane_size + idx]);
    float p1 = __half2float(P[1 * plane_size + idx]);
    float p2 = __half2float(P[2 * plane_size + idx]);
    float p3 = __half2float(P[3 * plane_size + idx]);

    float c0 = __bfloat162float(palette[g * 4 + 0]);
    float c1 = __bfloat162float(palette[g * 4 + 1]);
    float c2 = __bfloat162float(palette[g * 4 + 2]);
    float c3 = __bfloat162float(palette[g * 4 + 3]);

    // W[j, o] = Σ c[k] * p[k]
    float W_val = c0 * p0 + c1 * p1 + c2 * p2 + c3 * p3;

    // grad_logits[j, o, k] = dW * p[k] * (c[k] - W_val)
    grad_logits[0 * plane_size + idx] = __float2half(dW * p0 * (c0 - W_val));
    grad_logits[1 * plane_size + idx] = __float2half(dW * p1 * (c1 - W_val));
    grad_logits[2 * plane_size + idx] = __float2half(dW * p2 * (c2 - W_val));
    grad_logits[3 * plane_size + idx] = __float2half(dW * p3 * (c3 - W_val));
}

// ─────────────────────────────────────────────────────────────────────────────
//  Soft backward — grad_palette kernel (P-weighted scatter_add)
//
//  grad_palette[g, k] += grad_W[j, o] * P[j, o, k]
//
//  Each (j, o) contributes 4 weighted atomic adds (one per k).
//  Memory-bound, embarrassingly parallel.
//  Block: (16, 16) = 256 threads.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void fused_lut_linear_soft_bwd_grad_palette_kernel(
    const float*  __restrict__ grad_W,        // (K, N) fp32
    const __half* __restrict__ P,             // (4, K, N) fp16
    float*        __restrict__ grad_palette, // (G, 4) fp32 — atomicAdd target
    int K, int N, int group_size
) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int o = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= K || o >= N) return;

    const int g = o / group_size;
    const int idx = j * N + o;
    const int plane_size = K * N;

    float dW = grad_W[idx];
    float p0 = __half2float(P[0 * plane_size + idx]);
    float p1 = __half2float(P[1 * plane_size + idx]);
    float p2 = __half2float(P[2 * plane_size + idx]);
    float p3 = __half2float(P[3 * plane_size + idx]);

    // 4 weighted atomic adds — one per palette entry
    atomicAdd(&grad_palette[g * 4 + 0], dW * p0);
    atomicAdd(&grad_palette[g * 4 + 1], dW * p1);
    atomicAdd(&grad_palette[g * 4 + 2], dW * p2);
    atomicAdd(&grad_palette[g * 4 + 3], dW * p3);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Launchers for soft kernels
// ─────────────────────────────────────────────────────────────────────────────
void fused_lut_linear_soft_compute_P_W_Launcher(
    const c10::Half* logits, const c10::BFloat16* palette,
    c10::Half* P, c10::BFloat16* W_out,
    int K, int N, int group_size, float tau, uint32_t step_seed
) {
    dim3 grid((K + 15) / 16, (N + 15) / 16);
    dim3 block(16, 16);
    fused_lut_linear_soft_compute_P_W_kernel<<<grid, block, 0, 0>>>(
        reinterpret_cast<const __half*>(logits),
        reinterpret_cast<const __nv_bfloat16*>(palette),
        reinterpret_cast<__half*>(P),
        reinterpret_cast<__nv_bfloat16*>(W_out),
        K, N, group_size, tau, step_seed);
}

// Patch 5a — Launcher for the AoS-output compute_P_W kernel.
// P_aos is allocated by the caller as a (K*N*4,) fp16 buffer with the
// expected logical shape (K, N, 4) AoS.
void fused_lut_linear_soft_compute_P_W_aos_Launcher(
    const c10::Half* logits, const c10::BFloat16* palette,
    c10::Half* P_aos, c10::BFloat16* W_out,
    int K, int N, int group_size, float tau, uint32_t step_seed
) {
    dim3 grid((K + 15) / 16, (N + 15) / 16);
    dim3 block(16, 16);
    fused_lut_linear_soft_compute_P_W_aos_kernel<<<grid, block, 0, 0>>>(
        reinterpret_cast<const __half*>(logits),
        reinterpret_cast<const __nv_bfloat16*>(palette),
        reinterpret_cast<__half*>(P_aos),
        reinterpret_cast<__nv_bfloat16*>(W_out),
        K, N, group_size, tau, step_seed);
}

void fused_lut_linear_soft_bwd_grad_logits_Launcher(
    const float* grad_W, const c10::Half* P, const c10::BFloat16* palette,
    c10::Half* grad_logits,
    int K, int N, int group_size
) {
    dim3 grid((K + 15) / 16, (N + 15) / 16);
    dim3 block(16, 16);
    fused_lut_linear_soft_bwd_grad_logits_kernel<<<grid, block, 0, 0>>>(
        grad_W,
        reinterpret_cast<const __half*>(P),
        reinterpret_cast<const __nv_bfloat16*>(palette),
        reinterpret_cast<__half*>(grad_logits),
        K, N, group_size);
}

void fused_lut_linear_soft_bwd_grad_palette_Launcher(
    const float* grad_W, const c10::Half* P,
    float* grad_palette,
    int K, int N, int group_size
) {
    dim3 grid((K + 15) / 16, (N + 15) / 16);
    dim3 block(16, 16);
    fused_lut_linear_soft_bwd_grad_palette_kernel<<<grid, block, 0, 0>>>(
        grad_W,
        reinterpret_cast<const __half*>(P),
        grad_palette,
        K, N, group_size);
}

// ─────────────────────────────────────────────────────────────────────────────
//  PHASE IX.b — Fused soft backward kernel (eliminates grad_W intermediate)
//
//  Computes grad_logits + grad_palette in a SINGLE kernel pass, with grad_W
//  computed on-the-fly via M-reduction in shared memory. No (K, N) fp32
//  intermediate tensor needed.
//
//  Block: (16, 16) = 256 threads. Tile: (BK=16, BN=16). Each thread owns 1 (j, o).
//  M reduction: chunks of BM_CHUNK=128, loaded into smem.
// ─────────────────────────────────────────────────────────────────────────────

constexpr int SOFT_BWD_BK = 16;
constexpr int SOFT_BWD_BN = 16;
constexpr int SOFT_BWD_BM_CHUNK = 128;

__global__ void fused_lut_linear_soft_bwd_fused_kernel(
    const __nv_bfloat16* __restrict__ grad_y,    // (M, N) bf16
    const __nv_bfloat16* __restrict__ x,         // (M, K) bf16
    const __half*        __restrict__ P,         // (4, K, N) fp16
    const __nv_bfloat16* __restrict__ palette,   // (G, 4) bf16
    __half*              __restrict__ grad_logits, // (4, K, N) fp16 — output
    float*               __restrict__ grad_palette, // (G, 4) fp32 — atomicAdd output
    int M, int K, int N, int group_size
) {
    // Smem: sx_chunk[128][16] = 4 KB, sgy_chunk[128][16] = 4 KB → 8 KB total
    __shared__ __nv_bfloat16 sx_chunk[SOFT_BWD_BM_CHUNK][SOFT_BWD_BK];
    __shared__ __nv_bfloat16 sgy_chunk[SOFT_BWD_BM_CHUNK][SOFT_BWD_BN];

    const int tk = threadIdx.x;   // 0..15 → j direction
    const int tn = threadIdx.y;   // 0..15 → o direction
    const int tid = tn * 16 + tk;

    const int j = blockIdx.x * SOFT_BWD_BK + tk;
    const int o = blockIdx.y * SOFT_BWD_BN + tn;
    if (j >= K || o >= N) return;

    const int g = o / group_size;
    const int idx = j * N + o;
    const int plane_size = K * N;

    // ── Step 1: Compute grad_W[j, o] = Σ_i x[i, j] * grad_y[i, o] ──────────
    // Loop over M in chunks of BM_CHUNK=128, accumulate in register.
    float grad_W = 0.0f;

    for (int m_chunk = 0; m_chunk < M; m_chunk += SOFT_BWD_BM_CHUNK) {
        // Cooperatively load sx_chunk[BM_CHUNK][BK] = 128*16 = 2048 bf16 = 4 KB
        // 256 threads × 8 bf16 = 2048 ✓ (each thread loads 1 int4 = 8 bf16)
        {
            const int off = tid * 8;
            const int r = off / SOFT_BWD_BK;       // 0..127 (row in chunk)
            const int c = off % SOFT_BWD_BK;        // 0..8 (col, mult of 8)
            const int gm = m_chunk + r;
            const int gk = blockIdx.x * SOFT_BWD_BK + c;
            if (gm < M && gk + 8 <= K) {
                const int4* src = reinterpret_cast<const int4*>(&x[gm * K + gk]);
                int4 v = __ldg(src);
                *reinterpret_cast<int4*>(&sx_chunk[r][c]) = v;
            } else {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int cc = c + i;
                    const bool ok = (m_chunk + r < M) && (blockIdx.x * SOFT_BWD_BK + cc < K);
                    sx_chunk[r][cc] = ok ? x[(m_chunk + r) * K + (blockIdx.x * SOFT_BWD_BK + cc)]
                                          : __float2bfloat16(0.0f);
                }
            }
        }

        // Cooperatively load sgy_chunk[BM_CHUNK][BN] = 128*16 = 2048 bf16 = 4 KB
        {
            const int off = tid * 8;
            const int r = off / SOFT_BWD_BN;
            const int c = off % SOFT_BWD_BN;
            const int gm = m_chunk + r;
            const int gn = blockIdx.y * SOFT_BWD_BN + c;
            if (gm < M && gn + 8 <= N) {
                const int4* src = reinterpret_cast<const int4*>(&grad_y[gm * N + gn]);
                int4 v = __ldg(src);
                *reinterpret_cast<int4*>(&sgy_chunk[r][c]) = v;
            } else {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int cc = c + i;
                    const bool ok = (m_chunk + r < M) && (blockIdx.y * SOFT_BWD_BN + cc < N);
                    sgy_chunk[r][cc] = ok ? grad_y[(m_chunk + r) * N + (blockIdx.y * SOFT_BWD_BN + cc)]
                                            : __float2bfloat16(0.0f);
                }
            }
        }
        __syncthreads();

        // Each thread accumulates its grad_W[j, o] partial sum
        // grad_W += Σ_i sx_chunk[i][tk] * sgy_chunk[i][tn]
        #pragma unroll
        for (int i = 0; i < SOFT_BWD_BM_CHUNK; ++i) {
            float xv = __bfloat162float(sx_chunk[i][tk]);
            float gv = __bfloat162float(sgy_chunk[i][tn]);
            grad_W += xv * gv;
        }
        __syncthreads();
    }

    // ── Step 2: Load P[j, o, 0..3] and palette[g, 0..3] ─────────────────────
    float p0 = __half2float(P[0 * plane_size + idx]);
    float p1 = __half2float(P[1 * plane_size + idx]);
    float p2 = __half2float(P[2 * plane_size + idx]);
    float p3 = __half2float(P[3 * plane_size + idx]);

    float c0 = __bfloat162float(palette[g * 4 + 0]);
    float c1 = __bfloat162float(palette[g * 4 + 1]);
    float c2 = __bfloat162float(palette[g * 4 + 2]);
    float c3 = __bfloat162float(palette[g * 4 + 3]);

    // ── Step 3: Reconstruct W[j, o] = Σ_k P[k] * palette[g, k] ──────────────
    float W_val = c0 * p0 + c1 * p1 + c2 * p2 + c3 * p3;

    // ── Step 4: Compute grad_logits[j, o, k] = grad_W * P[k] * (palette[g, k] - W) ─
    grad_logits[0 * plane_size + idx] = __float2half(grad_W * p0 * (c0 - W_val));
    grad_logits[1 * plane_size + idx] = __float2half(grad_W * p1 * (c1 - W_val));
    grad_logits[2 * plane_size + idx] = __float2half(grad_W * p2 * (c2 - W_val));
    grad_logits[3 * plane_size + idx] = __float2half(grad_W * p3 * (c3 - W_val));

    // ── Step 5: grad_palette[g, k] += grad_W * P[k]  (atomicAdd) ────────────
    atomicAdd(&grad_palette[g * 4 + 0], grad_W * p0);
    atomicAdd(&grad_palette[g * 4 + 1], grad_W * p1);
    atomicAdd(&grad_palette[g * 4 + 2], grad_W * p2);
    atomicAdd(&grad_palette[g * 4 + 3], grad_W * p3);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Launcher for fused soft backward kernel
// ─────────────────────────────────────────────────────────────────────────────
void fused_lut_linear_soft_bwd_fused_Launcher(
    const c10::BFloat16* grad_y, const c10::BFloat16* x,
    const c10::Half* P, const c10::BFloat16* palette,
    c10::Half* grad_logits, float* grad_palette,
    int M, int K, int N, int group_size
) {
    dim3 grid((K + SOFT_BWD_BK - 1) / SOFT_BWD_BK, (N + SOFT_BWD_BN - 1) / SOFT_BWD_BN);
    dim3 block(SOFT_BWD_BK, SOFT_BWD_BN);   // (16, 16) = 256 threads
    fused_lut_linear_soft_bwd_fused_kernel<<<grid, block, 0, 0>>>(
        reinterpret_cast<const __nv_bfloat16*>(grad_y),
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __half*>(P),
        reinterpret_cast<const __nv_bfloat16*>(palette),
        reinterpret_cast<__half*>(grad_logits),
        grad_palette,
        M, K, N, group_size);
}

// ─────────────────────────────────────────────────────────────────────────────
//  PHASE IX.b (AoS) — Fused soft backward kernel reading (K, N, 4) P
//
//  Patch 5b — re-enable the fused bwd path. The SoA variant above was
//  disabled because its 4 strided loads to P[k * plane_size + idx] for
//  k = 0..3 each touched a separate L2 cache line 26 MB apart, making
//  the kernel 3.8× slower than the PyTorch elementwise path despite the
//  PyTorch path materialising a (K, N, 4) fp32 intermediate.
//
//  This variant reads P_aos[idx * 4 + 0..3] — 4 adjacent fp16 values (8
//  bytes total) which the warp coalesces into a single 64-bit LDG.E.U64.
//
//  Block: (16, 16) = 256 threads. Each thread owns 1 (j, o).
//  M-reduction still uses the smem-cached chunks of x and grad_y.
//  Smem: 8 KB per block (4 KB sx_chunk + 4 KB sgy_chunk).
//
//  Inputs:
//    grad_y:    (M, N) bf16
//    x:         (M, K) bf16
//    P_aos:     (K, N, 4) fp16  — AoS layout, contiguous last-dim
//    palette:   (G, 4)    bf16
//
//  Outputs:
//    grad_logits:  (4, K, N) fp16  — OUTPUT stays SoA (optimizer expects this)
//    grad_palette: (G, 4)    fp32  — atomicAdd target
// ─────────────────────────────────────────────────────────────────────────────
__global__ void fused_lut_linear_soft_bwd_fused_aos_kernel(
    const __nv_bfloat16* __restrict__ grad_y,    // (M, N) bf16
    const __nv_bfloat16* __restrict__ x,         // (M, K) bf16
    const __half*        __restrict__ P_aos,      // (K, N, 4) fp16 — AoS INPUT
    const __nv_bfloat16* __restrict__ palette,   // (G, 4)    bf16
    __half*              __restrict__ grad_logits, // (4, K, N) fp16 — OUTPUT stays SoA
    float*               __restrict__ grad_palette, // (G, 4)    fp32 — atomicAdd output
    int M, int K, int N, int group_size
) {
    // Smem: sx_chunk[128][16] = 4 KB, sgy_chunk[128][16] = 4 KB → 8 KB total
    __shared__ __nv_bfloat16 sx_chunk[SOFT_BWD_BM_CHUNK][SOFT_BWD_BK];
    __shared__ __nv_bfloat16 sgy_chunk[SOFT_BWD_BM_CHUNK][SOFT_BWD_BN];

    const int tk = threadIdx.x;   // 0..15 → j direction
    const int tn = threadIdx.y;   // 0..15 → o direction
    const int tid = tn * 16 + tk;

    const int j = blockIdx.x * SOFT_BWD_BK + tk;
    const int o = blockIdx.y * SOFT_BWD_BN + tn;
    if (j >= K || o >= N) return;

    const int g = o / group_size;
    const int idx = j * N + o;
    const int plane_size = K * N;

    // ── Step 1: Compute grad_W[j, o] = Σ_i x[i, j] * grad_y[i, o] ──────────
    // Loop over M in chunks of BM_CHUNK=128, accumulate in register.
    float grad_W = 0.0f;

    for (int m_chunk = 0; m_chunk < M; m_chunk += SOFT_BWD_BM_CHUNK) {
        // Cooperatively load sx_chunk[BM_CHUNK][BK] = 128*16 = 2048 bf16 = 4 KB
        // 256 threads × 8 bf16 = 2048 (each thread loads 1 int4 = 8 bf16)
        {
            const int off = tid * 8;
            const int r = off / SOFT_BWD_BK;       // 0..127 (row in chunk)
            const int c = off % SOFT_BWD_BK;       // 0..8 (col, mult of 8)
            const int gm = m_chunk + r;
            const int gk = blockIdx.x * SOFT_BWD_BK + c;
            if (gm < M && gk + 8 <= K) {
                const int4* src = reinterpret_cast<const int4*>(&x[gm * K + gk]);
                int4 v = __ldg(src);
                *reinterpret_cast<int4*>(&sx_chunk[r][c]) = v;
            } else {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int cc = c + i;
                    const bool ok = (m_chunk + r < M) && (blockIdx.x * SOFT_BWD_BK + cc < K);
                    sx_chunk[r][cc] = ok ? x[(m_chunk + r) * K + (blockIdx.x * SOFT_BWD_BK + cc)]
                                          : __float2bfloat16(0.0f);
                }
            }
        }

        // Cooperatively load sgy_chunk[BM_CHUNK][BN] = 128*16 = 2048 bf16 = 4 KB
        {
            const int off = tid * 8;
            const int r = off / SOFT_BWD_BN;
            const int c = off % SOFT_BWD_BN;
            const int gm = m_chunk + r;
            const int gn = blockIdx.y * SOFT_BWD_BN + c;
            if (gm < M && gn + 8 <= N) {
                const int4* src = reinterpret_cast<const int4*>(&grad_y[gm * N + gn]);
                int4 v = __ldg(src);
                *reinterpret_cast<int4*>(&sgy_chunk[r][c]) = v;
            } else {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int cc = c + i;
                    const bool ok = (m_chunk + r < M) && (blockIdx.y * SOFT_BWD_BN + cc < N);
                    sgy_chunk[r][cc] = ok ? grad_y[(m_chunk + r) * N + (blockIdx.y * SOFT_BWD_BN + cc)]
                                            : __float2bfloat16(0.0f);
                }
            }
        }
        __syncthreads();

        // Each thread accumulates its grad_W[j, o] partial sum
        #pragma unroll
        for (int i = 0; i < SOFT_BWD_BM_CHUNK; ++i) {
            float xv = __bfloat162float(sx_chunk[i][tk]);
            float gv = __bfloat162float(sgy_chunk[i][tn]);
            grad_W += xv * gv;
        }
        __syncthreads();
    }

    // ── Step 2: Load P[j, o, 0..3] — AoS, COALESCED ─────────────────────────
    // Patch 5b fix: 4 adjacent fp16 values, single 64-bit LDG.E.U64 per thread
    // (vs 4 strided 26-MB-apart LDG.E.U16 in the SoA variant).
    const int idx_aos = idx * 4;
    float p0 = __half2float(P_aos[idx_aos + 0]);
    float p1 = __half2float(P_aos[idx_aos + 1]);
    float p2 = __half2float(P_aos[idx_aos + 2]);
    float p3 = __half2float(P_aos[idx_aos + 3]);

    // ── Step 2b: Load palette[g, 0..3] ──────────────────────────────────────
    float c0 = __bfloat162float(palette[g * 4 + 0]);
    float c1 = __bfloat162float(palette[g * 4 + 1]);
    float c2 = __bfloat162float(palette[g * 4 + 2]);
    float c3 = __bfloat162float(palette[g * 4 + 3]);

    // ── Step 3: Reconstruct W[j, o] = Σ_k P[k] * palette[g, k] ──────────────
    float W_val = c0 * p0 + c1 * p1 + c2 * p2 + c3 * p3;

    // ── Step 4: grad_logits[j, o, k] = grad_W * P[k] * (palette[g, k] - W) ─
    // grad_logits OUTPUT stays (4, K, N) SoA — optimizer + checkpoint format
    // expect this layout, so we pay 4 strided STG here (acceptable, write-once).
    grad_logits[0 * plane_size + idx] = __float2half(grad_W * p0 * (c0 - W_val));
    grad_logits[1 * plane_size + idx] = __float2half(grad_W * p1 * (c1 - W_val));
    grad_logits[2 * plane_size + idx] = __float2half(grad_W * p2 * (c2 - W_val));
    grad_logits[3 * plane_size + idx] = __float2half(grad_W * p3 * (c3 - W_val));

    // ── Step 5: grad_palette[g, k] += grad_W * P[k]  (atomicAdd) ────────────
    atomicAdd(&grad_palette[g * 4 + 0], grad_W * p0);
    atomicAdd(&grad_palette[g * 4 + 1], grad_W * p1);
    atomicAdd(&grad_palette[g * 4 + 2], grad_W * p2);
    atomicAdd(&grad_palette[g * 4 + 3], grad_W * p3);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Launcher for fused soft backward AoS kernel
// ─────────────────────────────────────────────────────────────────────────────
void fused_lut_linear_soft_bwd_fused_aos_Launcher(
    const c10::BFloat16* grad_y, const c10::BFloat16* x,
    const c10::Half* P_aos, const c10::BFloat16* palette,
    c10::Half* grad_logits, float* grad_palette,
    int M, int K, int N, int group_size
) {
    dim3 grid((K + SOFT_BWD_BK - 1) / SOFT_BWD_BK, (N + SOFT_BWD_BN - 1) / SOFT_BWD_BN);
    dim3 block(SOFT_BWD_BK, SOFT_BWD_BN);   // (16, 16) = 256 threads
    fused_lut_linear_soft_bwd_fused_aos_kernel<<<grid, block, 0, 0>>>(
        reinterpret_cast<const __nv_bfloat16*>(grad_y),
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __half*>(P_aos),
        reinterpret_cast<const __nv_bfloat16*>(palette),
        reinterpret_cast<__half*>(grad_logits),
        grad_palette,
        M, K, N, group_size);
}
