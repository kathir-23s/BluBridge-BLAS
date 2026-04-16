#include "mycublas.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include <unordered_map>
#include "../SM86/Sgemm_core_template.cuh"

// ============================================================
// Sgemm Addmm SM89 — Fused Matmul + Bias for Ada Lovelace
//
// Operation: C = alpha * (A * B) + beta * bias
// Layout: NN  (A:[M,K] row-major, B:[K,N] row-major)
//
// SM89 vs v34 (SM86):
//   - 5-6 pipeline stages (vs 3) for deeper latency hiding
//   - 16-wide CTA swizzling for Ada Lovelace L2 locality
//   - Multiple tile configs parameterized at compile time
//   - BK=16 (6-stage) and BK=32 (3-stage) variants
// ============================================================

#ifndef MMA_TF32
#define MMA_TF32(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1)                \
    asm volatile(                                                   \
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "     \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"       \
        : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)                     \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1))
#endif

// bias_numel: 1  → scalar broadcast
//             N  → 1D row vector (common case: linear layer bias)
//             0 / nullptr → no bias; beta applied to existing C
template <int BM, int BN, int BK, int STAGES, int THREADS, bool IsAligned>
__global__ void __launch_bounds__(THREADS, 1)
sgemm_addmm_sm89_kernel(
    int M, int N, int K,
    float alpha,
    const float* __restrict__ A, int lda, long long strideA,
    const float* __restrict__ B, int ldb, long long strideB,
    float beta,
    const float* __restrict__ bias, int64_t bias_numel,
    float* __restrict__ C, int ldc, long long strideC,
    int batchCount)
{
    using Config = SgemmTileConfig<BM, BN, BK, STAGES, THREADS>;

    const int batch = blockIdx.z;
    if (batch >= batchCount) return;

    // Robust block mapping for non-square grids (8-wide strips)
    const int sw = 8;
    const int grid_x = (N + BN - 1) / BN, grid_y = (M + BM - 1) / BM;
    const int block_idx = blockIdx.y * grid_x + blockIdx.x;
    const int num_blocks_per_strip = sw * grid_y;
    const int strip_idx = block_idx / num_blocks_per_strip;
    const int strip_off = block_idx % num_blocks_per_strip;
    const int actual_sw = min(sw, grid_x - strip_idx * sw);
    const int bx = strip_idx * sw + (strip_off % actual_sw);
    const int by = strip_off / actual_sw;

    if (bx >= grid_x || by >= grid_y) return;

    const int tid  = threadIdx.x;
    const int lane = tid & 31, wid = tid >> 5;
    const int wy   = wid / Config::WARPS_N;
    const int wx   = wid % Config::WARPS_N;

    extern __shared__ float smem[];

    float acc[Config::MMA_M][Config::MMA_N][4];
    #pragma unroll
    for (int i = 0; i < Config::MMA_M; i++)
        #pragma unroll
        for (int j = 0; j < Config::MMA_N; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    // ----------------------------------------------------------------
    // Global pointer induction
    // A: [M, K] — NT load (BM rows × BK cols, vectorised along K)
    // B: [K, N] — TN load (BK rows × BN cols, vectorised along N)
    // ----------------------------------------------------------------
    const float* gA_ptr[Config::NT_LOAD_ITERS_A];
    const float* gB_ptr[Config::TN_LOAD_ITERS_B];

    #pragma unroll
    for (int i = 0; i < Config::NT_LOAD_ITERS_A; i++) {
        const int r = (tid / Config::NT_THREADS_PER_ROW_A) + i * Config::NT_ROWS_PER_ITER_A;
        const int c = (tid % Config::NT_THREADS_PER_ROW_A) * 4;
        gA_ptr[i] = A + (long long)batch * strideA
                      + (long long)(by * BM + r) * lda + c;
    }
    #pragma unroll
    for (int i = 0; i < Config::TN_LOAD_ITERS_B; i++) {
        const int r = (tid / Config::TN_THREADS_PER_ROW_B) + i * Config::TN_ROWS_PER_ITER_B;
        const int c = (tid % Config::TN_THREADS_PER_ROW_B) * 4;
        gB_ptr[i] = B + (long long)batch * strideB
                      + (long long)r * ldb + (bx * BN + c);
    }

    // ----------------------------------------------------------------
    // Async stage loader
    // ----------------------------------------------------------------
    uint32_t sm_a_off[Config::NT_LOAD_ITERS_A];
    #pragma unroll
    for (int i = 0; i < Config::NT_LOAD_ITERS_A; i++) {
        const int r = (tid / Config::NT_THREADS_PER_ROW_A) + i * Config::NT_ROWS_PER_ITER_A;
        const int c = (tid % Config::NT_THREADS_PER_ROW_A) * 4;
        const int sc = c ^ ((r & 3) << 2);
        sm_a_off[i] = r * Config::BK + sc;
    }

    uint32_t sm_b_off[Config::TN_LOAD_ITERS_B];
    #pragma unroll
    for (int i = 0; i < Config::TN_LOAD_ITERS_B; i++) {
        const int r = (tid / Config::TN_THREADS_PER_ROW_B) + i * Config::TN_ROWS_PER_ITER_B;
        const int c = (tid % Config::TN_THREADS_PER_ROW_B) * 4;
        const int sc = c ^ ((r & 7) << 2);
        sm_b_off[i] = r * Config::BN + sc;
    }

    auto load_to_stage = [&](int stage, int ko) {
        float* As = smem + stage * Config::STAGE_SIZE;
        float* Bs = As   + Config::AS_SIZE;
        #pragma unroll
        for (int i = 0; i < Config::NT_LOAD_ITERS_A; i++) {
            const int r = (tid / Config::NT_THREADS_PER_ROW_A) + i * Config::NT_ROWS_PER_ITER_A, c = (tid % Config::NT_THREADS_PER_ROW_A) * 4;
            const int gr = by * BM + r, gc = ko + c;
            uint32_t sm = __cvta_generic_to_shared(As + sm_a_off[i]);
            if (gr < M && r < BM) {
                int bytes = max(0, min(16, (K - gc) * 4));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm), "l"(gA_ptr[i]), "r"(bytes));
            } else if (r < BM) { *(float4*)(As + sm_a_off[i]) = {0,0,0,0}; }
            gA_ptr[i] += BK;
        }
        #pragma unroll
        for (int i = 0; i < Config::TN_LOAD_ITERS_B; i++) {
            const int r = (tid / Config::TN_THREADS_PER_ROW_B) + i * Config::TN_ROWS_PER_ITER_B, c = (tid % Config::TN_THREADS_PER_ROW_B) * 4;
            const int gk = ko + r, gn = bx * BN + c;
            uint32_t sm = __cvta_generic_to_shared(Bs + sm_b_off[i]);
            if (gk < K && r < BK) {
                int bytes = max(0, min(16, (N - gn) * 4));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm), "l"(gB_ptr[i]), "r"(bytes));
            } else if (r < BK) { *(float4*)(Bs + sm_b_off[i]) = {0,0,0,0}; }
            gB_ptr[i] += (long long)BK * ldb;
        }
    };

    // ----------------------------------------------------------------
    // Register fragment helpers
    // ----------------------------------------------------------------
    const int g_sh = lane / 4, t_sh = lane % 4;

    // ----------------------------------------------------------------
    // Multi-stage pipeline warmup  (STAGES-1 stages pre-filled)
    // ----------------------------------------------------------------

    auto load_frA = [&](uint32_t reg[4], int ks, int mi, int st) {
        float* As = smem + st * Config::STAGE_SIZE;
        const int lb = wy * Config::WARP_TILE_M + mi * 16, lr0 = lb + (lane / 4), lr8 = lr0 + 8, lc = ks + (lane % 4);
        auto ga = [&](int r, int c) { return *(const uint32_t*)(&As[r * Config::BK + (c ^ ((r & 3) << 2))]); };
        reg[0] = ga(lr0, lc); reg[1] = ga(lr8, lc); reg[2] = ga(lr0, lc+4); reg[3] = ga(lr8, lc+4);
    };

    auto load_frB = [&](uint32_t reg[2], int ks, int ni, int st) {
        float* Bs = smem + st * Config::STAGE_SIZE + Config::AS_SIZE;
        const int lr0 = ks + (lane % 4), lr4 = lr0 + 4, lc = wx * Config::WARP_TILE_N + ni * 8 + (lane / 4);
        auto gb = [&](int r, int c) { return *(const uint32_t*)(&Bs[r * Config::BN + (c ^ ((r & 7) << 2))]); };
        reg[0] = gb(lr0, lc); reg[1] = gb(lr4, lc);
    };

    // ----------------------------------------------------------------
    // Multi-stage pipeline warmup  (STAGES-1 stages pre-filled)
    // ----------------------------------------------------------------
    load_to_stage(0, 0);
    asm volatile("cp.async.commit_group;\n");
    #pragma unroll
    for (int s = 1; s < Config::STAGES - 1; s++) {
        if (s * Config::BK < K) load_to_stage(s, s * Config::BK);
        asm volatile("cp.async.commit_group;\n");
    }

    int ws = Config::STAGES - 1, rs = 0;
    uint32_t frA[2][Config::MMA_M][4], frB[2][Config::MMA_N][2];

    asm volatile("cp.async.wait_group %0;\n" :: "n"(Config::STAGES - 2));
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < Config::MMA_M; i++) load_frA(frA[0][i], 0, i, rs);
    #pragma unroll
    for (int j = 0; j < Config::MMA_N; j++) load_frB(frB[0][j], 0, j, rs);

    // ----------------------------------------------------------------
    // Main K loop
    // ----------------------------------------------------------------
    for (int k = 0; k < K; k += Config::BK) {
        if (k + (Config::STAGES - 1) * Config::BK < K)
            load_to_stage(ws, k + (Config::STAGES - 1) * Config::BK);
        asm volatile("cp.async.commit_group;\n");

        #pragma unroll
        for (int ks = 0; ks < Config::BK; ks += 16) {
            // First 8-step MMA (ks)
            #pragma unroll
            for (int i = 0; i < Config::MMA_M; i++) {
                #pragma unroll
                for (int j = 0; j < Config::MMA_N; j++) {
                    MMA_TF32(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3],
                             frA[0][i][0], frA[0][i][1], frA[0][i][2], frA[0][i][3],
                             frB[0][j][0], frB[0][j][1]);
                    if (i == 0) load_frB(frB[1][j], ks + 8, j, rs);
                }
                load_frA(frA[1][i], ks + 8, i, rs);
            }
            
            // Second 8-step MMA (ks + 8)
            #pragma unroll
            for (int i = 0; i < Config::MMA_M; i++) {
                #pragma unroll
                for (int j = 0; j < Config::MMA_N; j++) {
                    MMA_TF32(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3],
                             frA[1][i][0], frA[1][i][1], frA[1][i][2], frA[1][i][3],
                             frB[1][j][0], frB[1][j][1]);
                }
            }

            // Prepare for next 16-element block or next stage
            if (ks + 16 < Config::BK) {
                #pragma unroll
                for (int i = 0; i < Config::MMA_M; i++) {
                    #pragma unroll
                    for (int j = 0; j < Config::MMA_N; j++) if (i == 0) load_frB(frB[0][j], ks + 16, j, rs);
                    load_frA(frA[0][i], ks + 16, i, rs);
                }
            } else if (k + Config::BK < K) {
                asm volatile("cp.async.wait_group %0;\n" :: "n"(Config::STAGES - 2));
                __syncthreads();
                rs = (rs + 1) % Config::STAGES;
                ws = (ws + 1) % Config::STAGES;
                #pragma unroll
                for (int i = 0; i < Config::MMA_M; i++) {
                    #pragma unroll
                    for (int j = 0; j < Config::MMA_N; j++) if (i == 0) load_frB(frB[0][j], 0, j, rs);
                    load_frA(frA[0][i], 0, i, rs);
                }
            }
        }
    }

    // ----------------------------------------------------------------
    // Epilogue: C[r,c] = alpha * acc + beta * bias[c]
    // MMA output: lane l owns rows {l/4, l/4+8}, cols {2*(l%4), 2*(l%4)+1}
    // ----------------------------------------------------------------
    const int g_epi = lane / 4, t_epi = lane % 4;
    float* dC = C + (long long)batch * strideC;

    #pragma unroll
    for (int i = 0; i < Config::MMA_M; i++) {
        #pragma unroll
        for (int j = 0; j < Config::MMA_N; j++) {
            const int r0 = by * BM + wy * Config::WARP_TILE_M + i * 16 + g_epi;
            const int r8 = r0 + 8;
            const int c0 = bx * BN + wx * Config::WARP_TILE_N + j * 8 + t_epi * 2;
            const int c1 = c0 + 1;

            // Load bias values (Scalar, Vector, or Matrix broadcast)
            float b0 = 0.f, b1 = 0.f;
            if (bias) {
                if (bias_numel == 1) {
                    b0 = b1 = bias[0];
                } else if (bias_numel == (int64_t)N) {
                    if (c0 < N) b0 = bias[c0];
                    if (c1 < N) b1 = bias[c1];
                } else if (bias_numel == (int64_t)M * N) {
                    const float* pb = bias + (long long)(batch % M) * N + c0;
                    if (c0 < N) b0 = pb[0];
                    if (c1 < N) b1 = pb[1];
                }
            }

            auto store = [&](int r, int c, float f, float b) __attribute__((always_inline)) {
                if (r >= M || c >= N) return;
                float* dst = &dC[(long long)r * ldc + c];
                if (bias) {
                    *dst = alpha * f + beta * b;
                } else {
                    *dst = alpha * f + (beta == 0.f ? 0.f : beta * (*dst));
                }
            };

            store(r0, c0, acc[i][j][0], b0);
            store(r0, c1, acc[i][j][1], b1);
            store(r8, c0, acc[i][j][2], b0);
            store(r8, c1, acc[i][j][3], b1);
        }
    }
}

// ----------------------------------------------------------------
// Launch helper (aligned/unaligned specialisation)
// ----------------------------------------------------------------
template <int BM, int BN, int BK, int STAGES, int THREADS, bool IsAligned>
static void launch_sgemm_addmm_sm89(
    cudaStream_t stream, int M, int N, int K,
    float alpha,
    const float* A, int lda, long long strideA,
    const float* B, int ldb, long long strideB,
    float beta,
    const float* bias, int64_t bias_numel,
    float* C, int ldc, long long strideC,
    int batchCount)
{
    using Config = SgemmTileConfig<BM, BN, BK, STAGES, THREADS>;
    static const size_t smem_bytes = STAGES * Config::STAGE_SIZE * sizeof(float);

    static std::unordered_map<const void*, bool> done;
    const void* fn = (const void*)sgemm_addmm_sm89_kernel<BM, BN, BK, STAGES, THREADS, IsAligned>;
    if (!done[fn]) {
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
        done[fn] = true;
    }

    const int gx = (N + BN - 1) / BN, gy = (M + BM - 1) / BM;
    sgemm_addmm_sm89_kernel<BM, BN, BK, STAGES, THREADS, IsAligned>
        <<<dim3(gx, gy, batchCount), THREADS, smem_bytes, stream>>>(
            M, N, K, alpha, A, lda, strideA, B, ldb, strideB,
            beta, bias, bias_numel, C, ldc, strideC, batchCount);
}

// ----------------------------------------------------------------
// Dispatch: picks alignment variant, no splitK (addmm is single-pass)
// ----------------------------------------------------------------
template <int BM, int BN, int BK, int STAGES, int THREADS>
static void dispatch_sgemm_addmm_sm89(
    cudaStream_t stream, int M, int N, int K,
    float alpha,
    const float* A, int lda, long long strideA,
    const float* B, int ldb, long long strideB,
    float beta,
    const float* bias, int64_t bias_numel,
    float* C, int ldc, long long strideC,
    int batchCount)
{
    const bool aligned = (((size_t)A & 15) == 0) && (((size_t)B & 15) == 0)
                      && ((lda & 3) == 0) && ((ldb & 3) == 0);
    if (aligned)
        launch_sgemm_addmm_sm89<BM, BN, BK, STAGES, THREADS, true>(
            stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB,
            beta, bias, bias_numel, C, ldc, strideC, batchCount);
    else
        launch_sgemm_addmm_sm89<BM, BN, BK, STAGES, THREADS, false>(
            stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB,
            beta, bias, bias_numel, C, ldc, strideC, batchCount);
}

// ================================================================
// Public entry point
//
// Tile heuristic for SM89 Ada Lovelace:
//
//   M >= 256, N >= 128    →  256×128×32, 2-stage  (max tiles + wide BK)
//                              smem: 2 × (256×32 + 32×128) × 4 = 98.3 KB
//
//   M,N >= 128, K >= 256  →  128×128×32, 4-stage  (wide BK, better efficiency)
//                              smem: 4 × (128×32 + 32×128) × 4 = 131 KB (Requires SM89)
//                              Wait, SM89 (Ada) supports up to 100KB per block by default.
//                              128x128x32 with 3 stages is 96KB.
//
//   M,N >= 128, K <  256  →  128×128×16, 6-stage  (max stages, best latency hiding)
//                              smem: 6 × (128×16 + 16×128) × 4 = 98.3 KB
//
//   M < 128 or N < 128    →  64×64×16, 6-stage    (higher occupancy for small tiles)
//                              smem: 6 × (64×16 + 16×64) × 4 = 49.2 KB
// ================================================================
extern "C" void mycublasSgemmAddmm_sm89(
    mycublasHandle_t handle,
    int M, int N, int K,
    const float alpha,
    const float* A, int lda, long long int strideA,
    const float* B, int ldb, long long int strideB,
    const float beta,
    const float* bias, int64_t bias_numel,
    float* C, int ldc, long long int strideC,
    int batchCount)
{
    cudaStream_t stream = handle ? handle->stream : 0;

    if (M >= 256 && N >= 128) {
        // Large tiles: best for large GEMMs on high-SM count Ada GPUs
        dispatch_sgemm_addmm_sm89<256, 128, 32, 2, 256>(
            stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB,
            beta, bias, bias_numel, C, ldc, strideC, batchCount);
    } else if (M >= 128 && N >= 128) {
        if (K >= 256) {
            // Wide BK, 3 stages to stay under 100KB smem
            dispatch_sgemm_addmm_sm89<128, 128, 32, 3, 128>(
                stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB,
                beta, bias, bias_numel, C, ldc, strideC, batchCount);
        } else {
            // Narrow BK, 6 stages for max pipeline depth
            dispatch_sgemm_addmm_sm89<128, 128, 16, 6, 128>(
                stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB,
                beta, bias, bias_numel, C, ldc, strideC, batchCount);
        }
    } else {
        // Small tiles: 64×64 doubles occupancy vs 128×128
        dispatch_sgemm_addmm_sm89<64, 64, 16, 6, 64>(
            stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB,
            beta, bias, bias_numel, C, ldc, strideC, batchCount);
    }
}
