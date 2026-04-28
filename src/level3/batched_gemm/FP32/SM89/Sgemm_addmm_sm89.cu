// // // #include "mycublas.h"
// // // #include <cuda_runtime.h>
// // // #include <stdint.h>
// // // #include <unordered_map>
// // // #include "../SM86/Sgemm_core_template.cuh"

// // // // ============================================================
// // // // Sgemm Addmm SM89 — Fused Matmul + Bias for Ada Lovelace
// // // //
// // // // Operation: C = alpha * (A * B) + beta * bias
// // // // Layout: NN  (A:[M,K] row-major, B:[K,N] row-major)
// // // //
// // // // SM89 vs v34 (SM86):
// // // //   - 5-6 pipeline stages (vs 3) for deeper latency hiding
// // // //   - 16-wide CTA swizzling for Ada Lovelace L2 locality
// // // //   - Multiple tile configs parameterized at compile time
// // // //   - BK=16 (6-stage) and BK=32 (3-stage) variants
// // // // ============================================================

// // // #ifndef MMA_TF32
// // // #define MMA_TF32(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1)                \
// // //     asm volatile(                                                   \
// // //         "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "     \
// // //         "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"       \
// // //         : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)                     \
// // //         : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1))
// // // #endif

// // // // bias_numel: 1  → scalar broadcast
// // // //             N  → 1D row vector (common case: linear layer bias)
// // // //             0 / nullptr → no bias; beta applied to existing C
// // // template <int BM, int BN, int BK, int STAGES, int THREADS, bool IsAligned>
// // // __global__ void __launch_bounds__(THREADS, 1)
// // // sgemm_addmm_sm89_kernel(
// // //     int M, int N, int K,
// // //     float alpha,
// // //     const float* __restrict__ A, int lda, long long strideA,
// // //     const float* __restrict__ B, int ldb, long long strideB,
// // //     float beta,
// // //     const float* __restrict__ bias, int64_t bias_numel,
// // //     float* __restrict__ C, int ldc, long long strideC,
// // //     int batchCount)
// // // {
// // //     using Config = SgemmTileConfig<BM, BN, BK, STAGES, THREADS>;

// // //     const int batch = blockIdx.z;
// // //     if (batch >= batchCount) return;

// // //     // Robust block mapping for non-square grids (8-wide strips)
// // //     const int sw = 8;
// // //     const int grid_x = (N + BN - 1) / BN, grid_y = (M + BM - 1) / BM;
// // //     const int block_idx = blockIdx.y * grid_x + blockIdx.x;
// // //     const int num_blocks_per_strip = sw * grid_y;
// // //     const int strip_idx = block_idx / num_blocks_per_strip;
// // //     const int strip_off = block_idx % num_blocks_per_strip;
// // //     const int actual_sw = min(sw, grid_x - strip_idx * sw);
// // //     const int bx = strip_idx * sw + (strip_off % actual_sw);
// // //     const int by = strip_off / actual_sw;

// // //     if (bx >= grid_x || by >= grid_y) return;

// // //     const int tid  = threadIdx.x;
// // //     const int lane = tid & 31, wid = tid >> 5;
// // //     const int wy   = wid / Config::WARPS_N;
// // //     const int wx   = wid % Config::WARPS_N;

// // //     extern __shared__ float smem[];

// // //     float acc[Config::MMA_M][Config::MMA_N][4];
// // //     #pragma unroll
// // //     for (int i = 0; i < Config::MMA_M; i++)
// // //         #pragma unroll
// // //         for (int j = 0; j < Config::MMA_N; j++)
// // //             acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

// // //     // ----------------------------------------------------------------
// // //     // Global pointer induction
// // //     // A: [M, K] — NT load (BM rows × BK cols, vectorised along K)
// // //     // B: [K, N] — TN load (BK rows × BN cols, vectorised along N)
// // //     // ----------------------------------------------------------------
// // //     const float* gA_ptr[Config::NT_LOAD_ITERS_A];
// // //     const float* gB_ptr[Config::TN_LOAD_ITERS_B];

// // //     #pragma unroll
// // //     for (int i = 0; i < Config::NT_LOAD_ITERS_A; i++) {
// // //         const int r = (tid / Config::NT_THREADS_PER_ROW_A) + i * Config::NT_ROWS_PER_ITER_A;
// // //         const int c = (tid % Config::NT_THREADS_PER_ROW_A) * 4;
// // //         gA_ptr[i] = A + (long long)batch * strideA
// // //                       + (long long)(by * BM + r) * lda + c;
// // //     }
// // //     #pragma unroll
// // //     for (int i = 0; i < Config::TN_LOAD_ITERS_B; i++) {
// // //         const int r = (tid / Config::TN_THREADS_PER_ROW_B) + i * Config::TN_ROWS_PER_ITER_B;
// // //         const int c = (tid % Config::TN_THREADS_PER_ROW_B) * 4;
// // //         gB_ptr[i] = B + (long long)batch * strideB
// // //                       + (long long)r * ldb + (bx * BN + c);
// // //     }

// // //     // ----------------------------------------------------------------
// // //     // Async stage loader
// // //     // ----------------------------------------------------------------
// // //     uint32_t sm_a_off[Config::NT_LOAD_ITERS_A];
// // //     #pragma unroll
// // //     for (int i = 0; i < Config::NT_LOAD_ITERS_A; i++) {
// // //         const int r = (tid / Config::NT_THREADS_PER_ROW_A) + i * Config::NT_ROWS_PER_ITER_A;
// // //         const int c = (tid % Config::NT_THREADS_PER_ROW_A) * 4;
// // //         const int sc = c ^ ((r & 3) << 2);
// // //         sm_a_off[i] = r * Config::BK + sc;
// // //     }

// // //     uint32_t sm_b_off[Config::TN_LOAD_ITERS_B];
// // //     #pragma unroll
// // //     for (int i = 0; i < Config::TN_LOAD_ITERS_B; i++) {
// // //         const int r = (tid / Config::TN_THREADS_PER_ROW_B) + i * Config::TN_ROWS_PER_ITER_B;
// // //         const int c = (tid % Config::TN_THREADS_PER_ROW_B) * 4;
// // //         const int sc = c ^ ((r & 7) << 2);
// // //         sm_b_off[i] = r * Config::BN + sc;
// // //     }

// // //     auto load_to_stage = [&](int stage, int ko) {
// // //         float* As = smem + stage * Config::STAGE_SIZE;
// // //         float* Bs = As   + Config::AS_SIZE;
// // //         #pragma unroll
// // //         for (int i = 0; i < Config::NT_LOAD_ITERS_A; i++) {
// // //             const int r = (tid / Config::NT_THREADS_PER_ROW_A) + i * Config::NT_ROWS_PER_ITER_A, c = (tid % Config::NT_THREADS_PER_ROW_A) * 4;
// // //             const int gr = by * BM + r, gc = ko + c;
// // //             uint32_t sm = __cvta_generic_to_shared(As + sm_a_off[i]);
// // //             if (gr < M && r < BM) {
// // //                 int bytes = max(0, min(16, (K - gc) * 4));
// // //                 asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm), "l"(gA_ptr[i]), "r"(bytes));
// // //             } else if (r < BM) { *(float4*)(As + sm_a_off[i]) = {0,0,0,0}; }
// // //             gA_ptr[i] += BK;
// // //         }
// // //         #pragma unroll
// // //         for (int i = 0; i < Config::TN_LOAD_ITERS_B; i++) {
// // //             const int r = (tid / Config::TN_THREADS_PER_ROW_B) + i * Config::TN_ROWS_PER_ITER_B, c = (tid % Config::TN_THREADS_PER_ROW_B) * 4;
// // //             const int gk = ko + r, gn = bx * BN + c;
// // //             uint32_t sm = __cvta_generic_to_shared(Bs + sm_b_off[i]);
// // //             if (gk < K && r < BK) {
// // //                 int bytes = max(0, min(16, (N - gn) * 4));
// // //                 asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm), "l"(gB_ptr[i]), "r"(bytes));
// // //             } else if (r < BK) { *(float4*)(Bs + sm_b_off[i]) = {0,0,0,0}; }
// // //             gB_ptr[i] += (long long)BK * ldb;
// // //         }
// // //     };

// // //     // ----------------------------------------------------------------
// // //     // Register fragment helpers
// // //     // ----------------------------------------------------------------
// // //     const int g_sh = lane / 4, t_sh = lane % 4;

// // //     // ----------------------------------------------------------------
// // //     // Multi-stage pipeline warmup  (STAGES-1 stages pre-filled)
// // //     // ----------------------------------------------------------------

// // //     auto load_frA = [&](uint32_t reg[4], int ks, int mi, int st) {
// // //         float* As = smem + st * Config::STAGE_SIZE;
// // //         const int lb = wy * Config::WARP_TILE_M + mi * 16, lr0 = lb + (lane / 4), lr8 = lr0 + 8, lc = ks + (lane % 4);
// // //         auto ga = [&](int r, int c) { return *(const uint32_t*)(&As[r * Config::BK + (c ^ ((r & 3) << 2))]); };
// // //         reg[0] = ga(lr0, lc); reg[1] = ga(lr8, lc); reg[2] = ga(lr0, lc+4); reg[3] = ga(lr8, lc+4);
// // //     };

// // //     auto load_frB = [&](uint32_t reg[2], int ks, int ni, int st) {
// // //         float* Bs = smem + st * Config::STAGE_SIZE + Config::AS_SIZE;
// // //         const int lr0 = ks + (lane % 4), lr4 = lr0 + 4, lc = wx * Config::WARP_TILE_N + ni * 8 + (lane / 4);
// // //         auto gb = [&](int r, int c) { return *(const uint32_t*)(&Bs[r * Config::BN + (c ^ ((r & 7) << 2))]); };
// // //         reg[0] = gb(lr0, lc); reg[1] = gb(lr4, lc);
// // //     };

// // //     // ----------------------------------------------------------------
// // //     // Multi-stage pipeline warmup  (STAGES-1 stages pre-filled)
// // //     // ----------------------------------------------------------------
// // //     load_to_stage(0, 0);
// // //     asm volatile("cp.async.commit_group;\n");
// // //     #pragma unroll
// // //     for (int s = 1; s < Config::STAGES - 1; s++) {
// // //         if (s * Config::BK < K) load_to_stage(s, s * Config::BK);
// // //         asm volatile("cp.async.commit_group;\n");
// // //     }

// // //     int ws = Config::STAGES - 1, rs = 0;
// // //     uint32_t frA[2][Config::MMA_M][4], frB[2][Config::MMA_N][2];

// // //     asm volatile("cp.async.wait_group %0;\n" :: "n"(Config::STAGES - 2));
// // //     __syncthreads();
// // //     #pragma unroll
// // //     for (int i = 0; i < Config::MMA_M; i++) load_frA(frA[0][i], 0, i, rs);
// // //     #pragma unroll
// // //     for (int j = 0; j < Config::MMA_N; j++) load_frB(frB[0][j], 0, j, rs);

// // //     // ----------------------------------------------------------------
// // //     // Main K loop
// // //     // ----------------------------------------------------------------
// // //     for (int k = 0; k < K; k += Config::BK) {
// // //         if (k + (Config::STAGES - 1) * Config::BK < K)
// // //             load_to_stage(ws, k + (Config::STAGES - 1) * Config::BK);
// // //         asm volatile("cp.async.commit_group;\n");

// // //         #pragma unroll
// // //         for (int ks = 0; ks < Config::BK; ks += 16) {
// // //             // First 8-step MMA (ks)
// // //             #pragma unroll
// // //             for (int i = 0; i < Config::MMA_M; i++) {
// // //                 #pragma unroll
// // //                 for (int j = 0; j < Config::MMA_N; j++) {
// // //                     MMA_TF32(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3],
// // //                              frA[0][i][0], frA[0][i][1], frA[0][i][2], frA[0][i][3],
// // //                              frB[0][j][0], frB[0][j][1]);
// // //                     if (i == 0) load_frB(frB[1][j], ks + 8, j, rs);
// // //                 }
// // //                 load_frA(frA[1][i], ks + 8, i, rs);
// // //             }
            
// // //             // Second 8-step MMA (ks + 8)
// // //             #pragma unroll
// // //             for (int i = 0; i < Config::MMA_M; i++) {
// // //                 #pragma unroll
// // //                 for (int j = 0; j < Config::MMA_N; j++) {
// // //                     MMA_TF32(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3],
// // //                              frA[1][i][0], frA[1][i][1], frA[1][i][2], frA[1][i][3],
// // //                              frB[1][j][0], frB[1][j][1]);
// // //                 }
// // //             }

// // //             // Prepare for next 16-element block or next stage
// // //             if (ks + 16 < Config::BK) {
// // //                 #pragma unroll
// // //                 for (int i = 0; i < Config::MMA_M; i++) {
// // //                     #pragma unroll
// // //                     for (int j = 0; j < Config::MMA_N; j++) if (i == 0) load_frB(frB[0][j], ks + 16, j, rs);
// // //                     load_frA(frA[0][i], ks + 16, i, rs);
// // //                 }
// // //             } else if (k + Config::BK < K) {
// // //                 asm volatile("cp.async.wait_group %0;\n" :: "n"(Config::STAGES - 2));
// // //                 __syncthreads();
// // //                 rs = (rs + 1) % Config::STAGES;
// // //                 ws = (ws + 1) % Config::STAGES;
// // //                 #pragma unroll
// // //                 for (int i = 0; i < Config::MMA_M; i++) {
// // //                     #pragma unroll
// // //                     for (int j = 0; j < Config::MMA_N; j++) if (i == 0) load_frB(frB[0][j], 0, j, rs);
// // //                     load_frA(frA[0][i], 0, i, rs);
// // //                 }
// // //             }
// // //         }
// // //     }

// // //     // ----------------------------------------------------------------
// // //     // Epilogue: C[r,c] = alpha * acc + beta * bias[c]
// // //     // MMA output: lane l owns rows {l/4, l/4+8}, cols {2*(l%4), 2*(l%4)+1}
// // //     // ----------------------------------------------------------------
// // //     const int g_epi = lane / 4, t_epi = lane % 4;
// // //     float* dC = C + (long long)batch * strideC;

// // //     #pragma unroll
// // //     for (int i = 0; i < Config::MMA_M; i++) {
// // //         #pragma unroll
// // //         for (int j = 0; j < Config::MMA_N; j++) {
// // //             const int r0 = by * BM + wy * Config::WARP_TILE_M + i * 16 + g_epi;
// // //             const int r8 = r0 + 8;
// // //             const int c0 = bx * BN + wx * Config::WARP_TILE_N + j * 8 + t_epi * 2;
// // //             const int c1 = c0 + 1;

// // //             // Load bias values (Scalar, Vector, or Matrix broadcast)
// // //             float b0 = 0.f, b1 = 0.f;
// // //             if (bias) {
// // //                 if (bias_numel == 1) {
// // //                     b0 = b1 = bias[0];
// // //                 } else if (bias_numel == (int64_t)N) {
// // //                     if (c0 < N) b0 = bias[c0];
// // //                     if (c1 < N) b1 = bias[c1];
// // //                 } else if (bias_numel == (int64_t)M * N) {
// // //                     const float* pb = bias + (long long)(batch % M) * N + c0;
// // //                     if (c0 < N) b0 = pb[0];
// // //                     if (c1 < N) b1 = pb[1];
// // //                 }
// // //             }

// // //             auto store = [&](int r, int c, float f, float b) __attribute__((always_inline)) {
// // //                 if (r >= M || c >= N) return;
// // //                 float* dst = &dC[(long long)r * ldc + c];
// // //                 if (bias) {
// // //                     *dst = alpha * f + beta * b;
// // //                 } else {
// // //                     *dst = alpha * f + (beta == 0.f ? 0.f : beta * (*dst));
// // //                 }
// // //             };

// // //             store(r0, c0, acc[i][j][0], b0);
// // //             store(r0, c1, acc[i][j][1], b1);
// // //             store(r8, c0, acc[i][j][2], b0);
// // //             store(r8, c1, acc[i][j][3], b1);
// // //         }
// // //     }
// // // }

// // // // ----------------------------------------------------------------
// // // // Launch helper (aligned/unaligned specialisation)
// // // // ----------------------------------------------------------------
// // // template <int BM, int BN, int BK, int STAGES, int THREADS, bool IsAligned>
// // // static void launch_sgemm_addmm_sm89(
// // //     cudaStream_t stream, int M, int N, int K,
// // //     float alpha,
// // //     const float* A, int lda, long long strideA,
// // //     const float* B, int ldb, long long strideB,
// // //     float beta,
// // //     const float* bias, int64_t bias_numel,
// // //     float* C, int ldc, long long strideC,
// // //     int batchCount)
// // // {
// // //     using Config = SgemmTileConfig<BM, BN, BK, STAGES, THREADS>;
// // //     static const size_t smem_bytes = STAGES * Config::STAGE_SIZE * sizeof(float);

// // //     static std::unordered_map<const void*, bool> done;
// // //     const void* fn = (const void*)sgemm_addmm_sm89_kernel<BM, BN, BK, STAGES, THREADS, IsAligned>;
// // //     if (!done[fn]) {
// // //         cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
// // //         done[fn] = true;
// // //     }

// // //     const int gx = (N + BN - 1) / BN, gy = (M + BM - 1) / BM;
// // //     sgemm_addmm_sm89_kernel<BM, BN, BK, STAGES, THREADS, IsAligned>
// // //         <<<dim3(gx, gy, batchCount), THREADS, smem_bytes, stream>>>(
// // //             M, N, K, alpha, A, lda, strideA, B, ldb, strideB,
// // //             beta, bias, bias_numel, C, ldc, strideC, batchCount);
// // // }

// // // // ----------------------------------------------------------------
// // // // Dispatch: picks alignment variant, no splitK (addmm is single-pass)
// // // // ----------------------------------------------------------------
// // // template <int BM, int BN, int BK, int STAGES, int THREADS>
// // // static void dispatch_sgemm_addmm_sm89(
// // //     cudaStream_t stream, int M, int N, int K,
// // //     float alpha,
// // //     const float* A, int lda, long long strideA,
// // //     const float* B, int ldb, long long strideB,
// // //     float beta,
// // //     const float* bias, int64_t bias_numel,
// // //     float* C, int ldc, long long strideC,
// // //     int batchCount)
// // // {
// // //     const bool aligned = (((size_t)A & 15) == 0) && (((size_t)B & 15) == 0)
// // //                       && ((lda & 3) == 0) && ((ldb & 3) == 0);
// // //     if (aligned)
// // //         launch_sgemm_addmm_sm89<BM, BN, BK, STAGES, THREADS, true>(
// // //             stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB,
// // //             beta, bias, bias_numel, C, ldc, strideC, batchCount);
// // //     else
// // //         launch_sgemm_addmm_sm89<BM, BN, BK, STAGES, THREADS, false>(
// // //             stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB,
// // //             beta, bias, bias_numel, C, ldc, strideC, batchCount);
// // // }

// // // // ================================================================
// // // // Public entry point
// // // //
// // // // Tile heuristic for SM89 Ada Lovelace:
// // // //
// // // //   M >= 256, N >= 128    →  256×128×32, 2-stage  (max tiles + wide BK)
// // // //                              smem: 2 × (256×32 + 32×128) × 4 = 98.3 KB
// // // //
// // // //   M,N >= 128, K >= 256  →  128×128×32, 4-stage  (wide BK, better efficiency)
// // // //                              smem: 4 × (128×32 + 32×128) × 4 = 131 KB (Requires SM89)
// // // //                              Wait, SM89 (Ada) supports up to 100KB per block by default.
// // // //                              128x128x32 with 3 stages is 96KB.
// // // //
// // // //   M,N >= 128, K <  256  →  128×128×16, 6-stage  (max stages, best latency hiding)
// // // //                              smem: 6 × (128×16 + 16×128) × 4 = 98.3 KB
// // // //
// // // //   M < 128 or N < 128    →  64×64×16, 6-stage    (higher occupancy for small tiles)
// // // //                              smem: 6 × (64×16 + 16×64) × 4 = 49.2 KB
// // // // ================================================================
// // // extern "C" void mycublasSgemmAddmm_sm89(
// // //     mycublasHandle_t handle,
// // //     int M, int N, int K,
// // //     const float alpha,
// // //     const float* A, int lda, long long int strideA,
// // //     const float* B, int ldb, long long int strideB,
// // //     const float beta,
// // //     const float* bias, int64_t bias_numel,
// // //     float* C, int ldc, long long int strideC,
// // //     int batchCount)
// // // {
// // //     cudaStream_t stream = handle ? handle->stream : 0;

// // //     if (M >= 256 && N >= 128) {
// // //         // Large tiles: best for large GEMMs on high-SM count Ada GPUs
// // //         dispatch_sgemm_addmm_sm89<256, 128, 32, 2, 256>(
// // //             stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB,
// // //             beta, bias, bias_numel, C, ldc, strideC, batchCount);
// // //     } else if (M >= 128 && N >= 128) {
// // //         if (K >= 256) {
// // //             // Wide BK, 3 stages to stay under 100KB smem
// // //             dispatch_sgemm_addmm_sm89<128, 128, 32, 3, 128>(
// // //                 stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB,
// // //                 beta, bias, bias_numel, C, ldc, strideC, batchCount);
// // //         } else {
// // //             // Narrow BK, 6 stages for max pipeline depth
// // //             dispatch_sgemm_addmm_sm89<128, 128, 16, 6, 128>(
// // //                 stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB,
// // //                 beta, bias, bias_numel, C, ldc, strideC, batchCount);
// // //         }
// // //     } else {
// // //         // Small tiles: 64×64 doubles occupancy vs 128×128
// // //         dispatch_sgemm_addmm_sm89<64, 64, 16, 6, 64>(
// // //             stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB,
// // //             beta, bias, bias_numel, C, ldc, strideC, batchCount);
// // //     }
// // // }


























// // // #include "mycublas.h"
// // // #include <cuda_runtime.h>
// // // #include <stdint.h>
// // // #include <unordered_map>
// // // #include "../SM86/Sgemm_core_template.cuh"

// // // // ============================================================
// // // // Sgemm Addmm SM89 — Fused Matmul + Bias for Ada Lovelace
// // // //
// // // // Operation: C = alpha * (A * B) + beta * bias
// // // // Layout: NN  (A:[M,K] row-major, B:[K,N] row-major)
// // // //
// // // // SM89 vs v34 (SM86):
// // // //   - 5-6 pipeline stages (vs 3) for deeper latency hiding
// // // //   - 16-wide CTA swizzling for Ada Lovelace L2 locality
// // // //   - Multiple tile configs parameterized at compile time
// // // //   - BK=16 (6-stage) and BK=32 (3-stage) variants
// // // //   - ldmatrix.x4 for A-fragment loads (NN A is M-major in SMEM,
// // // //     which is the native-friendly layout for TF32-via-ldmatrix).
// // // //     B stays on manual scalar loads: NN B is K-major in SMEM and
// // // //     the ldmatrix register distribution for that layout is
// // // //     transposed relative to what MMA m16n8k8 expects (and the
// // // //     .trans variant corrupts TF32 values since the b16 halves of
// // // //     a TF32 word are adjacent along the col axis, not row axis).
// // // // ============================================================

// // // #ifndef MMA_TF32
// // // #define MMA_TF32(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1)                \
// // //     asm volatile(                                                   \
// // //         "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "     \
// // //         "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"       \
// // //         : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)                     \
// // //         : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1))
// // // #endif

// // // #ifndef LDSM_X4
// // // #define LDSM_X4(r0,r1,r2,r3,addr)                                \
// // //     asm volatile(                                                   \
// // //         "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "              \
// // //         "{%0,%1,%2,%3},[%4];"                                     \
// // //         : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(addr))
// // // #endif

// // // // bias_numel: 1  → scalar broadcast
// // // //             N  → 1D row vector (common case: linear layer bias)
// // // //             0 / nullptr → no bias; beta applied to existing C
// // // template <int BM, int BN, int BK, int STAGES, int THREADS, bool IsAligned>
// // // __global__ void __launch_bounds__(THREADS, 1)
// // // sgemm_addmm_sm89_kernel(
// // //     int M, int N, int K,
// // //     float alpha,
// // //     const float* __restrict__ A, int lda, long long strideA,
// // //     const float* __restrict__ B, int ldb, long long strideB,
// // //     float beta,
// // //     const float* __restrict__ bias, int64_t bias_numel,
// // //     float* __restrict__ C, int ldc, long long strideC,
// // //     int batchCount)
// // // {
// // //     using Config = SgemmTileConfig<BM, BN, BK, STAGES, THREADS>;

// // //     const int batch = blockIdx.z;
// // //     if (batch >= batchCount) return;

// // //     // Robust block mapping for non-square grids (8-wide strips)
// // //     const int sw = 8;
// // //     const int grid_x = (N + BN - 1) / BN, grid_y = (M + BM - 1) / BM;
// // //     const int block_idx = blockIdx.y * grid_x + blockIdx.x;
// // //     const int num_blocks_per_strip = sw * grid_y;
// // //     const int strip_idx = block_idx / num_blocks_per_strip;
// // //     const int strip_off = block_idx % num_blocks_per_strip;
// // //     const int actual_sw = min(sw, grid_x - strip_idx * sw);
// // //     const int bx = strip_idx * sw + (strip_off % actual_sw);
// // //     const int by = strip_off / actual_sw;

// // //     if (bx >= grid_x || by >= grid_y) return;

// // //     const int tid  = threadIdx.x;
// // //     const int lane = tid & 31, wid = tid >> 5;
// // //     const int wy   = wid / Config::WARPS_N;
// // //     const int wx   = wid % Config::WARPS_N;

// // //     extern __shared__ float smem[];

// // //     float acc[Config::MMA_M][Config::MMA_N][4];
// // //     #pragma unroll
// // //     for (int i = 0; i < Config::MMA_M; i++)
// // //         #pragma unroll
// // //         for (int j = 0; j < Config::MMA_N; j++)
// // //             acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

// // //     // ----------------------------------------------------------------
// // //     // Global pointer induction
// // //     // A: [M, K] — NT load (BM rows × BK cols, vectorised along K)
// // //     // B: [K, N] — TN load (BK rows × BN cols, vectorised along N)
// // //     // ----------------------------------------------------------------
// // //     const float* gA_ptr[Config::NT_LOAD_ITERS_A];
// // //     const float* gB_ptr[Config::TN_LOAD_ITERS_B];

// // //     #pragma unroll
// // //     for (int i = 0; i < Config::NT_LOAD_ITERS_A; i++) {
// // //         const int r = (tid / Config::NT_THREADS_PER_ROW_A) + i * Config::NT_ROWS_PER_ITER_A;
// // //         const int c = (tid % Config::NT_THREADS_PER_ROW_A) * 4;
// // //         gA_ptr[i] = A + (long long)batch * strideA
// // //                       + (long long)(by * BM + r) * lda + c;
// // //     }
// // //     #pragma unroll
// // //     for (int i = 0; i < Config::TN_LOAD_ITERS_B; i++) {
// // //         const int r = (tid / Config::TN_THREADS_PER_ROW_B) + i * Config::TN_ROWS_PER_ITER_B;
// // //         const int c = (tid % Config::TN_THREADS_PER_ROW_B) * 4;
// // //         gB_ptr[i] = B + (long long)batch * strideB
// // //                       + (long long)r * ldb + (bx * BN + c);
// // //     }

// // //     // ----------------------------------------------------------------
// // //     // Async stage loader
// // //     // ----------------------------------------------------------------
// // //     uint32_t sm_a_off[Config::NT_LOAD_ITERS_A];
// // //     #pragma unroll
// // //     for (int i = 0; i < Config::NT_LOAD_ITERS_A; i++) {
// // //         const int r = (tid / Config::NT_THREADS_PER_ROW_A) + i * Config::NT_ROWS_PER_ITER_A;
// // //         const int c = (tid % Config::NT_THREADS_PER_ROW_A) * 4;
// // //         const int sc = c ^ ((r & 3) << 2);
// // //         sm_a_off[i] = r * Config::BK + sc;
// // //     }

// // //     uint32_t sm_b_off[Config::TN_LOAD_ITERS_B];
// // //     #pragma unroll
// // //     for (int i = 0; i < Config::TN_LOAD_ITERS_B; i++) {
// // //         const int r = (tid / Config::TN_THREADS_PER_ROW_B) + i * Config::TN_ROWS_PER_ITER_B;
// // //         const int c = (tid % Config::TN_THREADS_PER_ROW_B) * 4;
// // //         const int sc = c ^ ((r & 7) << 2);
// // //         sm_b_off[i] = r * Config::BN + sc;
// // //     }

// // //     auto load_to_stage = [&](int stage, int ko) {
// // //         float* As = smem + stage * Config::STAGE_SIZE;
// // //         float* Bs = As   + Config::AS_SIZE;
// // //         #pragma unroll
// // //         for (int i = 0; i < Config::NT_LOAD_ITERS_A; i++) {
// // //             const int r = (tid / Config::NT_THREADS_PER_ROW_A) + i * Config::NT_ROWS_PER_ITER_A, c = (tid % Config::NT_THREADS_PER_ROW_A) * 4;
// // //             const int gr = by * BM + r, gc = ko + c;
// // //             uint32_t sm = __cvta_generic_to_shared(As + sm_a_off[i]);
// // //             if (gr < M && r < BM) {
// // //                 int bytes = max(0, min(16, (K - gc) * 4));
// // //                 asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm), "l"(gA_ptr[i]), "r"(bytes));
// // //             } else if (r < BM) { *(float4*)(As + sm_a_off[i]) = {0,0,0,0}; }
// // //             gA_ptr[i] += BK;
// // //         }
// // //         #pragma unroll
// // //         for (int i = 0; i < Config::TN_LOAD_ITERS_B; i++) {
// // //             const int r = (tid / Config::TN_THREADS_PER_ROW_B) + i * Config::TN_ROWS_PER_ITER_B, c = (tid % Config::TN_THREADS_PER_ROW_B) * 4;
// // //             const int gk = ko + r, gn = bx * BN + c;
// // //             uint32_t sm = __cvta_generic_to_shared(Bs + sm_b_off[i]);
// // //             if (gk < K && r < BK) {
// // //                 int bytes = max(0, min(16, (N - gn) * 4));
// // //                 asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm), "l"(gB_ptr[i]), "r"(bytes));
// // //             } else if (r < BK) { *(float4*)(Bs + sm_b_off[i]) = {0,0,0,0}; }
// // //             gB_ptr[i] += (long long)BK * ldb;
// // //         }
// // //     };

// // //     // ----------------------------------------------------------------
// // //     // Register fragment helpers
// // //     // ----------------------------------------------------------------
// // //     const int g_sh = lane / 4, t_sh = lane % 4;

// // //     // ----------------------------------------------------------------
// // //     // A-fragment load via ldmatrix.x4
// // //     //
// // //     // A is M-major in SMEM (rows=M, cols=K, row stride=BK). One call
// // //     // loads a 16 M × 8 TF32-K region covering the full A fragment for
// // //     // one m16n8k8 MMA.
// // //     //
// // //     // Per-lane row pointer:
// // //     //   mat        = lane >> 3          (0..3)
// // //     //   row_in_mat = lane & 7           (0..7)
// // //     //   m_off      = warp_M_base + (mat >> 1 ? 8 : 0) + row_in_mat
// // //     //   k_raw      = ks + (mat & 1 ? 4 : 0)            [TF32 elems]
// // //     //   k_sw       = k_raw XOR ((m_off & 3) << 2)      [matches store swizzle]
// // //     //   addr       = &As[m_off * BK + k_sw]
// // //     //
// // //     // Output (TF32 per thread):
// // //     //   d0: (M = warp_M_base + T/4,      K = ks + T%4)
// // //     //   d1: (M = warp_M_base + T/4,      K = ks + T%4 + 4)
// // //     //   d2: (M = warp_M_base + T/4 + 8,  K = ks + T%4)
// // //     //   d3: (M = warp_M_base + T/4 + 8,  K = ks + T%4 + 4)
// // //     //
// // //     // MMA A fragment wants:
// // //     //   a0: (T/4,   T%4),    a1: (T/4+8, T%4)
// // //     //   a2: (T/4,   T%4+4),  a3: (T/4+8, T%4+4)
// // //     //
// // //     // → remap reg[0..3] = { d0, d2, d1, d3 } (swap d1/d2).
// // //     //
// // //     // TF32 correctness: each TF32 word's two b16 halves are adjacent
// // //     // along K (the col direction in M-major SMEM). Non-trans ldmatrix
// // //     // packs two b16 along the col direction into one output register
// // //     // per thread — so each register is a complete TF32 value.
// // //     // ----------------------------------------------------------------
// // //     auto load_frA = [&](uint32_t reg[4], int ks, int mi, int st) {
// // //         float* As = smem + st * Config::STAGE_SIZE;
// // //         const int warp_m_base = wy * Config::WARP_TILE_M + mi * 16;
// // //         const int mat         = lane >> 3;            // 0..3
// // //         const int row_in_mat  = lane & 7;             // 0..7
// // //         const int m_off       = warp_m_base + ((mat >> 1) ? 8 : 0) + row_in_mat;
// // //         const int k_raw       = ks + ((mat & 1) ? 4 : 0);
// // //         const int k_sw        = k_raw ^ ((m_off & 3) << 2);
// // //         const uint32_t addr   = __cvta_generic_to_shared(
// // //                                     &As[m_off * Config::BK + k_sw]);
// // //         uint32_t d0, d1, d2, d3;
// // //         LDSM_X4(d0, d1, d2, d3, addr);
// // //         reg[0] = d0;  // a0
// // //         reg[1] = d2;  // a1
// // //         reg[2] = d1;  // a2
// // //         reg[3] = d3;  // a3
// // //     };

// // //     // ----------------------------------------------------------------
// // //     // B-fragment load (manual scalar loads)
// // //     //
// // //     // B is K-major in SMEM (rows=K, cols=N, row stride=BN). The MMA B
// // //     // fragment wants per-thread distribution (k=T%4, n=T/4), but any
// // //     // ldmatrix variant on K-major TF32 storage yields either
// // //     // (k=T/4, n=T%4) — transposed wrt MMA — or (with .trans) corrupted
// // //     // TF32 values (halves come from adjacent K rows rather than
// // //     // adjacent N cols within a single TF32 word). So we stick with
// // //     // the 2 scalar LDS per fragment.
// // //     // ----------------------------------------------------------------
// // //     auto load_frB = [&](uint32_t reg[2], int ks, int ni, int st) {
// // //         float* Bs = smem + st * Config::STAGE_SIZE + Config::AS_SIZE;
// // //         const int lr0 = ks + (lane % 4), lr4 = lr0 + 4, lc = wx * Config::WARP_TILE_N + ni * 8 + (lane / 4);
// // //         auto gb = [&](int r, int c) { return *(const uint32_t*)(&Bs[r * Config::BN + (c ^ ((r & 7) << 2))]); };
// // //         reg[0] = gb(lr0, lc); reg[1] = gb(lr4, lc);
// // //     };

// // //     // ----------------------------------------------------------------
// // //     // Multi-stage pipeline warmup  (STAGES-1 stages pre-filled)
// // //     // ----------------------------------------------------------------
// // //     load_to_stage(0, 0);
// // //     asm volatile("cp.async.commit_group;\n");
// // //     #pragma unroll
// // //     for (int s = 1; s < Config::STAGES - 1; s++) {
// // //         if (s * Config::BK < K) load_to_stage(s, s * Config::BK);
// // //         asm volatile("cp.async.commit_group;\n");
// // //     }

// // //     int ws = Config::STAGES - 1, rs = 0;
// // //     uint32_t frA[2][Config::MMA_M][4], frB[2][Config::MMA_N][2];

// // //     asm volatile("cp.async.wait_group %0;\n" :: "n"(Config::STAGES - 2));
// // //     __syncthreads();
// // //     #pragma unroll
// // //     for (int i = 0; i < Config::MMA_M; i++) load_frA(frA[0][i], 0, i, rs);
// // //     #pragma unroll
// // //     for (int j = 0; j < Config::MMA_N; j++) load_frB(frB[0][j], 0, j, rs);

// // //     // ----------------------------------------------------------------
// // //     // Main K loop
// // //     // ----------------------------------------------------------------
// // //     for (int k = 0; k < K; k += Config::BK) {
// // //         if (k + (Config::STAGES - 1) * Config::BK < K)
// // //             load_to_stage(ws, k + (Config::STAGES - 1) * Config::BK);
// // //         asm volatile("cp.async.commit_group;\n");

// // //         #pragma unroll
// // //         for (int ks = 0; ks < Config::BK; ks += 16) {
// // //             // First 8-step MMA (ks)
// // //             #pragma unroll
// // //             for (int i = 0; i < Config::MMA_M; i++) {
// // //                 #pragma unroll
// // //                 for (int j = 0; j < Config::MMA_N; j++) {
// // //                     MMA_TF32(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3],
// // //                              frA[0][i][0], frA[0][i][1], frA[0][i][2], frA[0][i][3],
// // //                              frB[0][j][0], frB[0][j][1]);
// // //                     if (i == 0) load_frB(frB[1][j], ks + 8, j, rs);
// // //                 }
// // //                 load_frA(frA[1][i], ks + 8, i, rs);
// // //             }

// // //             // Second 8-step MMA (ks + 8)
// // //             #pragma unroll
// // //             for (int i = 0; i < Config::MMA_M; i++) {
// // //                 #pragma unroll
// // //                 for (int j = 0; j < Config::MMA_N; j++) {
// // //                     MMA_TF32(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3],
// // //                              frA[1][i][0], frA[1][i][1], frA[1][i][2], frA[1][i][3],
// // //                              frB[1][j][0], frB[1][j][1]);
// // //                 }
// // //             }

// // //             // Prepare for next 16-element block or next stage
// // //             if (ks + 16 < Config::BK) {
// // //                 #pragma unroll
// // //                 for (int i = 0; i < Config::MMA_M; i++) {
// // //                     #pragma unroll
// // //                     for (int j = 0; j < Config::MMA_N; j++) if (i == 0) load_frB(frB[0][j], ks + 16, j, rs);
// // //                     load_frA(frA[0][i], ks + 16, i, rs);
// // //                 }
// // //             } else if (k + Config::BK < K) {
// // //                 asm volatile("cp.async.wait_group %0;\n" :: "n"(Config::STAGES - 2));
// // //                 __syncthreads();
// // //                 rs = (rs + 1) % Config::STAGES;
// // //                 ws = (ws + 1) % Config::STAGES;
// // //                 #pragma unroll
// // //                 for (int i = 0; i < Config::MMA_M; i++) {
// // //                     #pragma unroll
// // //                     for (int j = 0; j < Config::MMA_N; j++) if (i == 0) load_frB(frB[0][j], 0, j, rs);
// // //                     load_frA(frA[0][i], 0, i, rs);
// // //                 }
// // //             }
// // //         }
// // //     }

// // //     // ----------------------------------------------------------------
// // //     // Epilogue: C[r,c] = alpha * acc + beta * bias[c]
// // //     // MMA output: lane l owns rows {l/4, l/4+8}, cols {2*(l%4), 2*(l%4)+1}
// // //     // ----------------------------------------------------------------
// // //     const int g_epi = lane / 4, t_epi = lane % 4;
// // //     float* dC = C + (long long)batch * strideC;

// // //     #pragma unroll
// // //     for (int i = 0; i < Config::MMA_M; i++) {
// // //         #pragma unroll
// // //         for (int j = 0; j < Config::MMA_N; j++) {
// // //             const int r0 = by * BM + wy * Config::WARP_TILE_M + i * 16 + g_epi;
// // //             const int r8 = r0 + 8;
// // //             const int c0 = bx * BN + wx * Config::WARP_TILE_N + j * 8 + t_epi * 2;
// // //             const int c1 = c0 + 1;

// // //             // Load bias values (Scalar, Vector, or Matrix broadcast)
// // //             float b0 = 0.f, b1 = 0.f;
// // //             if (bias) {
// // //                 if (bias_numel == 1) {
// // //                     b0 = b1 = bias[0];
// // //                 } else if (bias_numel == (int64_t)N) {
// // //                     if (c0 < N) b0 = bias[c0];
// // //                     if (c1 < N) b1 = bias[c1];
// // //                 } else if (bias_numel == (int64_t)M * N) {
// // //                     const float* pb = bias + (long long)(batch % M) * N + c0;
// // //                     if (c0 < N) b0 = pb[0];
// // //                     if (c1 < N) b1 = pb[1];
// // //                 }
// // //             }

// // //             auto store = [&](int r, int c, float f, float b) __attribute__((always_inline)) {
// // //                 if (r >= M || c >= N) return;
// // //                 float* dst = &dC[(long long)r * ldc + c];
// // //                 if (bias) {
// // //                     *dst = alpha * f + beta * b;
// // //                 } else {
// // //                     *dst = alpha * f + (beta == 0.f ? 0.f : beta * (*dst));
// // //                 }
// // //             };

// // //             store(r0, c0, acc[i][j][0], b0);
// // //             store(r0, c1, acc[i][j][1], b1);
// // //             store(r8, c0, acc[i][j][2], b0);
// // //             store(r8, c1, acc[i][j][3], b1);
// // //         }
// // //     }
// // // }

// // // // ----------------------------------------------------------------
// // // // Launch helper (aligned/unaligned specialisation)
// // // // ----------------------------------------------------------------
// // // template <int BM, int BN, int BK, int STAGES, int THREADS, bool IsAligned>
// // // static void launch_sgemm_addmm_sm89(
// // //     cudaStream_t stream, int M, int N, int K,
// // //     float alpha,
// // //     const float* A, int lda, long long strideA,
// // //     const float* B, int ldb, long long strideB,
// // //     float beta,
// // //     const float* bias, int64_t bias_numel,
// // //     float* C, int ldc, long long strideC,
// // //     int batchCount)
// // // {
// // //     using Config = SgemmTileConfig<BM, BN, BK, STAGES, THREADS>;
// // //     static const size_t smem_bytes = STAGES * Config::STAGE_SIZE * sizeof(float);

// // //     static std::unordered_map<const void*, bool> done;
// // //     const void* fn = (const void*)sgemm_addmm_sm89_kernel<BM, BN, BK, STAGES, THREADS, IsAligned>;
// // //     if (!done[fn]) {
// // //         cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
// // //         done[fn] = true;
// // //     }

// // //     const int gx = (N + BN - 1) / BN, gy = (M + BM - 1) / BM;
// // //     sgemm_addmm_sm89_kernel<BM, BN, BK, STAGES, THREADS, IsAligned>
// // //         <<<dim3(gx, gy, batchCount), THREADS, smem_bytes, stream>>>(
// // //             M, N, K, alpha, A, lda, strideA, B, ldb, strideB,
// // //             beta, bias, bias_numel, C, ldc, strideC, batchCount);
// // // }

// // // // ----------------------------------------------------------------
// // // // Dispatch: picks alignment variant, no splitK (addmm is single-pass)
// // // // ----------------------------------------------------------------
// // // template <int BM, int BN, int BK, int STAGES, int THREADS>
// // // static void dispatch_sgemm_addmm_sm89(
// // //     cudaStream_t stream, int M, int N, int K,
// // //     float alpha,
// // //     const float* A, int lda, long long strideA,
// // //     const float* B, int ldb, long long strideB,
// // //     float beta,
// // //     const float* bias, int64_t bias_numel,
// // //     float* C, int ldc, long long strideC,
// // //     int batchCount)
// // // {
// // //     const bool aligned = (((size_t)A & 15) == 0) && (((size_t)B & 15) == 0)
// // //                       && ((lda & 3) == 0) && ((ldb & 3) == 0);
// // //     if (aligned)
// // //         launch_sgemm_addmm_sm89<BM, BN, BK, STAGES, THREADS, true>(
// // //             stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB,
// // //             beta, bias, bias_numel, C, ldc, strideC, batchCount);
// // //     else
// // //         launch_sgemm_addmm_sm89<BM, BN, BK, STAGES, THREADS, false>(
// // //             stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB,
// // //             beta, bias, bias_numel, C, ldc, strideC, batchCount);
// // // }

// // // // ================================================================
// // // // Public entry point
// // // //
// // // // Tile heuristic for SM89 Ada Lovelace:
// // // //
// // // //   M >= 256, N >= 128    →  256×128×32, 2-stage  (max tiles + wide BK)
// // // //                              smem: 2 × (256×32 + 32×128) × 4 = 98.3 KB
// // // //
// // // //   M,N >= 128, K >= 256  →  128×128×32, 4-stage  (wide BK, better efficiency)
// // // //                              smem: 4 × (128×32 + 32×128) × 4 = 131 KB (Requires SM89)
// // // //                              Wait, SM89 (Ada) supports up to 100KB per block by default.
// // // //                              128x128x32 with 3 stages is 96KB.
// // // //
// // // //   M,N >= 128, K <  256  →  128×128×16, 6-stage  (max stages, best latency hiding)
// // // //                              smem: 6 × (128×16 + 16×128) × 4 = 98.3 KB
// // // //
// // // //   M < 128 or N < 128    →  64×64×16, 6-stage    (higher occupancy for small tiles)
// // // //                              smem: 6 × (64×16 + 16×64) × 4 = 49.2 KB
// // // // ================================================================
// // // extern "C" void mycublasSgemmAddmm_sm89(
// // //     mycublasHandle_t handle,
// // //     int M, int N, int K,
// // //     const float alpha,
// // //     const float* A, int lda, long long int strideA,
// // //     const float* B, int ldb, long long int strideB,
// // //     const float beta,
// // //     const float* bias, int64_t bias_numel,
// // //     float* C, int ldc, long long int strideC,
// // //     int batchCount)
// // // {
// // //     cudaStream_t stream = handle ? handle->stream : 0;

// // //     if (M >= 256 && N >= 128) {
// // //         // Large tiles: best for large GEMMs on high-SM count Ada GPUs
// // //         dispatch_sgemm_addmm_sm89<256, 128, 32, 2, 256>(
// // //             stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB,
// // //             beta, bias, bias_numel, C, ldc, strideC, batchCount);
// // //     } else if (M >= 128 && N >= 128) {
// // //         if (K >= 256) {
// // //             // Wide BK, 3 stages to stay under 100KB smem
// // //             dispatch_sgemm_addmm_sm89<128, 128, 32, 3, 128>(
// // //                 stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB,
// // //                 beta, bias, bias_numel, C, ldc, strideC, batchCount);
// // //         } else {
// // //             // Narrow BK, 6 stages for max pipeline depth
// // //             dispatch_sgemm_addmm_sm89<128, 128, 16, 6, 128>(
// // //                 stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB,
// // //                 beta, bias, bias_numel, C, ldc, strideC, batchCount);
// // //         }
// // //     } else {
// // //         // Small tiles: 64×64 doubles occupancy vs 128×128
// // //         dispatch_sgemm_addmm_sm89<64, 64, 16, 6, 64>(
// // //             stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB,
// // //             beta, bias, bias_numel, C, ldc, strideC, batchCount);
// // //     }
// // // }



























// // #include "mycublas.h"
// // #include <cuda_runtime.h>
// // #include <stdint.h>
// // #include "../SM86/Sgemm_core_template.cuh"

// // // ============================================================
// // // Sgemm Addmm SM89 — Fused Matmul + Bias for Ada Lovelace
// // //
// // // Operation: C = alpha * (A * B) + beta * bias
// // // Layout: NN  (A:[M,K] row-major, B:[K,N] row-major)
// // //
// // // Optimizations (full sweep):
// // //   - ldmatrix.x4 for A (M-major in SMEM: natural for TF32-via-ldmatrix)
// // //   - Lane-invariant ldmatrix address arithmetic hoisted out of the fetch
// // //     lambda (swizzle XOR folded into a lane-const since m*16 / warp-base
// // //     don't perturb the low 2 bits of M).
// // //   - Bias mode (scalar / vector / matrix) dispatched once outside the
// // //     epilogue m,n loops; bias matrix row pointer resolved once.
// // //   - __launch_bounds__ occupancy uses the same heuristic as core template
// // //     (MAX_OCC=2 for tiles <256x128).
// // //   - Dropped the std::unordered_map guard on cudaFuncSetAttribute (which was
// // //     thread-unsafe); the call is cheap and idempotent.
// // //   - B stays on manual scalar loads — NN B is K-major in SMEM and no
// // //     ldmatrix variant delivers the MMA B-fragment thread distribution for
// // //     that storage orientation without corrupting TF32 values.
// // // ============================================================

// // #ifndef MMA_TF32
// // #define MMA_TF32(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1)                \
// //     asm volatile(                                                   \
// //         "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "     \
// //         "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"       \
// //         : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)                     \
// //         : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1))
// // #endif

// // #ifndef LDSM_X4
// // #define LDSM_X4(r0,r1,r2,r3,addr)                                \
// //     asm volatile(                                                   \
// //         "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "              \
// //         "{%0,%1,%2,%3},[%4];"                                     \
// //         : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(addr))
// // #endif

// // // Occupancy heuristic mirroring SgemmTileConfigSM89::MAX_OCC
// // template <int BM, int BN> struct AddmmOcc { static constexpr int V = (BM * BN >= 256 * 128) ? 1 : 2; };

// // // bias_numel: 1   → scalar broadcast
// // //             N   → 1D row vector (common case: linear layer bias)
// // //             M*N → matrix bias, indexed as batch-major (batch % M)
// // template <int BM, int BN, int BK, int STAGES, int THREADS, bool IsAligned>
// // __global__ void __launch_bounds__(THREADS, AddmmOcc<BM,BN>::V)
// // sgemm_addmm_sm89_kernel(
// //     int M, int N, int K,
// //     float alpha,
// //     const float* __restrict__ A, int lda, long long strideA,
// //     const float* __restrict__ B, int ldb, long long strideB,
// //     float beta,
// //     const float* __restrict__ bias, int64_t bias_numel,
// //     float* __restrict__ C, int ldc, long long strideC,
// //     int batchCount)
// // {
// //     using Config = SgemmTileConfig<BM, BN, BK, STAGES, THREADS>;

// //     const int batch = blockIdx.z;
// //     if (batch >= batchCount) return;

// //     // CTA swizzle (8-wide strips) guarded against short-last-strip edge cases.
// //     const int sw = 8;
// //     const int grid_x = (N + BN - 1) / BN, grid_y = (M + BM - 1) / BM;
// //     const int block_idx = blockIdx.y * grid_x + blockIdx.x;
// //     const int num_blocks_per_strip = sw * grid_y;
// //     const int strip_idx = block_idx / num_blocks_per_strip;
// //     const int strip_off = block_idx % num_blocks_per_strip;
// //     const int strip_base = strip_idx * sw;
// //     if (strip_base >= grid_x) return;
// //     const int actual_sw = min(sw, grid_x - strip_base);
// //     const int bx = strip_base + (strip_off % actual_sw);
// //     const int by = strip_off / actual_sw;

// //     if (bx >= grid_x || by >= grid_y) return;

// //     const int tid  = threadIdx.x;
// //     const int lane = tid & 31, wid = tid >> 5;
// //     const int wy   = wid / Config::WARPS_N;
// //     const int wx   = wid % Config::WARPS_N;

// //     extern __shared__ float smem[];

// //     float acc[Config::MMA_M][Config::MMA_N][4];
// //     #pragma unroll
// //     for (int i = 0; i < Config::MMA_M; i++)
// //         #pragma unroll
// //         for (int j = 0; j < Config::MMA_N; j++) {
// //             acc[i][j][0] = 0.f; acc[i][j][1] = 0.f;
// //             acc[i][j][2] = 0.f; acc[i][j][3] = 0.f;
// //         }

// //     // ----------------------------------------------------------------
// //     // Global pointer induction
// //     // A: [M, K] — NT load (BM rows × BK cols, vectorised along K)
// //     // B: [K, N] — TN load (BK rows × BN cols, vectorised along N)
// //     // ----------------------------------------------------------------
// //     const float* gA_ptr[Config::NT_LOAD_ITERS_A];
// //     const float* gB_ptr[Config::TN_LOAD_ITERS_B];

// //     #pragma unroll
// //     for (int i = 0; i < Config::NT_LOAD_ITERS_A; i++) {
// //         const int r = (tid / Config::NT_THREADS_PER_ROW_A) + i * Config::NT_ROWS_PER_ITER_A;
// //         const int c = (tid % Config::NT_THREADS_PER_ROW_A) * 4;
// //         gA_ptr[i] = A + (long long)batch * strideA
// //                       + (long long)(by * BM + r) * lda + c;
// //     }
// //     #pragma unroll
// //     for (int i = 0; i < Config::TN_LOAD_ITERS_B; i++) {
// //         const int r = (tid / Config::TN_THREADS_PER_ROW_B) + i * Config::TN_ROWS_PER_ITER_B;
// //         const int c = (tid % Config::TN_THREADS_PER_ROW_B) * 4;
// //         gB_ptr[i] = B + (long long)batch * strideB
// //                       + (long long)r * ldb + (bx * BN + c);
// //     }

// //     // ----------------------------------------------------------------
// //     // SMEM offsets (computed once per thread)
// //     // ----------------------------------------------------------------
// //     uint32_t sm_a_off[Config::NT_LOAD_ITERS_A];
// //     #pragma unroll
// //     for (int i = 0; i < Config::NT_LOAD_ITERS_A; i++) {
// //         const int r = (tid / Config::NT_THREADS_PER_ROW_A) + i * Config::NT_ROWS_PER_ITER_A;
// //         const int c = (tid % Config::NT_THREADS_PER_ROW_A) * 4;
// //         const int sc = c ^ ((r & 3) << 2);
// //         sm_a_off[i] = r * Config::BK + sc;
// //     }

// //     uint32_t sm_b_off[Config::TN_LOAD_ITERS_B];
// //     #pragma unroll
// //     for (int i = 0; i < Config::TN_LOAD_ITERS_B; i++) {
// //         const int r = (tid / Config::TN_THREADS_PER_ROW_B) + i * Config::TN_ROWS_PER_ITER_B;
// //         const int c = (tid % Config::TN_THREADS_PER_ROW_B) * 4;
// //         const int sc = c ^ ((r & 7) << 2);
// //         sm_b_off[i] = r * Config::BN + sc;
// //     }

// //     auto load_to_stage = [&](int stage, int ko) {
// //         float* As = smem + stage * Config::STAGE_SIZE;
// //         float* Bs = As   + Config::AS_SIZE;
// //         #pragma unroll
// //         for (int i = 0; i < Config::NT_LOAD_ITERS_A; i++) {
// //             const int r = (tid / Config::NT_THREADS_PER_ROW_A) + i * Config::NT_ROWS_PER_ITER_A;
// //             const int c = (tid % Config::NT_THREADS_PER_ROW_A) * 4;
// //             const int gr = by * BM + r, gc = ko + c;
// //             uint32_t sm = __cvta_generic_to_shared(As + sm_a_off[i]);
// //             if (gr < M && r < BM) {
// //                 int bytes = max(0, min(16, (K - gc) * 4));
// //                 asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm), "l"(gA_ptr[i]), "r"(bytes));
// //             } else if (r < BM) { *(float4*)(As + sm_a_off[i]) = {0,0,0,0}; }
// //             gA_ptr[i] += BK;
// //         }
// //         #pragma unroll
// //         for (int i = 0; i < Config::TN_LOAD_ITERS_B; i++) {
// //             const int r = (tid / Config::TN_THREADS_PER_ROW_B) + i * Config::TN_ROWS_PER_ITER_B;
// //             const int c = (tid % Config::TN_THREADS_PER_ROW_B) * 4;
// //             const int gk = ko + r, gn = bx * BN + c;
// //             uint32_t sm = __cvta_generic_to_shared(Bs + sm_b_off[i]);
// //             if (gk < K && r < BK) {
// //                 int bytes = max(0, min(16, (N - gn) * 4));
// //                 asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm), "l"(gB_ptr[i]), "r"(bytes));
// //             } else if (r < BK) { *(float4*)(Bs + sm_b_off[i]) = {0,0,0,0}; }
// //             gB_ptr[i] += (long long)BK * ldb;
// //         }
// //     };

// //     // ----------------------------------------------------------------
// //     // Register fragment loads (lane-invariant addr parts hoisted out)
// //     // ----------------------------------------------------------------

// //     // A (ldmatrix.x4, M-major SMEM):
// //     //   lane-dep:   a_ld_row  (M-offset within warp tile),
// //     //               a_ld_kbit (K-half within fragment),
// //     //               a_sw_xor  (swizzle XOR, depends only on low 2 bits of a_ld_row).
// //     //   per-call:   + mi*16 on the row, + ks on the K-half.
// //     // The swizzle XOR depends only on (m_off & 3), and m_off = warp_m_base +
// //     // a_ld_row + mi*16 where warp_m_base and mi*16 are multiples of 16 (and
// //     // hence of 4), so the low 2 bits match a_ld_row's — XOR is lane-const.
// //     const int a_ld_row       = (((lane >> 3) >> 1) ? 8 : 0) + (lane & 7);
// //     const int a_ld_kbit      = ((lane >> 3) & 1) ? 4 : 0;
// //     const int a_sw_xor       = (a_ld_row & 3) << 2;
// //     const int a_row_abs_base = wy * Config::WARP_TILE_M + a_ld_row;

// //     auto load_frA = [&](uint32_t reg[4], int ks, int mi, int st) {
// //         float* As = smem + st * Config::STAGE_SIZE;
// //         const int m_off = a_row_abs_base + mi * 16;
// //         const int k_sw  = (ks + a_ld_kbit) ^ a_sw_xor;
// //         const uint32_t addr = __cvta_generic_to_shared(&As[m_off * Config::BK + k_sw]);
// //         uint32_t d0, d1, d2, d3;
// //         LDSM_X4(d0, d1, d2, d3, addr);
// //         reg[0] = d0;  // a0 = (T/4,   T%4)
// //         reg[1] = d2;  // a1 = (T/4+8, T%4)
// //         reg[2] = d1;  // a2 = (T/4,   T%4+4)
// //         reg[3] = d3;  // a3 = (T/4+8, T%4+4)
// //     };

// //     // B (manual scalar loads, K-major SMEM):
// //     const int b_k_lane = lane & 3;                               // + ks
// //     const int b_n_base = wx * Config::WARP_TILE_N + (lane >> 2); // + ni*8

// //     auto load_frB = [&](uint32_t reg[2], int ks, int ni, int st) {
// //         float* Bs = smem + st * Config::STAGE_SIZE + Config::AS_SIZE;
// //         const int lr0 = ks + b_k_lane, lr4 = lr0 + 4;
// //         const int lc  = b_n_base + ni * 8;
// //         auto gb = [&](int r, int c) {
// //             return *(const uint32_t*)(&Bs[r * Config::BN + (c ^ ((r & 7) << 2))]);
// //         };
// //         reg[0] = gb(lr0, lc);
// //         reg[1] = gb(lr4, lc);
// //     };

// //     // ----------------------------------------------------------------
// //     // Multi-stage pipeline warmup  (STAGES-1 stages pre-filled)
// //     // ----------------------------------------------------------------
// //     load_to_stage(0, 0);
// //     asm volatile("cp.async.commit_group;\n");
// //     #pragma unroll
// //     for (int s = 1; s < Config::STAGES - 1; s++) {
// //         if (s * Config::BK < K) load_to_stage(s, s * Config::BK);
// //         asm volatile("cp.async.commit_group;\n");
// //     }

// //     int ws = Config::STAGES - 1, rs = 0;
// //     uint32_t frA[2][Config::MMA_M][4], frB[2][Config::MMA_N][2];

// //     asm volatile("cp.async.wait_group %0;\n" :: "n"(Config::STAGES - 2));
// //     __syncthreads();
// //     #pragma unroll
// //     for (int i = 0; i < Config::MMA_M; i++) load_frA(frA[0][i], 0, i, rs);
// //     #pragma unroll
// //     for (int j = 0; j < Config::MMA_N; j++) load_frB(frB[0][j], 0, j, rs);

// //     // ----------------------------------------------------------------
// //     // Main K loop
// //     // ----------------------------------------------------------------
// //     for (int k = 0; k < K; k += Config::BK) {
// //         if (k + (Config::STAGES - 1) * Config::BK < K)
// //             load_to_stage(ws, k + (Config::STAGES - 1) * Config::BK);
// //         asm volatile("cp.async.commit_group;\n");

// //         #pragma unroll
// //         for (int ks = 0; ks < Config::BK; ks += 16) {
// //             // First 8-step MMA (ks)
// //             #pragma unroll
// //             for (int i = 0; i < Config::MMA_M; i++) {
// //                 #pragma unroll
// //                 for (int j = 0; j < Config::MMA_N; j++) {
// //                     MMA_TF32(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3],
// //                              frA[0][i][0], frA[0][i][1], frA[0][i][2], frA[0][i][3],
// //                              frB[0][j][0], frB[0][j][1]);
// //                     if (i == 0) load_frB(frB[1][j], ks + 8, j, rs);
// //                 }
// //                 load_frA(frA[1][i], ks + 8, i, rs);
// //             }

// //             // Second 8-step MMA (ks + 8)
// //             #pragma unroll
// //             for (int i = 0; i < Config::MMA_M; i++) {
// //                 #pragma unroll
// //                 for (int j = 0; j < Config::MMA_N; j++) {
// //                     MMA_TF32(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3],
// //                              frA[1][i][0], frA[1][i][1], frA[1][i][2], frA[1][i][3],
// //                              frB[1][j][0], frB[1][j][1]);
// //                 }
// //             }

// //             // Prepare for next 16-element block or next stage
// //             if (ks + 16 < Config::BK) {
// //                 #pragma unroll
// //                 for (int i = 0; i < Config::MMA_M; i++) {
// //                     #pragma unroll
// //                     for (int j = 0; j < Config::MMA_N; j++) if (i == 0) load_frB(frB[0][j], ks + 16, j, rs);
// //                     load_frA(frA[0][i], ks + 16, i, rs);
// //                 }
// //             } else if (k + Config::BK < K) {
// //                 asm volatile("cp.async.wait_group %0;\n" :: "n"(Config::STAGES - 2));
// //                 __syncthreads();
// //                 rs = (rs + 1) % Config::STAGES;
// //                 ws = (ws + 1) % Config::STAGES;
// //                 #pragma unroll
// //                 for (int i = 0; i < Config::MMA_M; i++) {
// //                     #pragma unroll
// //                     for (int j = 0; j < Config::MMA_N; j++) if (i == 0) load_frB(frB[0][j], 0, j, rs);
// //                     load_frA(frA[0][i], 0, i, rs);
// //                 }
// //             }
// //         }
// //     }

// //     // ----------------------------------------------------------------
// //     // Epilogue: C[r,c] = alpha * acc + beta * bias[c]
// //     // MMA output: lane l owns rows {l/4, l/4+8}, cols {2*(l%4), 2*(l%4)+1}
// //     //
// //     // Bias-mode dispatch lifted out of the m,n loop. For MODE_VECTOR and
// //     // MODE_MATRIX the bias lookup is still per-column, but the stride test
// //     // chain (== 1 / == N / == M*N) runs once per thread, not MMA_M*MMA_N times.
// //     // ----------------------------------------------------------------
// //     const int g_epi = lane / 4, t_epi = lane & 3;
// //     float* dC = C + (long long)batch * strideC;

// //     enum { BIAS_NONE = 0, BIAS_SCALAR = 1, BIAS_VECTOR = 2, BIAS_MATRIX = 3 };
// //     const int bias_mode = !bias                           ? BIAS_NONE
// //                         : (bias_numel == 1)               ? BIAS_SCALAR
// //                         : (bias_numel == (int64_t)N)      ? BIAS_VECTOR
// //                         : (bias_numel == (int64_t)M * N)  ? BIAS_MATRIX
// //                         : BIAS_NONE;
// //     const float  bias_scalar     = (bias_mode == BIAS_SCALAR) ? bias[0] : 0.f;
// //     const float* bias_matrix_row = (bias_mode == BIAS_MATRIX)
// //                                    ? (bias + (long long)(batch % M) * N)
// //                                    : nullptr;

// //     #pragma unroll
// //     for (int i = 0; i < Config::MMA_M; i++) {
// //         #pragma unroll
// //         for (int j = 0; j < Config::MMA_N; j++) {
// //             const int r0 = by * BM + wy * Config::WARP_TILE_M + i * 16 + g_epi;
// //             const int r8 = r0 + 8;
// //             const int c0 = bx * BN + wx * Config::WARP_TILE_N + j * 8 + t_epi * 2;
// //             const int c1 = c0 + 1;

// //             float b0 = 0.f, b1 = 0.f;
// //             switch (bias_mode) {
// //                 case BIAS_SCALAR:
// //                     b0 = bias_scalar; b1 = bias_scalar; break;
// //                 case BIAS_VECTOR:
// //                     if (c0 < N) b0 = bias[c0];
// //                     if (c1 < N) b1 = bias[c1];
// //                     break;
// //                 case BIAS_MATRIX:
// //                     if (c0 < N) b0 = bias_matrix_row[c0];
// //                     if (c1 < N) b1 = bias_matrix_row[c1];
// //                     break;
// //                 default: break;
// //             }

// //             auto store = [&](int r, int c, float f, float b) __attribute__((always_inline)) {
// //                 if (r >= M || c >= N) return;
// //                 float* dst = &dC[(long long)r * ldc + c];
// //                 if (bias_mode != BIAS_NONE) {
// //                     *dst = alpha * f + beta * b;
// //                 } else {
// //                     *dst = alpha * f + (beta == 0.f ? 0.f : beta * (*dst));
// //                 }
// //             };

// //             store(r0, c0, acc[i][j][0], b0);
// //             store(r0, c1, acc[i][j][1], b1);
// //             store(r8, c0, acc[i][j][2], b0);
// //             store(r8, c1, acc[i][j][3], b1);
// //         }
// //     }
// // }

// // // ----------------------------------------------------------------
// // // Launch helper (aligned/unaligned specialisation)
// // // ----------------------------------------------------------------
// // template <int BM, int BN, int BK, int STAGES, int THREADS, bool IsAligned>
// // static void launch_sgemm_addmm_sm89(
// //     cudaStream_t stream, int M, int N, int K,
// //     float alpha,
// //     const float* A, int lda, long long strideA,
// //     const float* B, int ldb, long long strideB,
// //     float beta,
// //     const float* bias, int64_t bias_numel,
// //     float* C, int ldc, long long strideC,
// //     int batchCount)
// // {
// //     using Config = SgemmTileConfig<BM, BN, BK, STAGES, THREADS>;
// //     constexpr size_t smem_bytes = STAGES * Config::STAGE_SIZE * sizeof(float);

// //     // cudaFuncSetAttribute is idempotent and cheap; the prior std::unordered_map
// //     // guard was thread-unsafe. Just call it every time.
// //     cudaFuncSetAttribute(
// //         (const void*)sgemm_addmm_sm89_kernel<BM, BN, BK, STAGES, THREADS, IsAligned>,
// //         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);

// //     const int gx = (N + BN - 1) / BN, gy = (M + BM - 1) / BM;
// //     sgemm_addmm_sm89_kernel<BM, BN, BK, STAGES, THREADS, IsAligned>
// //         <<<dim3(gx, gy, batchCount), THREADS, smem_bytes, stream>>>(
// //             M, N, K, alpha, A, lda, strideA, B, ldb, strideB,
// //             beta, bias, bias_numel, C, ldc, strideC, batchCount);
// // }

// // // ----------------------------------------------------------------
// // // Dispatch: picks alignment variant, no splitK (addmm is single-pass)
// // // ----------------------------------------------------------------
// // template <int BM, int BN, int BK, int STAGES, int THREADS>
// // static void dispatch_sgemm_addmm_sm89(
// //     cudaStream_t stream, int M, int N, int K,
// //     float alpha,
// //     const float* A, int lda, long long strideA,
// //     const float* B, int ldb, long long strideB,
// //     float beta,
// //     const float* bias, int64_t bias_numel,
// //     float* C, int ldc, long long strideC,
// //     int batchCount)
// // {
// //     const bool aligned = (((size_t)A & 15) == 0) && (((size_t)B & 15) == 0)
// //                       && ((lda & 3) == 0) && ((ldb & 3) == 0);
// //     if (aligned)
// //         launch_sgemm_addmm_sm89<BM, BN, BK, STAGES, THREADS, true>(
// //             stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB,
// //             beta, bias, bias_numel, C, ldc, strideC, batchCount);
// //     else
// //         launch_sgemm_addmm_sm89<BM, BN, BK, STAGES, THREADS, false>(
// //             stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB,
// //             beta, bias, bias_numel, C, ldc, strideC, batchCount);
// // }

// // // ================================================================
// // // Public entry point
// // //
// // // Tile heuristic for SM89 Ada Lovelace:
// // //
// // //   M >= 256, N >= 128    →  256×128×32, 2-stage  (max tiles + wide BK)
// // //                              smem: 2 × (256×32 + 32×128) × 4 = 98.3 KB
// // //
// // //   M,N >= 128, K >= 256  →  128×128×32, 3-stage  (wide BK, better efficiency)
// // //                              smem: 3 × (128×32 + 32×128) × 4 = 96 KB
// // //
// // //   M,N >= 128, K <  256  →  128×128×16, 6-stage  (max stages, best latency hiding)
// // //                              smem: 6 × (128×16 + 16×128) × 4 = 98.3 KB
// // //
// // //   M < 128 or N < 128    →  64×64×16, 6-stage    (higher occupancy for small tiles)
// // //                              smem: 6 × (64×16 + 16×64) × 4 = 49.2 KB
// // // ================================================================
// // extern "C" void mycublasSgemmAddmm_sm89(
// //     mycublasHandle_t handle,
// //     int M, int N, int K,
// //     const float alpha,
// //     const float* A, int lda, long long int strideA,
// //     const float* B, int ldb, long long int strideB,
// //     const float beta,
// //     const float* bias, int64_t bias_numel,
// //     float* C, int ldc, long long int strideC,
// //     int batchCount)
// // {
// //     cudaStream_t stream = handle ? handle->stream : 0;

// //     if (M >= 256 && N >= 128) {
// //         dispatch_sgemm_addmm_sm89<256, 128, 32, 2, 256>(
// //             stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB,
// //             beta, bias, bias_numel, C, ldc, strideC, batchCount);
// //     } else if (M >= 128 && N >= 128) {
// //         if (K >= 256) {
// //             dispatch_sgemm_addmm_sm89<128, 128, 32, 3, 128>(
// //                 stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB,
// //                 beta, bias, bias_numel, C, ldc, strideC, batchCount);
// //         } else {
// //             dispatch_sgemm_addmm_sm89<128, 128, 16, 6, 128>(
// //                 stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB,
// //                 beta, bias, bias_numel, C, ldc, strideC, batchCount);
// //         }
// //     } else {
// //         dispatch_sgemm_addmm_sm89<64, 64, 16, 6, 64>(
// //             stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB,
// //             beta, bias, bias_numel, C, ldc, strideC, batchCount);
// //     }
// // }



















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























// #include "mycublas.h"
// #include <cuda_runtime.h>
// #include <stdint.h>
// #include "../SM86/Sgemm_core_template.cuh"

// // ============================================================
// // Sgemm Addmm SM89 — Fused Matmul + Bias for Ada Lovelace
// //
// // Operation: C = alpha * (A * B) + beta * bias
// // Layout: NN  (A:[M,K] row-major, B:[K,N] row-major)
// //
// // Optimizations (full sweep):
// //   - ldmatrix.x4 for A (M-major in SMEM: natural for TF32-via-ldmatrix)
// //   - Lane-invariant ldmatrix address arithmetic hoisted out of the fetch
// //     lambda (swizzle XOR folded into a lane-const since m*16 / warp-base
// //     don't perturb the low 2 bits of M).
// //   - Bias mode (scalar / vector / matrix) dispatched once outside the
// //     epilogue m,n loops; bias matrix row pointer resolved once.
// //   - __launch_bounds__ occupancy uses the same heuristic as core template
// //     (MAX_OCC=2 for tiles <256x128).
// //   - Dropped the std::unordered_map guard on cudaFuncSetAttribute (which was
// //     thread-unsafe); the call is cheap and idempotent.
// //   - B stays on manual scalar loads — NN B is K-major in SMEM and no
// //     ldmatrix variant delivers the MMA B-fragment thread distribution for
// //     that storage orientation without corrupting TF32 values.
// // ============================================================

// #ifndef MMA_TF32
// #define MMA_TF32(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1)                \
//     asm volatile(                                                   \
//         "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "     \
//         "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"       \
//         : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)                     \
//         : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1))
// #endif

// #ifndef LDSM_X4
// #define LDSM_X4(r0,r1,r2,r3,addr)                                \
//     asm volatile(                                                   \
//         "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "              \
//         "{%0,%1,%2,%3},[%4];"                                     \
//         : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(addr))
// #endif

// // Occupancy heuristic mirroring SgemmTileConfigSM89::MAX_OCC
// template <int BM, int BN> struct AddmmOcc { static constexpr int V = (BM * BN >= 256 * 128) ? 1 : 2; };

// // bias_numel: 1   → scalar broadcast
// //             N   → 1D row vector (common case: linear layer bias)
// //             M*N → matrix bias, indexed as batch-major (batch % M)
// template <int BM, int BN, int BK, int STAGES, int THREADS, bool IsAligned>
// __global__ void __launch_bounds__(THREADS, AddmmOcc<BM,BN>::V)
// sgemm_addmm_sm89_kernel(
//     int M, int N, int K,
//     float alpha,
//     const float* __restrict__ A, int lda, long long strideA,
//     const float* __restrict__ B, int ldb, long long strideB,
//     float beta,
//     const float* __restrict__ bias, int64_t bias_numel,
//     float* __restrict__ C, int ldc, long long strideC,
//     int batchCount)
// {
//     using Config = SgemmTileConfig<BM, BN, BK, STAGES, THREADS>;

//     const int batch = blockIdx.z;
//     if (batch >= batchCount) return;

//     // CTA swizzle (8-wide strips) guarded against short-last-strip edge cases.
//     const int sw = 8;
//     const int grid_x = (N + BN - 1) / BN, grid_y = (M + BM - 1) / BM;
//     const int block_idx = blockIdx.y * grid_x + blockIdx.x;
//     const int num_blocks_per_strip = sw * grid_y;
//     const int strip_idx = block_idx / num_blocks_per_strip;
//     const int strip_off = block_idx % num_blocks_per_strip;
//     const int strip_base = strip_idx * sw;
//     if (strip_base >= grid_x) return;
//     const int actual_sw = min(sw, grid_x - strip_base);
//     const int bx = strip_base + (strip_off % actual_sw);
//     const int by = strip_off / actual_sw;

//     if (bx >= grid_x || by >= grid_y) return;

//     const int tid  = threadIdx.x;
//     const int lane = tid & 31, wid = tid >> 5;
//     const int wy   = wid / Config::WARPS_N;
//     const int wx   = wid % Config::WARPS_N;

//     extern __shared__ float smem[];

//     float acc[Config::MMA_M][Config::MMA_N][4];
//     #pragma unroll
//     for (int i = 0; i < Config::MMA_M; i++)
//         #pragma unroll
//         for (int j = 0; j < Config::MMA_N; j++) {
//             acc[i][j][0] = 0.f; acc[i][j][1] = 0.f;
//             acc[i][j][2] = 0.f; acc[i][j][3] = 0.f;
//         }

//     // ----------------------------------------------------------------
//     // Global pointer induction
//     // A: [M, K] — NT load (BM rows × BK cols, vectorised along K)
//     // B: [K, N] — TN load (BK rows × BN cols, vectorised along N)
//     // ----------------------------------------------------------------
//     const float* gA_ptr[Config::NT_LOAD_ITERS_A];
//     const float* gB_ptr[Config::TN_LOAD_ITERS_B];

//     #pragma unroll
//     for (int i = 0; i < Config::NT_LOAD_ITERS_A; i++) {
//         const int r = (tid / Config::NT_THREADS_PER_ROW_A) + i * Config::NT_ROWS_PER_ITER_A;
//         const int c = (tid % Config::NT_THREADS_PER_ROW_A) * 4;
//         gA_ptr[i] = A + (long long)batch * strideA
//                       + (long long)(by * BM + r) * lda + c;
//     }
//     #pragma unroll
//     for (int i = 0; i < Config::TN_LOAD_ITERS_B; i++) {
//         const int r = (tid / Config::TN_THREADS_PER_ROW_B) + i * Config::TN_ROWS_PER_ITER_B;
//         const int c = (tid % Config::TN_THREADS_PER_ROW_B) * 4;
//         gB_ptr[i] = B + (long long)batch * strideB
//                       + (long long)r * ldb + (bx * BN + c);
//     }

//     // ----------------------------------------------------------------
//     // SMEM offsets (computed once per thread)
//     // ----------------------------------------------------------------
//     uint32_t sm_a_off[Config::NT_LOAD_ITERS_A];
//     #pragma unroll
//     for (int i = 0; i < Config::NT_LOAD_ITERS_A; i++) {
//         const int r = (tid / Config::NT_THREADS_PER_ROW_A) + i * Config::NT_ROWS_PER_ITER_A;
//         const int c = (tid % Config::NT_THREADS_PER_ROW_A) * 4;
//         const int sc = c ^ ((r & 3) << 2);
//         sm_a_off[i] = r * Config::BK + sc;
//     }

//     uint32_t sm_b_off[Config::TN_LOAD_ITERS_B];
//     #pragma unroll
//     for (int i = 0; i < Config::TN_LOAD_ITERS_B; i++) {
//         const int r = (tid / Config::TN_THREADS_PER_ROW_B) + i * Config::TN_ROWS_PER_ITER_B;
//         const int c = (tid % Config::TN_THREADS_PER_ROW_B) * 4;
//         const int sc = c ^ ((r & 7) << 2);
//         sm_b_off[i] = r * Config::BN + sc;
//     }

//     auto load_to_stage = [&](int stage, int ko) {
//         float* As = smem + stage * Config::STAGE_SIZE;
//         float* Bs = As   + Config::AS_SIZE;
//         #pragma unroll
//         for (int i = 0; i < Config::NT_LOAD_ITERS_A; i++) {
//             const int r = (tid / Config::NT_THREADS_PER_ROW_A) + i * Config::NT_ROWS_PER_ITER_A;
//             const int c = (tid % Config::NT_THREADS_PER_ROW_A) * 4;
//             const int gr = by * BM + r, gc = ko + c;
//             uint32_t sm = __cvta_generic_to_shared(As + sm_a_off[i]);
//             if (gr < M && r < BM) {
//                 int bytes = max(0, min(16, (K - gc) * 4));
//                 asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm), "l"(gA_ptr[i]), "r"(bytes));
//             } else if (r < BM) { *(float4*)(As + sm_a_off[i]) = {0,0,0,0}; }
//             gA_ptr[i] += BK;
//         }
//         #pragma unroll
//         for (int i = 0; i < Config::TN_LOAD_ITERS_B; i++) {
//             const int r = (tid / Config::TN_THREADS_PER_ROW_B) + i * Config::TN_ROWS_PER_ITER_B;
//             const int c = (tid % Config::TN_THREADS_PER_ROW_B) * 4;
//             const int gk = ko + r, gn = bx * BN + c;
//             uint32_t sm = __cvta_generic_to_shared(Bs + sm_b_off[i]);
//             if (gk < K && r < BK) {
//                 int bytes = max(0, min(16, (N - gn) * 4));
//                 asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm), "l"(gB_ptr[i]), "r"(bytes));
//             } else if (r < BK) { *(float4*)(Bs + sm_b_off[i]) = {0,0,0,0}; }
//             gB_ptr[i] += (long long)BK * ldb;
//         }
//     };

//     // ----------------------------------------------------------------
//     // Register fragment loads (lane-invariant addr parts hoisted out)
//     // ----------------------------------------------------------------

//     // A (ldmatrix.x4, M-major SMEM):
//     //   lane-dep:   a_ld_row  (M-offset within warp tile),
//     //               a_ld_kbit (K-half within fragment),
//     //               a_sw_xor  (swizzle XOR, depends only on low 2 bits of a_ld_row).
//     //   per-call:   + mi*16 on the row, + ks on the K-half.
//     // The swizzle XOR depends only on (m_off & 3), and m_off = warp_m_base +
//     // a_ld_row + mi*16 where warp_m_base and mi*16 are multiples of 16 (and
//     // hence of 4), so the low 2 bits match a_ld_row's — XOR is lane-const.
//     const int a_ld_row       = (((lane >> 3) >> 1) ? 8 : 0) + (lane & 7);
//     const int a_ld_kbit      = ((lane >> 3) & 1) ? 4 : 0;
//     const int a_sw_xor       = (a_ld_row & 3) << 2;
//     const int a_row_abs_base = wy * Config::WARP_TILE_M + a_ld_row;

//     auto load_frA = [&](uint32_t reg[4], int ks, int mi, int st) {
//         float* As = smem + st * Config::STAGE_SIZE;
//         const int m_off = a_row_abs_base + mi * 16;
//         const int k_sw  = (ks + a_ld_kbit) ^ a_sw_xor;
//         const uint32_t addr = __cvta_generic_to_shared(&As[m_off * Config::BK + k_sw]);
//         uint32_t d0, d1, d2, d3;
//         LDSM_X4(d0, d1, d2, d3, addr);
//         reg[0] = d0;  // a0 = (T/4,   T%4)
//         reg[1] = d2;  // a1 = (T/4+8, T%4)
//         reg[2] = d1;  // a2 = (T/4,   T%4+4)
//         reg[3] = d3;  // a3 = (T/4+8, T%4+4)
//     };

//     // B (manual scalar loads, K-major SMEM):
//     const int b_k_lane = lane & 3;                               // + ks
//     const int b_n_base = wx * Config::WARP_TILE_N + (lane >> 2); // + ni*8

//     auto load_frB = [&](uint32_t reg[2], int ks, int ni, int st) {
//         float* Bs = smem + st * Config::STAGE_SIZE + Config::AS_SIZE;
//         const int lr0 = ks + b_k_lane, lr4 = lr0 + 4;
//         const int lc  = b_n_base + ni * 8;
//         auto gb = [&](int r, int c) {
//             return *(const uint32_t*)(&Bs[r * Config::BN + (c ^ ((r & 7) << 2))]);
//         };
//         reg[0] = gb(lr0, lc);
//         reg[1] = gb(lr4, lc);
//     };

//     // ----------------------------------------------------------------
//     // Multi-stage pipeline warmup  (STAGES-1 stages pre-filled)
//     // ----------------------------------------------------------------
//     load_to_stage(0, 0);
//     asm volatile("cp.async.commit_group;\n");
//     #pragma unroll
//     for (int s = 1; s < Config::STAGES - 1; s++) {
//         if (s * Config::BK < K) load_to_stage(s, s * Config::BK);
//         asm volatile("cp.async.commit_group;\n");
//     }

//     int ws = Config::STAGES - 1, rs = 0;
//     uint32_t frA[2][Config::MMA_M][4], frB[2][Config::MMA_N][2];

//     asm volatile("cp.async.wait_group %0;\n" :: "n"(Config::STAGES - 2));
//     __syncthreads();
//     #pragma unroll
//     for (int i = 0; i < Config::MMA_M; i++) load_frA(frA[0][i], 0, i, rs);
//     #pragma unroll
//     for (int j = 0; j < Config::MMA_N; j++) load_frB(frB[0][j], 0, j, rs);

//     // ----------------------------------------------------------------
//     // Main K loop
//     // ----------------------------------------------------------------
//     for (int k = 0; k < K; k += Config::BK) {
//         if (k + (Config::STAGES - 1) * Config::BK < K)
//             load_to_stage(ws, k + (Config::STAGES - 1) * Config::BK);
//         asm volatile("cp.async.commit_group;\n");

//         #pragma unroll
//         for (int ks = 0; ks < Config::BK; ks += 16) {
//             // First 8-step MMA (ks)
//             #pragma unroll
//             for (int i = 0; i < Config::MMA_M; i++) {
//                 #pragma unroll
//                 for (int j = 0; j < Config::MMA_N; j++) {
//                     MMA_TF32(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3],
//                              frA[0][i][0], frA[0][i][1], frA[0][i][2], frA[0][i][3],
//                              frB[0][j][0], frB[0][j][1]);
//                     if (i == 0) load_frB(frB[1][j], ks + 8, j, rs);
//                 }
//                 load_frA(frA[1][i], ks + 8, i, rs);
//             }

//             // Second 8-step MMA (ks + 8)
//             #pragma unroll
//             for (int i = 0; i < Config::MMA_M; i++) {
//                 #pragma unroll
//                 for (int j = 0; j < Config::MMA_N; j++) {
//                     MMA_TF32(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3],
//                              frA[1][i][0], frA[1][i][1], frA[1][i][2], frA[1][i][3],
//                              frB[1][j][0], frB[1][j][1]);
//                 }
//             }

//             // Prepare for next 16-element block or next stage
//             if (ks + 16 < Config::BK) {
//                 #pragma unroll
//                 for (int i = 0; i < Config::MMA_M; i++) {
//                     #pragma unroll
//                     for (int j = 0; j < Config::MMA_N; j++) if (i == 0) load_frB(frB[0][j], ks + 16, j, rs);
//                     load_frA(frA[0][i], ks + 16, i, rs);
//                 }
//             } else if (k + Config::BK < K) {
//                 asm volatile("cp.async.wait_group %0;\n" :: "n"(Config::STAGES - 2));
//                 __syncthreads();
//                 rs = (rs + 1) % Config::STAGES;
//                 ws = (ws + 1) % Config::STAGES;
//                 #pragma unroll
//                 for (int i = 0; i < Config::MMA_M; i++) {
//                     #pragma unroll
//                     for (int j = 0; j < Config::MMA_N; j++) if (i == 0) load_frB(frB[0][j], 0, j, rs);
//                     load_frA(frA[0][i], 0, i, rs);
//                 }
//             }
//         }
//     }

//     // ----------------------------------------------------------------
//     // Epilogue: C[r,c] = alpha * acc + beta * bias[c]
//     // MMA output: lane l owns rows {l/4, l/4+8}, cols {2*(l%4), 2*(l%4)+1}
//     //
//     // Bias-mode dispatch lifted out of the m,n loop. For MODE_VECTOR and
//     // MODE_MATRIX the bias lookup is still per-column, but the stride test
//     // chain (== 1 / == N / == M*N) runs once per thread, not MMA_M*MMA_N times.
//     // ----------------------------------------------------------------
//     const int g_epi = lane / 4, t_epi = lane & 3;
//     float* dC = C + (long long)batch * strideC;

//     enum { BIAS_NONE = 0, BIAS_SCALAR = 1, BIAS_VECTOR = 2, BIAS_MATRIX = 3 };
//     const int bias_mode = !bias                           ? BIAS_NONE
//                         : (bias_numel == 1)               ? BIAS_SCALAR
//                         : (bias_numel == (int64_t)N)      ? BIAS_VECTOR
//                         : (bias_numel == (int64_t)M * N)  ? BIAS_MATRIX
//                         : BIAS_NONE;
//     const float  bias_scalar     = (bias_mode == BIAS_SCALAR) ? bias[0] : 0.f;
//     const float* bias_matrix_row = (bias_mode == BIAS_MATRIX)
//                                    ? (bias + (long long)(batch % M) * N)
//                                    : nullptr;

//     #pragma unroll
//     for (int i = 0; i < Config::MMA_M; i++) {
//         #pragma unroll
//         for (int j = 0; j < Config::MMA_N; j++) {
//             const int r0 = by * BM + wy * Config::WARP_TILE_M + i * 16 + g_epi;
//             const int r8 = r0 + 8;
//             const int c0 = bx * BN + wx * Config::WARP_TILE_N + j * 8 + t_epi * 2;
//             const int c1 = c0 + 1;

//             float b0 = 0.f, b1 = 0.f;
//             switch (bias_mode) {
//                 case BIAS_SCALAR:
//                     b0 = bias_scalar; b1 = bias_scalar; break;
//                 case BIAS_VECTOR:
//                     if (c0 < N) b0 = bias[c0];
//                     if (c1 < N) b1 = bias[c1];
//                     break;
//                 case BIAS_MATRIX:
//                     if (c0 < N) b0 = bias_matrix_row[c0];
//                     if (c1 < N) b1 = bias_matrix_row[c1];
//                     break;
//                 default: break;
//             }

//             auto store = [&](int r, int c, float f, float b) __attribute__((always_inline)) {
//                 if (r >= M || c >= N) return;
//                 float* dst = &dC[(long long)r * ldc + c];
//                 if (bias_mode != BIAS_NONE) {
//                     *dst = alpha * f + beta * b;
//                 } else {
//                     *dst = alpha * f + (beta == 0.f ? 0.f : beta * (*dst));
//                 }
//             };

//             store(r0, c0, acc[i][j][0], b0);
//             store(r0, c1, acc[i][j][1], b1);
//             store(r8, c0, acc[i][j][2], b0);
//             store(r8, c1, acc[i][j][3], b1);
//         }
//     }
// }

// // ----------------------------------------------------------------
// // Launch helper (aligned/unaligned specialisation)
// // ----------------------------------------------------------------
// template <int BM, int BN, int BK, int STAGES, int THREADS, bool IsAligned>
// static void launch_sgemm_addmm_sm89(
//     cudaStream_t stream, int M, int N, int K,
//     float alpha,
//     const float* A, int lda, long long strideA,
//     const float* B, int ldb, long long strideB,
//     float beta,
//     const float* bias, int64_t bias_numel,
//     float* C, int ldc, long long strideC,
//     int batchCount)
// {
//     using Config = SgemmTileConfig<BM, BN, BK, STAGES, THREADS>;
//     constexpr size_t smem_bytes = STAGES * Config::STAGE_SIZE * sizeof(float);

//     // cudaFuncSetAttribute is idempotent and cheap; the prior std::unordered_map
//     // guard was thread-unsafe. Just call it every time.
//     cudaFuncSetAttribute(
//         (const void*)sgemm_addmm_sm89_kernel<BM, BN, BK, STAGES, THREADS, IsAligned>,
//         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);

//     const int gx = (N + BN - 1) / BN, gy = (M + BM - 1) / BM;
//     sgemm_addmm_sm89_kernel<BM, BN, BK, STAGES, THREADS, IsAligned>
//         <<<dim3(gx, gy, batchCount), THREADS, smem_bytes, stream>>>(
//             M, N, K, alpha, A, lda, strideA, B, ldb, strideB,
//             beta, bias, bias_numel, C, ldc, strideC, batchCount);
// }

// // ----------------------------------------------------------------
// // Dispatch: picks alignment variant, no splitK (addmm is single-pass)
// // ----------------------------------------------------------------
// template <int BM, int BN, int BK, int STAGES, int THREADS>
// static void dispatch_sgemm_addmm_sm89(
//     cudaStream_t stream, int M, int N, int K,
//     float alpha,
//     const float* A, int lda, long long strideA,
//     const float* B, int ldb, long long strideB,
//     float beta,
//     const float* bias, int64_t bias_numel,
//     float* C, int ldc, long long strideC,
//     int batchCount)
// {
//     const bool aligned = (((size_t)A & 15) == 0) && (((size_t)B & 15) == 0)
//                       && ((lda & 3) == 0) && ((ldb & 3) == 0);
//     if (aligned)
//         launch_sgemm_addmm_sm89<BM, BN, BK, STAGES, THREADS, true>(
//             stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB,
//             beta, bias, bias_numel, C, ldc, strideC, batchCount);
//     else
//         launch_sgemm_addmm_sm89<BM, BN, BK, STAGES, THREADS, false>(
//             stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB,
//             beta, bias, bias_numel, C, ldc, strideC, batchCount);
// }

// // ================================================================
// // Public entry point
// //
// // Tile heuristic for SM89 Ada Lovelace:
// //
// //   M >= 256, N >= 128    →  256×128×32, 2-stage  (max tiles + wide BK)
// //                              smem: 2 × (256×32 + 32×128) × 4 = 98.3 KB
// //
// //   M,N >= 128, K >= 256  →  128×128×32, 3-stage  (wide BK, better efficiency)
// //                              smem: 3 × (128×32 + 32×128) × 4 = 96 KB
// //
// //   M,N >= 128, K <  256  →  128×128×16, 6-stage  (max stages, best latency hiding)
// //                              smem: 6 × (128×16 + 16×128) × 4 = 98.3 KB
// //
// //   M < 128 or N < 128    →  64×64×16, 6-stage    (higher occupancy for small tiles)
// //                              smem: 6 × (64×16 + 16×64) × 4 = 49.2 KB
// // ================================================================
// extern "C" void mycublasSgemmAddmm_sm89(
//     mycublasHandle_t handle,
//     int M, int N, int K,
//     const float alpha,
//     const float* A, int lda, long long int strideA,
//     const float* B, int ldb, long long int strideB,
//     const float beta,
//     const float* bias, int64_t bias_numel,
//     float* C, int ldc, long long int strideC,
//     int batchCount)
// {
//     cudaStream_t stream = handle ? handle->stream : 0;

//     if (M >= 256 && N >= 128) {
//         dispatch_sgemm_addmm_sm89<256, 128, 32, 2, 256>(
//             stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB,
//             beta, bias, bias_numel, C, ldc, strideC, batchCount);
//     } else if (M >= 128 && N >= 128) {
//         if (K >= 256) {
//             dispatch_sgemm_addmm_sm89<128, 128, 32, 3, 128>(
//                 stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB,
//                 beta, bias, bias_numel, C, ldc, strideC, batchCount);
//         } else {
//             dispatch_sgemm_addmm_sm89<128, 128, 16, 6, 128>(
//                 stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB,
//                 beta, bias, bias_numel, C, ldc, strideC, batchCount);
//         }
//     } else {
//         dispatch_sgemm_addmm_sm89<64, 64, 16, 6, 64>(
//             stream, M, N, K, alpha, A, lda, strideA, B, ldb, strideB,
//             beta, bias, bias_numel, C, ldc, strideC, batchCount);
//     }
// }