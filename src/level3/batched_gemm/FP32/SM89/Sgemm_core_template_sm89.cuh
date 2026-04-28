// // #pragma once

// // #include <cuda_runtime.h>
// // #include <stdint.h>
// // #include <stdio.h>

// // /**
// //  * Optimized SM89 SGEMM Core — High Performance & Accuracy
// //  */

// // #ifndef MYCUBLAS_LAYOUT_ENUM
// // #define MYCUBLAS_LAYOUT_ENUM
// // enum class SgemmLayout { NT, TN, NN };
// // #endif

// // // ---------------------------------------------------------------------------
// // // PTX helpers
// // // ---------------------------------------------------------------------------
// // #ifndef MMA_TF32
// // #define MMA_TF32(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1)                           \
// //     asm volatile(                                                              \
// //         "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "                \
// //         "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"                  \
// //         : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)                                \
// //         : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1))
// // #endif

// // #ifndef LDSM_X4
// // #define LDSM_X4(r0,r1,r2,r3,addr)                                            \
// //     asm volatile(                                                              \
// //         "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];"      \
// //         : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(addr))
// // #endif

// // #ifndef LDSM_X2
// // #define LDSM_X2(r0,r1,addr)                                                  \
// //     asm volatile(                                                              \
// //         "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1},[%2];"            \
// //         : "=r"(r0),"=r"(r1) : "r"(addr))
// // #endif

// // #ifndef LDSM_X1
// // #define LDSM_X1(r0,r1,addr)                                                  \
// //     asm volatile(                                                              \
// //         "ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0,%1},[%2];"            \
// //         : "=r"(r0),"=r"(r1) : "r"(addr))
// // #endif

// // #ifndef CP_ASYNC_CG
// // #define CP_ASYNC_CG(dst, src)                                                 \
// //     asm volatile("cp.async.cg.shared.global [%0],[%1],16;" :: "r"(dst),"l"(src))
// // #endif

// // // ---------------------------------------------------------------------------
// // // Tile config
// // // ---------------------------------------------------------------------------
// // template <int BM_, int BN_, int BK_, int STAGES_, int THREADS_>
// // struct SgemmTileConfigSM89 {
// //     static constexpr int BM      = BM_;
// //     static constexpr int BN      = BN_;
// //     static constexpr int BK      = BK_;
// //     static constexpr int STAGES  = STAGES_;
// //     static constexpr int THREADS = THREADS_;

// //     static constexpr int AS_SIZE    = BM * BK;
// //     static constexpr int BS_SIZE    = BK * BN;
// //     static constexpr int STAGE_SIZE = AS_SIZE + BS_SIZE;

// //     static constexpr int SMEM_BYTES = STAGES * (BM + BN) * BK * sizeof(float);
// //     static constexpr int MAX_OCC    = (BM * BN >= 256 * 128) ? 1 : 2;

// //     static constexpr int WARPS_TOTAL = THREADS / 32;
// //     static constexpr int WARPS_N     = (BN >= 64 && WARPS_TOTAL > 1)
// //                                          ? ((BN / 64 < WARPS_TOTAL / 2) ? BN / 64 : WARPS_TOTAL / 2)
// //                                          : 1;
// //     static constexpr int WARPS_M     = (WARPS_TOTAL > 1) ? (WARPS_TOTAL / WARPS_N) : 1;

// //     static constexpr int WARP_TILE_M = BM / WARPS_M;
// //     static constexpr int WARP_TILE_N = BN / WARPS_N;

// //     static constexpr int MMA_M = WARP_TILE_M / 16;
// //     static constexpr int MMA_N = WARP_TILE_N / 8;
// // };

// // // ---------------------------------------------------------------------------
// // // Kernel
// // // ---------------------------------------------------------------------------
// // template <typename Cfg, bool IsAligned, bool IsSplitK, SgemmLayout Layout>
// // __global__ void __launch_bounds__(Cfg::THREADS, Cfg::MAX_OCC)
// // sgemm_sm89_kernel(
// //     int M, int N, int K, float alpha,
// //     const float* __restrict__ A, int lda, long long strideA,
// //     const float* __restrict__ B, int ldb, long long strideB,
// //     float beta,
// //     float* __restrict__ C, int ldc, long long strideC,
// //     const float* __restrict__ bias, long long bias_stride,
// //     int batchCount, int splitK)
// // {
// //     const int batch = blockIdx.z / splitK;
// //     const int sk_idx = blockIdx.z % splitK;
// //     if (batch >= batchCount) return;

// //     // Robust CTA swizzle for L2 locality (8-wide strips)
// //     const int sw = 8;
// //     const int grid_x = gridDim.x, grid_y = gridDim.y;
// //     const int block_idx = blockIdx.y * grid_x + blockIdx.x;
    
// //     const int num_blocks_per_strip = sw * grid_y;
// //     const int strip_idx = block_idx / num_blocks_per_strip;
// //     const int strip_off = block_idx % num_blocks_per_strip;
// //     const int actual_sw = min(sw, grid_x - strip_idx * sw);
    
// //     const int bx = strip_idx * sw + (strip_off % actual_sw);
// //     const int by = strip_off / actual_sw;

// //     if (bx >= grid_x || by >= grid_y) return;

// //     const int tid  = threadIdx.x;
// //     const int lane = tid & 31, wid = tid >> 5;
// //     const int wy   = wid / Cfg::WARPS_N, wx = wid % Cfg::WARPS_N;

// //     extern __shared__ float smem[];

// //     float acc[Cfg::MMA_M][Cfg::MMA_N][4];
// //     #pragma unroll
// //     for (int i = 0; i < Cfg::MMA_M; i++)
// //         for (int j = 0; j < Cfg::MMA_N; j++)
// //             acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

// //     const int k_chunk = ((K + 15) / 16 + splitK - 1) / splitK * 16;
// //     const int k_start = sk_idx * k_chunk;
// //     const int k_end   = min(K, k_start + k_chunk);
// //     const int ktcnt   = (k_end - k_start + 15) / 16;

// //     const float* A_base = A + (long long)batch * strideA;
// //     const float* B_base = B + (long long)batch * strideB;

// //     // Load params
// //     constexpr int A_STRIDE = (Layout == SgemmLayout::TN) ? Cfg::BM : Cfg::BK;
// //     constexpr int A_TPR    = A_STRIDE / 4;
// //     constexpr int A_RPL    = Cfg::THREADS / A_TPR;
// //     constexpr int A_ITERS  = (Layout == SgemmLayout::TN) ? (Cfg::BK / A_RPL) : (Cfg::BM / A_RPL);
// //     const int a_row0 = tid / A_TPR, a_col0 = (tid % A_TPR) * 4;

// //     uint32_t sm_a_off[A_ITERS];
// //     #pragma unroll
// //     for (int i = 0; i < A_ITERS; i++) {
// //         const int r = a_row0 + i * A_RPL;
// //         if constexpr (Layout == SgemmLayout::TN) sm_a_off[i] = r * Cfg::BM + (a_col0 ^ ((r & 7) << 3));
// //         else sm_a_off[i] = r * Cfg::BK + (a_col0 ^ ((r & 3) << 2)); // BK=16 swizzle
// //     }

// //     constexpr int B_STRIDE = (Layout == SgemmLayout::NT) ? Cfg::BK : Cfg::BN;
// //     constexpr int B_TPR    = B_STRIDE / 4;
// //     constexpr int B_RPL    = Cfg::THREADS / B_TPR;
// //     constexpr int B_ITERS  = (Layout == SgemmLayout::NT) ? (Cfg::BN / B_RPL) : (Cfg::BK / B_RPL);
// //     const int b_row0 = tid / B_TPR, b_col0 = (tid % B_TPR) * 4;

// //     uint32_t sm_b_off[B_ITERS];
// //     #pragma unroll
// //     for (int i = 0; i < B_ITERS; i++) {
// //         const int r = b_row0 + i * B_RPL;
// //         if constexpr (Layout == SgemmLayout::NT) sm_b_off[i] = r * Cfg::BK + (b_col0 ^ ((r & 3) << 2));
// //         else sm_b_off[i] = r * Cfg::BN + (b_col0 ^ ((r & 7) << 2)); // BN=128 swizzle (8 row)
// //     }

// //     auto load_to_smem = [&](int stage, int ko) {
// //         float* As = smem + stage * Cfg::STAGE_SIZE;
// //         float* Bs = As   + Cfg::AS_SIZE;
// //         #pragma unroll
// //         for (int i = 0; i < A_ITERS; i++) {
// //             const int r = a_row0 + i * A_RPL, gm = (Layout == SgemmLayout::TN) ? (by * Cfg::BM + a_col0) : (ko + a_col0);
// //             const int gr = (Layout == SgemmLayout::TN) ? (ko + r) : (by * Cfg::BM + r);
// //             if (gr < M || (Layout == SgemmLayout::TN && gr < K)) { // Bounds check simplified
// //                 uint32_t sm = (uint32_t)__cvta_generic_to_shared(As + sm_a_off[i]);
// //                 const float* g = A_base + (long long)gr * lda + gm;
// //                 if (IsAligned && ((Layout == SgemmLayout::TN) ? (gm+3 < M) : (gm+3 < K))) CP_ASYNC_CG(sm, g);
// //                 else {
// //                     float4 val = {0.f,0.f,0.f,0.f};
// //                     const int lim = (Layout == SgemmLayout::TN) ? M : K;
// //                     if (gm < lim) val.x = g[0]; if (gm+1 < lim) val.y = g[1]; if (gm+2 < lim) val.z = g[2]; if (gm+3 < lim) val.w = g[3];
// //                     *(float4*)(As + sm_a_off[i]) = val;
// //                 }
// //             }
// //         }
// //         #pragma unroll
// //         for (int i = 0; i < B_ITERS; i++) {
// //             const int r = b_row0 + i * B_RPL, gn = (Layout == SgemmLayout::NT) ? (bx * Cfg::BN + r) : (bx * Cfg::BN + b_col0);
// //             const int gk = (Layout == SgemmLayout::NT) ? (ko + b_col0) : (ko + r);
// //             const int row_g = (Layout == SgemmLayout::NT) ? gn : gk, col_g = (Layout == SgemmLayout::NT) ? gk : gn;
// //             uint32_t sm = (uint32_t)__cvta_generic_to_shared(Bs + sm_b_off[i]);
// //             const float* g = B_base + (long long)row_g * ldb + col_g;
// //             const int lim = (Layout == SgemmLayout::NT) ? K : N;
// //             if ((Layout == SgemmLayout::NT ? gn < N : gk < K) && IsAligned && col_g + 3 < lim) CP_ASYNC_CG(sm, g);
// //             else if (Layout == SgemmLayout::NT ? gn < N : gk < K) {
// //                 float4 val = {0,0,0,0};
// //                 if (col_g < lim) val.x = g[0]; if (col_g+1 < lim) val.y = g[1]; if (col_g+2 < lim) val.z = g[2]; if (col_g+3 < lim) val.w = g[3];
// //                 *(float4*)(Bs + sm_b_off[i]) = val;
// //             }
// //         }
// //     };

// //     auto fetch_a = [&](uint32_t reg[4], int ks, int m, const float* As_st) {
// //         if constexpr (Layout == SgemmLayout::TN) {
// //             const int m_idx = wy * Cfg::WARP_TILE_M + m * 16 + (lane / 4);
// //             const int k0 = ks + (lane % 4), k4 = k0 + 4;
// //             auto la = [&](int cur_k, int cur_m) { return *(const uint32_t*)(&As_st[cur_k * Cfg::BM + (cur_m ^ ((cur_k & 7) << 3))]); };
// //             reg[0] = la(k0, m_idx); reg[1] = la(k0, m_idx + 8); reg[2] = la(k4, m_idx); reg[3] = la(k4, m_idx + 8);
// //         } else {
// //             const int lb = wy * Cfg::WARP_TILE_M + m * 16, lr0 = lb + (lane / 4), lr8 = lr0 + 8, lc = ks + (lane % 4);
// //             auto ga = [&](int r, int c) { return *(const uint32_t*)(&As_st[r * Cfg::BK + (c ^ ((r & 3) << 2))]); };
// //             reg[0] = ga(lr0, lc); reg[1] = ga(lr8, lc); reg[2] = ga(lr0, lc + 4); reg[3] = ga(lr8, lc + 4);
// //         }
// //     };

// //     auto fetch_b = [&](uint32_t reg[2], int ks, int n, const float* Bs_st) {
// //         if constexpr (Layout == SgemmLayout::NT) {
// //             const int lr = wx * Cfg::WARP_TILE_N + n * 8 + (lane / 4), lc0 = ks + (lane % 4), lc4 = lc0 + 4;
// //             auto gb = [&](int r, int c) { return *(const uint32_t*)(&Bs_st[r * Cfg::BK + (c ^ ((r & 3) << 2))]); };
// //             reg[0] = gb(lr, lc0); reg[1] = gb(lr, lc4);
// //         } else {
// //             const int lr0 = ks + (lane % 4), lr4 = lr0 + 4, lc = wx * Cfg::WARP_TILE_N + n * 8 + (lane / 4);
// //             auto gb = [&](int r, int c) { return *(const uint32_t*)(&Bs_st[r * Cfg::BN + (c ^ ((r & 7) << 2))]); };
// //             reg[0] = gb(lr0, lc); reg[1] = gb(lr4, lc);
// //         }
// //     };

// //     // Warmup
// //     constexpr int PREFETCH = Cfg::STAGES - 1;
// //     int rs = 0, ws = 0;
// //     #pragma unroll
// //     for (int i = 0; i < PREFETCH; i++) { if (i < ktcnt) load_to_smem(ws, k_start + i * Cfg::BK); asm volatile("cp.async.commit_group;"); ws = (ws + 1) % Cfg::STAGES; }
// //     asm volatile("cp.async.wait_group %0;" :: "n"(Cfg::STAGES - 2)); __syncthreads();

// //     uint32_t frA[2][Cfg::MMA_M][4], frB[2][Cfg::MMA_N][2];
// //     #pragma unroll
// //     for (int m = 0; m < Cfg::MMA_M; m++) fetch_a(frA[0][m], 0, m, smem + rs * Cfg::STAGE_SIZE);
// //     #pragma unroll
// //     for (int n = 0; n < Cfg::MMA_N; n++) fetch_b(frB[0][n], 0, n, smem + rs * Cfg::STAGE_SIZE + Cfg::AS_SIZE);

// //     for (int kt = 0; kt < ktcnt; kt++) {
// //         const int kf = k_start + (kt + PREFETCH) * Cfg::BK;
// //         if (kf < k_end) load_to_smem(ws, kf);
// //         asm volatile("cp.async.commit_group;");

// //         #pragma unroll
// //         for (int m = 0; m < Cfg::MMA_M; m++) {
// //             #pragma unroll
// //             for (int n = 0; n < Cfg::MMA_N; n++) {
// //                 float &d0 = acc[m][n][0], &d1 = acc[m][n][1], &d2 = acc[m][n][2], &d3 = acc[m][n][3];
// //                 uint32_t a0 = frA[0][m][0], a1 = frA[0][m][1], a2 = frA[0][m][2], a3 = frA[0][m][3];
// //                 uint32_t b0 = frB[0][n][0], b1 = frB[0][n][1];
// //                 MMA_TF32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
// //                 if (m == 0) fetch_b(frB[1][n], 8, n, smem + rs * Cfg::STAGE_SIZE + Cfg::AS_SIZE);
// //             }
// //             fetch_a(frA[1][m], 8, m, smem + rs * Cfg::STAGE_SIZE);
// //         }
// //         asm volatile("cp.async.wait_group %0;" :: "n"(Cfg::STAGES - 2)); __syncthreads();
// //         rs = (rs + 1) % Cfg::STAGES; ws = (ws + 1) % Cfg::STAGES;
// //         #pragma unroll
// //         for (int m = 0; m < Cfg::MMA_M; m++) {
// //             #pragma unroll
// //             for (int n = 0; n < Cfg::MMA_N; n++) {
// //                 float &d0 = acc[m][n][0], &d1 = acc[m][n][1], &d2 = acc[m][n][2], &d3 = acc[m][n][3];
// //                 uint32_t a0 = frA[1][m][0], a1 = frA[1][m][1], a2 = frA[1][m][2], a3 = frA[1][m][3];
// //                 uint32_t b0 = frB[1][n][0], b1 = frB[1][n][1];
// //                 MMA_TF32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
// //                 if (m == 0 && kt + 1 < ktcnt) fetch_b(frB[0][n], 0, n, smem + rs * Cfg::STAGE_SIZE + Cfg::AS_SIZE);
// //             }
// //             if (kt + 1 < ktcnt) fetch_a(frA[0][m], 0, m, smem + rs * Cfg::STAGE_SIZE);
// //         }
// //     }

// //     float* dC = C + (long long)batch * strideC;
// //     const int ge = lane >> 2, te = lane & 3;
// //     #pragma unroll
// //     for (int m = 0; m < Cfg::MMA_M; m++) {
// //         const int r0 = by * Cfg::BM + wy * Cfg::WARP_TILE_M + m * 16 + ge, r8 = r0 + 8;
// //         if (r0 >= M) continue;
// //         #pragma unroll
// //         for (int n = 0; n < Cfg::MMA_N; n++) {
// //             const int c0 = bx * Cfg::BN + wx * Cfg::WARP_TILE_N + n * 8 + te * 2;
// //             if (c0 >= N) continue;
// //             auto st = [&](int r, float v0, float v1) {
// //                 if (r >= M) return;
// //                 float* p = dC + (long long)r * ldc + c0;
// //                 float f0 = alpha * v0, f1 = alpha * v1;
// //                 if (bias) {
// //                     if (bias_stride == (long long)N) {
// //                         f0 += bias[c0]; if (c0 + 1 < N) f1 += bias[c0 + 1];
// //                     } else if (bias_stride == (long long)M * N) {
// //                         const float* pb = bias + (long long)(batch % M) * N + c0;
// //                         f0 += pb[0]; if (c0 + 1 < N) f1 += pb[1];
// //                     } else if (bias_stride == 1) {
// //                         f0 += bias[0]; f1 += bias[0];
// //                     }
// //                 }
// //                 if constexpr (IsSplitK) {
// //                     atomicAdd(p, f0); if (c0 + 1 < N) atomicAdd(p + 1, f1);
// //                 } else {
// //                     if (beta != 0.f) { 
// //                         if (c0 + 1 < N) { float2 o = *(float2*)p; f0 += beta * o.x; f1 += beta * o.y; } 
// //                         else f0 += beta * p[0]; 
// //                     }
// //                     if (c0 + 1 < N) *(float2*)p = make_float2(f0, f1); else p[0] = f0;
// //                 }
// //             };
// //             st(r0, acc[m][n][0], acc[m][n][1]); st(r8, acc[m][n][2], acc[m][n][3]);
// //         }
// //     }
// // }

// // template <typename Config, bool IsAligned, int SplitK, SgemmLayout Layout>
// // __global__ void sgemm_sm89_template_kernel( // Compatibility wrapper for old name if needed
// //     int M, int N, int K, float alpha, const float* A, int lda, long long sA, const float* B, int ldb, long long sB, float beta, float* C, int ldc, long long sC, int bc)
// // {
// //     sgemm_sm89_kernel<Config, IsAligned, SplitK, Layout><<<1, 1, 0, 0>>>(M, N, K, alpha, A, lda, sA, B, ldb, sB, beta, C, ldc, sC, bc, SplitK);
// // }

// // template <typename Config>
// // __global__ void sgemm_sm89_scale_kernel(float* C, float beta, int M, int N, int ldc, long long strideC, int batchCount) {
// //     const int r = blockIdx.y * blockDim.y + threadIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x, b = blockIdx.z;
// //     if (r < M && c < N && b < batchCount) { float* dst = &C[(long long)b * strideC + (long long)r * ldc + c]; *dst = (beta == 0.f) ? 0.f : (*dst * beta); }
// // }


























































// // #pragma once

// // #include <cuda_runtime.h>
// // #include <stdint.h>
// // #include <stdio.h>

// // /**
// //  * Optimized SM89 SGEMM Core — High Performance & Accuracy
// //  *
// //  * ldmatrix-enabled loads for A and B in NN and NT layouts.
// //  */

// // #ifndef MYCUBLAS_LAYOUT_ENUM
// // #define MYCUBLAS_LAYOUT_ENUM
// // enum class SgemmLayout { NT, TN, NN };
// // #endif

// // // ---------------------------------------------------------------------------
// // // PTX helpers
// // // ---------------------------------------------------------------------------
// // #ifndef MMA_TF32
// // #define MMA_TF32(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1)                           \
// //     asm volatile(                                                              \
// //         "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "                \
// //         "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"                  \
// //         : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)                                \
// //         : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1))
// // #endif

// // #ifndef LDSM_X4
// // #define LDSM_X4(r0,r1,r2,r3,addr)                                            \
// //     asm volatile(                                                              \
// //         "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];"      \
// //         : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(addr))
// // #endif

// // #ifndef LDSM_X2
// // #define LDSM_X2(r0,r1,addr)                                                  \
// //     asm volatile(                                                              \
// //         "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1},[%2];"            \
// //         : "=r"(r0),"=r"(r1) : "r"(addr))
// // #endif

// // #ifndef LDSM_X2_TRANS
// // #define LDSM_X2_TRANS(r0,r1,addr)                                            \
// //     asm volatile(                                                              \
// //         "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1},[%2];"      \
// //         : "=r"(r0),"=r"(r1) : "r"(addr))
// // #endif

// // #ifndef LDSM_X1
// // #define LDSM_X1(r0,r1,addr)                                                  \
// //     asm volatile(                                                              \
// //         "ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0,%1},[%2];"            \
// //         : "=r"(r0),"=r"(r1) : "r"(addr))
// // #endif

// // #ifndef CP_ASYNC_CG
// // #define CP_ASYNC_CG(dst, src)                                                 \
// //     asm volatile("cp.async.cg.shared.global [%0],[%1],16;" :: "r"(dst),"l"(src))
// // #endif

// // // ---------------------------------------------------------------------------
// // // Tile config
// // // ---------------------------------------------------------------------------
// // template <int BM_, int BN_, int BK_, int STAGES_, int THREADS_>
// // struct SgemmTileConfigSM89 {
// //     static constexpr int BM      = BM_;
// //     static constexpr int BN      = BN_;
// //     static constexpr int BK      = BK_;
// //     static constexpr int STAGES  = STAGES_;
// //     static constexpr int THREADS = THREADS_;

// //     static constexpr int AS_SIZE    = BM * BK;
// //     static constexpr int BS_SIZE    = BK * BN;
// //     static constexpr int STAGE_SIZE = AS_SIZE + BS_SIZE;

// //     static constexpr int SMEM_BYTES = STAGES * (BM + BN) * BK * sizeof(float);
// //     static constexpr int MAX_OCC    = (BM * BN >= 256 * 128) ? 1 : 2;

// //     static constexpr int WARPS_TOTAL = THREADS / 32;
// //     static constexpr int WARPS_N     = (BN >= 64 && WARPS_TOTAL > 1)
// //                                          ? ((BN / 64 < WARPS_TOTAL / 2) ? BN / 64 : WARPS_TOTAL / 2)
// //                                          : 1;
// //     static constexpr int WARPS_M     = (WARPS_TOTAL > 1) ? (WARPS_TOTAL / WARPS_N) : 1;

// //     static constexpr int WARP_TILE_M = BM / WARPS_M;
// //     static constexpr int WARP_TILE_N = BN / WARPS_N;

// //     static constexpr int MMA_M = WARP_TILE_M / 16;
// //     static constexpr int MMA_N = WARP_TILE_N / 8;
// // };

// // // ---------------------------------------------------------------------------
// // // Kernel
// // // ---------------------------------------------------------------------------
// // template <typename Cfg, bool IsAligned, bool IsSplitK, SgemmLayout Layout>
// // __global__ void __launch_bounds__(Cfg::THREADS, Cfg::MAX_OCC)
// // sgemm_sm89_kernel(
// //     int M, int N, int K, float alpha,
// //     const float* __restrict__ A, int lda, long long strideA,
// //     const float* __restrict__ B, int ldb, long long strideB,
// //     float beta,
// //     float* __restrict__ C, int ldc, long long strideC,
// //     const float* __restrict__ bias, long long bias_stride,
// //     int batchCount, int splitK)
// // {
// //     const int batch = blockIdx.z / splitK;
// //     const int sk_idx = blockIdx.z % splitK;
// //     if (batch >= batchCount) return;

// //     // Robust CTA swizzle for L2 locality (8-wide strips)
// //     const int sw = 8;
// //     const int grid_x = gridDim.x, grid_y = gridDim.y;
// //     const int block_idx = blockIdx.y * grid_x + blockIdx.x;

// //     const int num_blocks_per_strip = sw * grid_y;
// //     const int strip_idx = block_idx / num_blocks_per_strip;
// //     const int strip_off = block_idx % num_blocks_per_strip;
// //     const int actual_sw = min(sw, grid_x - strip_idx * sw);

// //     const int bx = strip_idx * sw + (strip_off % actual_sw);
// //     const int by = strip_off / actual_sw;

// //     if (bx >= grid_x || by >= grid_y) return;

// //     const int tid  = threadIdx.x;
// //     const int lane = tid & 31, wid = tid >> 5;
// //     const int wy   = wid / Cfg::WARPS_N, wx = wid % Cfg::WARPS_N;

// //     extern __shared__ float smem[];

// //     float acc[Cfg::MMA_M][Cfg::MMA_N][4];
// //     #pragma unroll
// //     for (int i = 0; i < Cfg::MMA_M; i++)
// //         for (int j = 0; j < Cfg::MMA_N; j++)
// //             acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

// //     const int k_chunk = ((K + 15) / 16 + splitK - 1) / splitK * 16;
// //     const int k_start = sk_idx * k_chunk;
// //     const int k_end   = min(K, k_start + k_chunk);
// //     const int ktcnt   = (k_end - k_start + 15) / 16;

// //     const float* A_base = A + (long long)batch * strideA;
// //     const float* B_base = B + (long long)batch * strideB;

// //     // Load params
// //     constexpr int A_STRIDE = (Layout == SgemmLayout::TN) ? Cfg::BM : Cfg::BK;
// //     constexpr int A_TPR    = A_STRIDE / 4;
// //     constexpr int A_RPL    = Cfg::THREADS / A_TPR;
// //     constexpr int A_ITERS  = (Layout == SgemmLayout::TN) ? (Cfg::BK / A_RPL) : (Cfg::BM / A_RPL);
// //     const int a_row0 = tid / A_TPR, a_col0 = (tid % A_TPR) * 4;

// //     uint32_t sm_a_off[A_ITERS];
// //     #pragma unroll
// //     for (int i = 0; i < A_ITERS; i++) {
// //         const int r = a_row0 + i * A_RPL;
// //         if constexpr (Layout == SgemmLayout::TN) sm_a_off[i] = r * Cfg::BM + (a_col0 ^ ((r & 7) << 3));
// //         else sm_a_off[i] = r * Cfg::BK + (a_col0 ^ ((r & 3) << 2)); // BK=16 swizzle
// //     }

// //     constexpr int B_STRIDE = (Layout == SgemmLayout::NT) ? Cfg::BK : Cfg::BN;
// //     constexpr int B_TPR    = B_STRIDE / 4;
// //     constexpr int B_RPL    = Cfg::THREADS / B_TPR;
// //     constexpr int B_ITERS  = (Layout == SgemmLayout::NT) ? (Cfg::BN / B_RPL) : (Cfg::BK / B_RPL);
// //     const int b_row0 = tid / B_TPR, b_col0 = (tid % B_TPR) * 4;

// //     uint32_t sm_b_off[B_ITERS];
// //     #pragma unroll
// //     for (int i = 0; i < B_ITERS; i++) {
// //         const int r = b_row0 + i * B_RPL;
// //         if constexpr (Layout == SgemmLayout::NT) sm_b_off[i] = r * Cfg::BK + (b_col0 ^ ((r & 3) << 2));
// //         else sm_b_off[i] = r * Cfg::BN + (b_col0 ^ ((r & 7) << 2)); // BN=128 swizzle (8 row)
// //     }

// //     auto load_to_smem = [&](int stage, int ko) {
// //         float* As = smem + stage * Cfg::STAGE_SIZE;
// //         float* Bs = As   + Cfg::AS_SIZE;
// //         #pragma unroll
// //         for (int i = 0; i < A_ITERS; i++) {
// //             const int r = a_row0 + i * A_RPL, gm = (Layout == SgemmLayout::TN) ? (by * Cfg::BM + a_col0) : (ko + a_col0);
// //             const int gr = (Layout == SgemmLayout::TN) ? (ko + r) : (by * Cfg::BM + r);
// //             if (gr < M || (Layout == SgemmLayout::TN && gr < K)) { // Bounds check simplified
// //                 uint32_t sm = (uint32_t)__cvta_generic_to_shared(As + sm_a_off[i]);
// //                 const float* g = A_base + (long long)gr * lda + gm;
// //                 if (IsAligned && ((Layout == SgemmLayout::TN) ? (gm+3 < M) : (gm+3 < K))) CP_ASYNC_CG(sm, g);
// //                 else {
// //                     float4 val = {0.f,0.f,0.f,0.f};
// //                     const int lim = (Layout == SgemmLayout::TN) ? M : K;
// //                     if (gm < lim) val.x = g[0]; if (gm+1 < lim) val.y = g[1]; if (gm+2 < lim) val.z = g[2]; if (gm+3 < lim) val.w = g[3];
// //                     *(float4*)(As + sm_a_off[i]) = val;
// //                 }
// //             }
// //         }
// //         #pragma unroll
// //         for (int i = 0; i < B_ITERS; i++) {
// //             const int r = b_row0 + i * B_RPL, gn = (Layout == SgemmLayout::NT) ? (bx * Cfg::BN + r) : (bx * Cfg::BN + b_col0);
// //             const int gk = (Layout == SgemmLayout::NT) ? (ko + b_col0) : (ko + r);
// //             const int row_g = (Layout == SgemmLayout::NT) ? gn : gk, col_g = (Layout == SgemmLayout::NT) ? gk : gn;
// //             uint32_t sm = (uint32_t)__cvta_generic_to_shared(Bs + sm_b_off[i]);
// //             const float* g = B_base + (long long)row_g * ldb + col_g;
// //             const int lim = (Layout == SgemmLayout::NT) ? K : N;
// //             if ((Layout == SgemmLayout::NT ? gn < N : gk < K) && IsAligned && col_g + 3 < lim) CP_ASYNC_CG(sm, g);
// //             else if (Layout == SgemmLayout::NT ? gn < N : gk < K) {
// //                 float4 val = {0,0,0,0};
// //                 if (col_g < lim) val.x = g[0]; if (col_g+1 < lim) val.y = g[1]; if (col_g+2 < lim) val.z = g[2]; if (col_g+3 < lim) val.w = g[3];
// //                 *(float4*)(Bs + sm_b_off[i]) = val;
// //             }
// //         }
// //     };

// //     // -----------------------------------------------------------------------
// //     // Register fetch helpers
// //     //
// //     // For Layout == TN / NN, we use ldmatrix.x4 to load A from M-major SMEM
// //     // (NN / NT case) or fall back to manual loads (TN case, since A is K-major
// //     // in SMEM and the TF32-via-ldmatrix trick requires the fast axis to be
// //     // along the K dimension for A).
// //     //
// //     // For Layout == NT, we use ldmatrix.x2.trans to load B from N-major SMEM.
// //     // For NN B (K-major SMEM) the MMA-expected register distribution is
// //     // transposed relative to what ldmatrix produces on K-major data, so we
// //     // keep manual scalar loads for that case.
// //     //
// //     // TF32 layout note for ldmatrix:
// //     //   The 32-bit TF32 word is viewed by ldmatrix as two adjacent b16 halves.
// //     //   For the load to reconstruct the original TF32 value correctly, the two
// //     //   halves must remain in the same thread's register. This holds when the
// //     //   tile's fast dimension (which the two b16 halves span in SMEM) is the
// //     //   same as the fast dimension in the per-thread output, i.e.:
// //     //     - non-trans:  needs fast axis along "col" in SMEM
// //     //     - trans:      needs fast axis along "row" in SMEM (so post-trans
// //     //                   they become "col" of the register view)
// //     //   Our NN/NT A (M-major, fast axis = K along col) → non-trans OK.
// //     //   Our NT B   (N-major, fast axis = K along col) → trans loads pairs
// //     //                   of b16 halves into the same register correctly because
// //     //                   the halves are adjacent in K (col) → trans swaps so
// //     //                   they become the row direction of the tile, but stay
// //     //                   co-located in one register per thread.
// //     // -----------------------------------------------------------------------

// //     auto fetch_a = [&](uint32_t reg[4], int ks, int m, const float* As_st) {
// //         if constexpr (Layout == SgemmLayout::TN) {
// //             // A is K-major in SMEM (rows=K, cols=M). Manual scalar loads.
// //             const int m_idx = wy * Cfg::WARP_TILE_M + m * 16 + (lane / 4);
// //             const int k0 = ks + (lane % 4), k4 = k0 + 4;
// //             auto la = [&](int cur_k, int cur_m) { return *(const uint32_t*)(&As_st[cur_k * Cfg::BM + (cur_m ^ ((cur_k & 7) << 3))]); };
// //             reg[0] = la(k0, m_idx); reg[1] = la(k0, m_idx + 8); reg[2] = la(k4, m_idx); reg[3] = la(k4, m_idx + 8);
// //         } else {
// //             // NN / NT: A is M-major in SMEM (rows=M, cols=K, stride=BK).
// //             // ldmatrix.x4 loads a 16×16 b16 region = 16 M × 8 TF32-K
// //             // covering the A fragment for one MMA tile (ks..ks+7 K-range).
// //             //
// //             // Row addresses per lane (lane = thread-in-warp, 0..31):
// //             //   matrix idx  = lane / 8             (0..3)
// //             //   row_in_mat  = lane & 7             (0..7)
// //             //   M-offset    = warp_M_base + (matrix_idx >= 2 ? 8 : 0) + row_in_mat
// //             //   K-offset    = ks + (matrix_idx & 1) * 4     [in TF32 elems]
// //             //   swizzled_K  = K-offset XOR ((M-offset & 3) << 2)
// //             //
// //             // Output register distribution (as TF32 per thread):
// //             //   d0: (M = warp_M_base + T/4,       K = ks + T%4)
// //             //   d1: (M = warp_M_base + T/4,       K = ks + T%4 + 4)
// //             //   d2: (M = warp_M_base + T/4 + 8,   K = ks + T%4)
// //             //   d3: (M = warp_M_base + T/4 + 8,   K = ks + T%4 + 4)
// //             // MMA A fragment wants: a0,a1,a2,a3 = (T/4,T%4), (T/4+8,T%4),
// //             //                                     (T/4, T%4+4), (T/4+8, T%4+4)
// //             // → a0 = d0, a1 = d2, a2 = d1, a3 = d3.
// //             const int warp_m_base = wy * Cfg::WARP_TILE_M + m * 16;
// //             const int mat         = lane >> 3;            // 0..3
// //             const int row_in_mat  = lane & 7;             // 0..7
// //             const int m_off       = warp_m_base + ((mat >> 1) ? 8 : 0) + row_in_mat;
// //             const int k_raw       = ks + ((mat & 1) ? 4 : 0);
// //             const int k_sw        = k_raw ^ ((m_off & 3) << 2);
// //             const uint32_t addr   = (uint32_t)__cvta_generic_to_shared(
// //                                         &As_st[m_off * Cfg::BK + k_sw]);
// //             uint32_t d0, d1, d2, d3;
// //             LDSM_X4(d0, d1, d2, d3, addr);
// //             reg[0] = d0;  // a0
// //             reg[1] = d2;  // a1
// //             reg[2] = d1;  // a2
// //             reg[3] = d3;  // a3
// //         }
// //     };

// //     auto fetch_b = [&](uint32_t reg[2], int ks, int n, const float* Bs_st) {
// //         if constexpr (Layout == SgemmLayout::NT) {
// //             // B is N-major in SMEM (rows=N, cols=K, stride=BK).
// //             // ldmatrix.x2.trans loads a 8×16 b16 region = 8 N × 8 TF32-K,
// //             // one B fragment per call.
// //             //
// //             // Row addresses (only lanes 0..15 matter):
// //             //   matrix idx = lane / 8        (0 or 1)
// //             //   row_in_mat = lane & 7        (0..7)
// //             //   N-offset   = warp_N_base + row_in_mat
// //             //   K-offset   = (matrix_idx & 1) * 4           [TF32 elems]
// //             //   swizzled_K = (ks + K-offset) XOR ((N-offset & 3) << 2)
// //             //
// //             // Output registers (TF32 per thread):
// //             //   d0: (K = ks + T%4,     N = warp_N_base + T/4)
// //             //   d1: (K = ks + T%4 + 4, N = warp_N_base + T/4)
// //             // MMA B fragment wants exactly this distribution →
// //             //   reg[0] = d0, reg[1] = d1.
// //             const int warp_n_base = wx * Cfg::WARP_TILE_N + n * 8;
// //             const int mat         = (lane >> 3) & 1;       // 0 or 1 (lanes 0..15)
// //             const int row_in_mat  = lane & 7;
// //             const int n_off       = warp_n_base + row_in_mat;
// //             const int k_raw       = ks + (mat ? 4 : 0);
// //             const int k_sw        = k_raw ^ ((n_off & 3) << 2);
// //             const uint32_t addr   = (uint32_t)__cvta_generic_to_shared(
// //                                         &Bs_st[n_off * Cfg::BK + k_sw]);
// //             uint32_t d0, d1;
// //             LDSM_X2_TRANS(d0, d1, addr);
// //             reg[0] = d0;
// //             reg[1] = d1;
// //         } else {
// //             // TN / NN: B is K-major in SMEM (rows=K, cols=N, stride=BN).
// //             // ldmatrix would require b16-level transpose which corrupts TF32
// //             // for this storage layout, so we use manual scalar loads.
// //             const int lr0 = ks + (lane % 4), lr4 = lr0 + 4;
// //             const int lc  = wx * Cfg::WARP_TILE_N + n * 8 + (lane / 4);
// //             auto gb = [&](int r, int c) {
// //                 return *(const uint32_t*)(&Bs_st[r * Cfg::BN + (c ^ ((r & 7) << 2))]);
// //             };
// //             reg[0] = gb(lr0, lc);
// //             reg[1] = gb(lr4, lc);
// //         }
// //     };

// //     // Warmup
// //     constexpr int PREFETCH = Cfg::STAGES - 1;
// //     int rs = 0, ws = 0;
// //     #pragma unroll
// //     for (int i = 0; i < PREFETCH; i++) { if (i < ktcnt) load_to_smem(ws, k_start + i * Cfg::BK); asm volatile("cp.async.commit_group;"); ws = (ws + 1) % Cfg::STAGES; }
// //     asm volatile("cp.async.wait_group %0;" :: "n"(Cfg::STAGES - 2)); __syncthreads();

// //     uint32_t frA[2][Cfg::MMA_M][4], frB[2][Cfg::MMA_N][2];
// //     #pragma unroll
// //     for (int m = 0; m < Cfg::MMA_M; m++) fetch_a(frA[0][m], 0, m, smem + rs * Cfg::STAGE_SIZE);
// //     #pragma unroll
// //     for (int n = 0; n < Cfg::MMA_N; n++) fetch_b(frB[0][n], 0, n, smem + rs * Cfg::STAGE_SIZE + Cfg::AS_SIZE);

// //     for (int kt = 0; kt < ktcnt; kt++) {
// //         const int kf = k_start + (kt + PREFETCH) * Cfg::BK;
// //         if (kf < k_end) load_to_smem(ws, kf);
// //         asm volatile("cp.async.commit_group;");

// //         #pragma unroll
// //         for (int m = 0; m < Cfg::MMA_M; m++) {
// //             #pragma unroll
// //             for (int n = 0; n < Cfg::MMA_N; n++) {
// //                 float &d0 = acc[m][n][0], &d1 = acc[m][n][1], &d2 = acc[m][n][2], &d3 = acc[m][n][3];
// //                 uint32_t a0 = frA[0][m][0], a1 = frA[0][m][1], a2 = frA[0][m][2], a3 = frA[0][m][3];
// //                 uint32_t b0 = frB[0][n][0], b1 = frB[0][n][1];
// //                 MMA_TF32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
// //                 if (m == 0) fetch_b(frB[1][n], 8, n, smem + rs * Cfg::STAGE_SIZE + Cfg::AS_SIZE);
// //             }
// //             fetch_a(frA[1][m], 8, m, smem + rs * Cfg::STAGE_SIZE);
// //         }
// //         asm volatile("cp.async.wait_group %0;" :: "n"(Cfg::STAGES - 2)); __syncthreads();
// //         rs = (rs + 1) % Cfg::STAGES; ws = (ws + 1) % Cfg::STAGES;
// //         #pragma unroll
// //         for (int m = 0; m < Cfg::MMA_M; m++) {
// //             #pragma unroll
// //             for (int n = 0; n < Cfg::MMA_N; n++) {
// //                 float &d0 = acc[m][n][0], &d1 = acc[m][n][1], &d2 = acc[m][n][2], &d3 = acc[m][n][3];
// //                 uint32_t a0 = frA[1][m][0], a1 = frA[1][m][1], a2 = frA[1][m][2], a3 = frA[1][m][3];
// //                 uint32_t b0 = frB[1][n][0], b1 = frB[1][n][1];
// //                 MMA_TF32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
// //                 if (m == 0 && kt + 1 < ktcnt) fetch_b(frB[0][n], 0, n, smem + rs * Cfg::STAGE_SIZE + Cfg::AS_SIZE);
// //             }
// //             if (kt + 1 < ktcnt) fetch_a(frA[0][m], 0, m, smem + rs * Cfg::STAGE_SIZE);
// //         }
// //     }

// //     float* dC = C + (long long)batch * strideC;
// //     const int ge = lane >> 2, te = lane & 3;
// //     #pragma unroll
// //     for (int m = 0; m < Cfg::MMA_M; m++) {
// //         const int r0 = by * Cfg::BM + wy * Cfg::WARP_TILE_M + m * 16 + ge, r8 = r0 + 8;
// //         if (r0 >= M) continue;
// //         #pragma unroll
// //         for (int n = 0; n < Cfg::MMA_N; n++) {
// //             const int c0 = bx * Cfg::BN + wx * Cfg::WARP_TILE_N + n * 8 + te * 2;
// //             if (c0 >= N) continue;
// //             auto st = [&](int r, float v0, float v1) {
// //                 if (r >= M) return;
// //                 float* p = dC + (long long)r * ldc + c0;
// //                 float f0 = alpha * v0, f1 = alpha * v1;
// //                 if (bias) {
// //                     if (bias_stride == (long long)N) {
// //                         f0 += bias[c0]; if (c0 + 1 < N) f1 += bias[c0 + 1];
// //                     } else if (bias_stride == (long long)M * N) {
// //                         const float* pb = bias + (long long)(batch % M) * N + c0;
// //                         f0 += pb[0]; if (c0 + 1 < N) f1 += pb[1];
// //                     } else if (bias_stride == 1) {
// //                         f0 += bias[0]; f1 += bias[0];
// //                     }
// //                 }
// //                 if constexpr (IsSplitK) {
// //                     atomicAdd(p, f0); if (c0 + 1 < N) atomicAdd(p + 1, f1);
// //                 } else {
// //                     if (beta != 0.f) {
// //                         if (c0 + 1 < N) { float2 o = *(float2*)p; f0 += beta * o.x; f1 += beta * o.y; }
// //                         else f0 += beta * p[0];
// //                     }
// //                     if (c0 + 1 < N) *(float2*)p = make_float2(f0, f1); else p[0] = f0;
// //                 }
// //             };
// //             st(r0, acc[m][n][0], acc[m][n][1]); st(r8, acc[m][n][2], acc[m][n][3]);
// //         }
// //     }
// // }

// // template <typename Config, bool IsAligned, int SplitK, SgemmLayout Layout>
// // __global__ void sgemm_sm89_template_kernel( // Compatibility wrapper for old name if needed
// //     int M, int N, int K, float alpha, const float* A, int lda, long long sA, const float* B, int ldb, long long sB, float beta, float* C, int ldc, long long sC, int bc)
// // {
// //     sgemm_sm89_kernel<Config, IsAligned, SplitK, Layout><<<1, 1, 0, 0>>>(M, N, K, alpha, A, lda, sA, B, ldb, sB, beta, C, ldc, sC, bc, SplitK);
// // }

// // template <typename Config>
// // __global__ void sgemm_sm89_scale_kernel(float* C, float beta, int M, int N, int ldc, long long strideC, int batchCount) {
// //     const int r = blockIdx.y * blockDim.y + threadIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x, b = blockIdx.z;
// //     if (r < M && c < N && b < batchCount) { float* dst = &C[(long long)b * strideC + (long long)r * ldc + c]; *dst = (beta == 0.f) ? 0.f : (*dst * beta); }
// // }





























// #pragma once

// #include <cuda_runtime.h>
// #include <stdint.h>
// #include <stdio.h>

// /**
//  * Optimized SM89 SGEMM Core — High Performance & Accuracy
//  *
//  * Changes vs previous revision (full optimization sweep):
//  *   - ldmatrix.x4 for A in NN / NT layouts (M-major SMEM)
//  *   - ldmatrix.x2.trans for B in NT layout (N-major SMEM)
//  *   - Lane-invariant ldmatrix addressing hoisted out of per-m / per-n calls
//  *   - Epilogue: bias-mode dispatch lifted out of the m,n inner loops
//  *   - splitK correctness: bias is applied only by sk_idx == 0. beta*C_old is
//  *     expected to be pre-applied by the companion scale kernel before the
//  *     splitK launch (that kernel already exists in this file); the splitK
//  *     path in the epilogue ignores beta, matching the scale-then-add pattern.
//  *     Previously the splitK path both ignored beta AND applied bias splitK
//  *     times via atomicAdd, making bias incorrect.
//  *   - TN layout bounds check fixed (previously used an `||` that could let
//  *     out-of-range K rows pass a spurious M-bound test)
//  *   - Grid swizzle guarded against actual_sw <= 0 for degenerate grids
//  *   - Accumulator init fully unrolled
//  *
//  * Layout conventions (SMEM):
//  *   - NN / NT : A is M-major (rows=M, cols=K, stride=BK). fast axis = K
//  *   - TN      : A is K-major (rows=K, cols=M, stride=BM). fast axis = M
//  *   - NT      : B is N-major (rows=N, cols=K, stride=BK). fast axis = K
//  *   - NN / TN : B is K-major (rows=K, cols=N, stride=BN). fast axis = N
//  *
//  * ldmatrix / TF32 compatibility (why A-NN/NT and B-NT use ldmatrix but not
//  * the other cases):
//  *   A TF32 32-bit word's two b16 halves are adjacent along the SMEM fast
//  *   axis. Non-trans ldmatrix pairs b16 halves along the tile-col direction;
//  *   .trans pairs them along the tile-row direction. For a thread's register
//  *   to hold a complete TF32 value the two paired halves must correspond to
//  *   one 32-bit word, i.e. the SMEM fast axis must match the ldmatrix pair
//  *   direction. Crossing that rule silently corrupts values, so we use
//  *   ldmatrix only where the layouts agree.
//  */

// #ifndef MYCUBLAS_LAYOUT_ENUM
// #define MYCUBLAS_LAYOUT_ENUM
// enum class SgemmLayout { NT, TN, NN };
// #endif

// // ---------------------------------------------------------------------------
// // PTX helpers
// // ---------------------------------------------------------------------------
// #ifndef MMA_TF32
// #define MMA_TF32(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1)                           \
//     asm volatile(                                                              \
//         "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "                \
//         "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"                  \
//         : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)                                \
//         : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1))
// #endif

// #ifndef LDSM_X4
// #define LDSM_X4(r0,r1,r2,r3,addr)                                            \
//     asm volatile(                                                              \
//         "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];"      \
//         : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(addr))
// #endif

// #ifndef LDSM_X2
// #define LDSM_X2(r0,r1,addr)                                                  \
//     asm volatile(                                                              \
//         "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1},[%2];"            \
//         : "=r"(r0),"=r"(r1) : "r"(addr))
// #endif

// #ifndef LDSM_X2_TRANS
// #define LDSM_X2_TRANS(r0,r1,addr)                                            \
//     asm volatile(                                                              \
//         "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1},[%2];"      \
//         : "=r"(r0),"=r"(r1) : "r"(addr))
// #endif

// #ifndef LDSM_X1
// #define LDSM_X1(r0,r1,addr)                                                  \
//     asm volatile(                                                              \
//         "ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0,%1},[%2];"            \
//         : "=r"(r0),"=r"(r1) : "r"(addr))
// #endif

// #ifndef CP_ASYNC_CG
// #define CP_ASYNC_CG(dst, src)                                                 \
//     asm volatile("cp.async.cg.shared.global [%0],[%1],16;" :: "r"(dst),"l"(src))
// #endif

// // ---------------------------------------------------------------------------
// // Tile config
// // ---------------------------------------------------------------------------
// template <int BM_, int BN_, int BK_, int STAGES_, int THREADS_>
// struct SgemmTileConfigSM89 {
//     static constexpr int BM      = BM_;
//     static constexpr int BN      = BN_;
//     static constexpr int BK      = BK_;
//     static constexpr int STAGES  = STAGES_;
//     static constexpr int THREADS = THREADS_;

//     static constexpr int AS_SIZE    = BM * BK;
//     static constexpr int BS_SIZE    = BK * BN;
//     static constexpr int STAGE_SIZE = AS_SIZE + BS_SIZE;

//     static constexpr int SMEM_BYTES = STAGES * (BM + BN) * BK * sizeof(float);
//     static constexpr int MAX_OCC    = (BM * BN >= 256 * 128) ? 1 : 2;

//     static constexpr int WARPS_TOTAL = THREADS / 32;
//     static constexpr int WARPS_N     = (BN >= 64 && WARPS_TOTAL > 1)
//                                          ? ((BN / 64 < WARPS_TOTAL / 2) ? BN / 64 : WARPS_TOTAL / 2)
//                                          : 1;
//     static constexpr int WARPS_M     = (WARPS_TOTAL > 1) ? (WARPS_TOTAL / WARPS_N) : 1;

//     static constexpr int WARP_TILE_M = BM / WARPS_M;
//     static constexpr int WARP_TILE_N = BN / WARPS_N;

//     static constexpr int MMA_M = WARP_TILE_M / 16;
//     static constexpr int MMA_N = WARP_TILE_N / 8;
// };

// // ---------------------------------------------------------------------------
// // Kernel
// // ---------------------------------------------------------------------------
// template <typename Cfg, bool IsAligned, bool IsSplitK, SgemmLayout Layout>
// __global__ void __launch_bounds__(Cfg::THREADS, Cfg::MAX_OCC)
// sgemm_sm89_kernel(
//     int M, int N, int K, float alpha,
//     const float* __restrict__ A, int lda, long long strideA,
//     const float* __restrict__ B, int ldb, long long strideB,
//     float beta,
//     float* __restrict__ C, int ldc, long long strideC,
//     const float* __restrict__ bias, long long bias_stride,
//     int batchCount, int splitK)
// {
//     const int batch  = blockIdx.z / splitK;
//     const int sk_idx = blockIdx.z % splitK;
//     if (batch >= batchCount) return;

//     // Robust CTA swizzle for L2 locality (8-wide strips).
//     // Guard against degenerate grids where (strip_idx * sw) overshoots grid_x,
//     // which would make `actual_sw <= 0` and break the subsequent modulo.
//     const int sw = 8;
//     const int grid_x = gridDim.x, grid_y = gridDim.y;
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
//     const int wy   = wid / Cfg::WARPS_N, wx = wid % Cfg::WARPS_N;

//     extern __shared__ float smem[];

//     float acc[Cfg::MMA_M][Cfg::MMA_N][4];
//     #pragma unroll
//     for (int i = 0; i < Cfg::MMA_M; i++) {
//         #pragma unroll
//         for (int j = 0; j < Cfg::MMA_N; j++) {
//             acc[i][j][0] = 0.f;
//             acc[i][j][1] = 0.f;
//             acc[i][j][2] = 0.f;
//             acc[i][j][3] = 0.f;
//         }
//     }

//     const int k_chunk = ((K + 15) / 16 + splitK - 1) / splitK * 16;
//     const int k_start = sk_idx * k_chunk;
//     const int k_end   = min(K, k_start + k_chunk);
//     const int ktcnt   = (k_end - k_start + 15) / 16;

//     const float* A_base = A + (long long)batch * strideA;
//     const float* B_base = B + (long long)batch * strideB;

//     // -----------------------------------------------------------------------
//     // Global → shared load setup
//     // -----------------------------------------------------------------------
//     constexpr int A_STRIDE = (Layout == SgemmLayout::TN) ? Cfg::BM : Cfg::BK;
//     constexpr int A_TPR    = A_STRIDE / 4;
//     constexpr int A_RPL    = Cfg::THREADS / A_TPR;
//     constexpr int A_ITERS  = (Layout == SgemmLayout::TN) ? (Cfg::BK / A_RPL) : (Cfg::BM / A_RPL);
//     const int a_row0 = tid / A_TPR, a_col0 = (tid % A_TPR) * 4;

//     uint32_t sm_a_off[A_ITERS];
//     #pragma unroll
//     for (int i = 0; i < A_ITERS; i++) {
//         const int r = a_row0 + i * A_RPL;
//         if constexpr (Layout == SgemmLayout::TN) sm_a_off[i] = r * Cfg::BM + (a_col0 ^ ((r & 7) << 3));
//         else                                     sm_a_off[i] = r * Cfg::BK + (a_col0 ^ ((r & 3) << 2));
//     }

//     constexpr int B_STRIDE = (Layout == SgemmLayout::NT) ? Cfg::BK : Cfg::BN;
//     constexpr int B_TPR    = B_STRIDE / 4;
//     constexpr int B_RPL    = Cfg::THREADS / B_TPR;
//     constexpr int B_ITERS  = (Layout == SgemmLayout::NT) ? (Cfg::BN / B_RPL) : (Cfg::BK / B_RPL);
//     const int b_row0 = tid / B_TPR, b_col0 = (tid % B_TPR) * 4;

//     uint32_t sm_b_off[B_ITERS];
//     #pragma unroll
//     for (int i = 0; i < B_ITERS; i++) {
//         const int r = b_row0 + i * B_RPL;
//         if constexpr (Layout == SgemmLayout::NT) sm_b_off[i] = r * Cfg::BK + (b_col0 ^ ((r & 3) << 2));
//         else                                     sm_b_off[i] = r * Cfg::BN + (b_col0 ^ ((r & 7) << 2));
//     }

//     auto load_to_smem = [&](int stage, int ko) {
//         float* As = smem + stage * Cfg::STAGE_SIZE;
//         float* Bs = As   + Cfg::AS_SIZE;
//         #pragma unroll
//         for (int i = 0; i < A_ITERS; i++) {
//             const int r  = a_row0 + i * A_RPL;
//             const int gm = (Layout == SgemmLayout::TN) ? (by * Cfg::BM + a_col0) : (ko + a_col0);
//             const int gr = (Layout == SgemmLayout::TN) ? (ko + r)                : (by * Cfg::BM + r);
//             // Bounds check on the row dimension (K for TN, M otherwise).
//             const bool row_ok = (Layout == SgemmLayout::TN) ? (gr < K) : (gr < M);
//             if (row_ok) {
//                 uint32_t sm = (uint32_t)__cvta_generic_to_shared(As + sm_a_off[i]);
//                 const float* g = A_base + (long long)gr * lda + gm;
//                 const int lim = (Layout == SgemmLayout::TN) ? M : K;
//                 if (IsAligned && gm + 3 < lim) {
//                     CP_ASYNC_CG(sm, g);
//                 } else {
//                     float4 val = {0.f, 0.f, 0.f, 0.f};
//                     if (gm     < lim) val.x = g[0];
//                     if (gm + 1 < lim) val.y = g[1];
//                     if (gm + 2 < lim) val.z = g[2];
//                     if (gm + 3 < lim) val.w = g[3];
//                     *(float4*)(As + sm_a_off[i]) = val;
//                 }
//             } else {
//                 *(float4*)(As + sm_a_off[i]) = {0.f, 0.f, 0.f, 0.f};
//             }
//         }
//         #pragma unroll
//         for (int i = 0; i < B_ITERS; i++) {
//             const int r  = b_row0 + i * B_RPL;
//             const int gn = (Layout == SgemmLayout::NT) ? (bx * Cfg::BN + r) : (bx * Cfg::BN + b_col0);
//             const int gk = (Layout == SgemmLayout::NT) ? (ko + b_col0)      : (ko + r);
//             const int row_g = (Layout == SgemmLayout::NT) ? gn : gk;
//             const int col_g = (Layout == SgemmLayout::NT) ? gk : gn;
//             const int lim = (Layout == SgemmLayout::NT) ? K : N;
//             const bool row_ok = (Layout == SgemmLayout::NT) ? (gn < N) : (gk < K);
//             if (row_ok) {
//                 uint32_t sm = (uint32_t)__cvta_generic_to_shared(Bs + sm_b_off[i]);
//                 const float* g = B_base + (long long)row_g * ldb + col_g;
//                 if (IsAligned && col_g + 3 < lim) {
//                     CP_ASYNC_CG(sm, g);
//                 } else {
//                     float4 val = {0.f, 0.f, 0.f, 0.f};
//                     if (col_g     < lim) val.x = g[0];
//                     if (col_g + 1 < lim) val.y = g[1];
//                     if (col_g + 2 < lim) val.z = g[2];
//                     if (col_g + 3 < lim) val.w = g[3];
//                     *(float4*)(Bs + sm_b_off[i]) = val;
//                 }
//             } else {
//                 *(float4*)(Bs + sm_b_off[i]) = {0.f, 0.f, 0.f, 0.f};
//             }
//         }
//     };

//     // -----------------------------------------------------------------------
//     // Register fragment loads
//     //
//     // Lane-dependent address terms are computed once outside the k-loop and
//     // combined with per-call (m, n, ks) offsets at the fetch site. The XOR
//     // swizzle of the K/N-column offset uses only the low 2 bits of the row
//     // offset; WARP_TILE_M is a multiple of 16 and m*16 is a multiple of 16,
//     // so neither perturbs the low 2 bits of M — the swizzle can be folded in
//     // as a lane-invariant constant.
//     // -----------------------------------------------------------------------

//     // --- A: lane-invariant parts for NN/NT ldmatrix.x4 ---
//     const int a_ld_row  = ((lane >> 3) >> 1 ? 8 : 0) + (lane & 7);
//     const int a_ld_kbit = ((lane >> 3) & 1) ? 4 : 0;
//     const int a_sw_xor  = (a_ld_row & 3) << 2;
//     const int a_row_abs_base = wy * Cfg::WARP_TILE_M + a_ld_row;  // add m*16 at call site

//     // --- B: lane-invariant parts for NT ldmatrix.x2.trans ---
//     const int b_ld_row_nt  = lane & 7;
//     const int b_ld_kbit_nt = ((lane >> 3) & 1) ? 4 : 0;
//     const int b_sw_xor_nt  = (b_ld_row_nt & 3) << 2;
//     const int b_row_abs_base_nt = wx * Cfg::WARP_TILE_N + b_ld_row_nt;

//     // --- A: lane-invariant parts for TN scalar loads ---
//     const int a_tn_m_base = wy * Cfg::WARP_TILE_M + (lane >> 2);  // + m*16
//     const int a_tn_k_lane = lane & 3;                              // + ks

//     // --- B: lane-invariant parts for NN/TN scalar loads ---
//     const int b_nn_k_lane = lane & 3;                              // + ks
//     const int b_nn_n_base = wx * Cfg::WARP_TILE_N + (lane >> 2);   // + n*8

//     auto fetch_a = [&](uint32_t reg[4], int ks, int m, const float* As_st) {
//         if constexpr (Layout == SgemmLayout::TN) {
//             // TN: A is K-major in SMEM. Manual scalar loads — register
//             // distribution for ldmatrix variants on K-major TF32 storage
//             // cannot match the MMA A-fragment layout without value corruption.
//             const int m_idx = a_tn_m_base + m * 16;
//             const int k0 = ks + a_tn_k_lane, k4 = k0 + 4;
//             auto la = [&](int cur_k, int cur_m) {
//                 return *(const uint32_t*)(&As_st[cur_k * Cfg::BM + (cur_m ^ ((cur_k & 7) << 3))]);
//             };
//             reg[0] = la(k0, m_idx);
//             reg[1] = la(k0, m_idx + 8);
//             reg[2] = la(k4, m_idx);
//             reg[3] = la(k4, m_idx + 8);
//         } else {
//             // NN / NT: A is M-major in SMEM. ldmatrix.x4 loads one A fragment
//             // per call. Register remap (d0, d1, d2, d3) → (a0, a2, a1, a3).
//             const int m_off = a_row_abs_base + m * 16;
//             const int k_sw  = (ks + a_ld_kbit) ^ a_sw_xor;
//             const uint32_t addr = (uint32_t)__cvta_generic_to_shared(
//                                       &As_st[m_off * Cfg::BK + k_sw]);
//             uint32_t d0, d1, d2, d3;
//             LDSM_X4(d0, d1, d2, d3, addr);
//             reg[0] = d0;  // a0 = (T/4,   T%4)
//             reg[1] = d2;  // a1 = (T/4+8, T%4)
//             reg[2] = d1;  // a2 = (T/4,   T%4+4)
//             reg[3] = d3;  // a3 = (T/4+8, T%4+4)
//         }
//     };

//     auto fetch_b = [&](uint32_t reg[2], int ks, int n, const float* Bs_st) {
//         if constexpr (Layout == SgemmLayout::NT) {
//             // NT: B is N-major in SMEM. ldmatrix.x2.trans delivers the exact
//             // MMA B-fragment distribution (reg[0]=d0, reg[1]=d1).
//             const int n_off = b_row_abs_base_nt + n * 8;
//             const int k_sw  = (ks + b_ld_kbit_nt) ^ b_sw_xor_nt;
//             const uint32_t addr = (uint32_t)__cvta_generic_to_shared(
//                                       &Bs_st[n_off * Cfg::BK + k_sw]);
//             uint32_t d0, d1;
//             LDSM_X2_TRANS(d0, d1, addr);
//             reg[0] = d0;
//             reg[1] = d1;
//         } else {
//             // NN / TN: B is K-major in SMEM. Manual scalar loads.
//             const int lr0 = ks + b_nn_k_lane, lr4 = lr0 + 4;
//             const int lc  = b_nn_n_base + n * 8;
//             auto gb = [&](int r, int c) {
//                 return *(const uint32_t*)(&Bs_st[r * Cfg::BN + (c ^ ((r & 7) << 2))]);
//             };
//             reg[0] = gb(lr0, lc);
//             reg[1] = gb(lr4, lc);
//         }
//     };

//     // -----------------------------------------------------------------------
//     // Pipeline warmup
//     // -----------------------------------------------------------------------
//     constexpr int PREFETCH = Cfg::STAGES - 1;
//     int rs = 0, ws = 0;
//     #pragma unroll
//     for (int i = 0; i < PREFETCH; i++) {
//         if (i < ktcnt) load_to_smem(ws, k_start + i * Cfg::BK);
//         asm volatile("cp.async.commit_group;");
//         ws = (ws + 1) % Cfg::STAGES;
//     }
//     asm volatile("cp.async.wait_group %0;" :: "n"(Cfg::STAGES - 2));
//     __syncthreads();

//     uint32_t frA[2][Cfg::MMA_M][4], frB[2][Cfg::MMA_N][2];
//     #pragma unroll
//     for (int m = 0; m < Cfg::MMA_M; m++) fetch_a(frA[0][m], 0, m, smem + rs * Cfg::STAGE_SIZE);
//     #pragma unroll
//     for (int n = 0; n < Cfg::MMA_N; n++) fetch_b(frB[0][n], 0, n, smem + rs * Cfg::STAGE_SIZE + Cfg::AS_SIZE);

//     // -----------------------------------------------------------------------
//     // Main K loop
//     // -----------------------------------------------------------------------
//     for (int kt = 0; kt < ktcnt; kt++) {
//         const int kf = k_start + (kt + PREFETCH) * Cfg::BK;
//         if (kf < k_end) load_to_smem(ws, kf);
//         asm volatile("cp.async.commit_group;");

//         // First inner K-step (ks=0): consume frA[0]/frB[0], prefetch frA[1]/frB[1]
//         #pragma unroll
//         for (int m = 0; m < Cfg::MMA_M; m++) {
//             #pragma unroll
//             for (int n = 0; n < Cfg::MMA_N; n++) {
//                 float &d0 = acc[m][n][0], &d1 = acc[m][n][1];
//                 float &d2 = acc[m][n][2], &d3 = acc[m][n][3];
//                 uint32_t a0 = frA[0][m][0], a1 = frA[0][m][1];
//                 uint32_t a2 = frA[0][m][2], a3 = frA[0][m][3];
//                 uint32_t b0 = frB[0][n][0], b1 = frB[0][n][1];
//                 MMA_TF32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
//                 if (m == 0) fetch_b(frB[1][n], 8, n, smem + rs * Cfg::STAGE_SIZE + Cfg::AS_SIZE);
//             }
//             fetch_a(frA[1][m], 8, m, smem + rs * Cfg::STAGE_SIZE);
//         }

//         asm volatile("cp.async.wait_group %0;" :: "n"(Cfg::STAGES - 2));
//         __syncthreads();
//         rs = (rs + 1) % Cfg::STAGES;
//         ws = (ws + 1) % Cfg::STAGES;

//         // Second inner K-step (ks=8): consume frA[1]/frB[1], prefetch for next kt
//         #pragma unroll
//         for (int m = 0; m < Cfg::MMA_M; m++) {
//             #pragma unroll
//             for (int n = 0; n < Cfg::MMA_N; n++) {
//                 float &d0 = acc[m][n][0], &d1 = acc[m][n][1];
//                 float &d2 = acc[m][n][2], &d3 = acc[m][n][3];
//                 uint32_t a0 = frA[1][m][0], a1 = frA[1][m][1];
//                 uint32_t a2 = frA[1][m][2], a3 = frA[1][m][3];
//                 uint32_t b0 = frB[1][n][0], b1 = frB[1][n][1];
//                 MMA_TF32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
//                 if (m == 0 && kt + 1 < ktcnt) fetch_b(frB[0][n], 0, n, smem + rs * Cfg::STAGE_SIZE + Cfg::AS_SIZE);
//             }
//             if (kt + 1 < ktcnt) fetch_a(frA[0][m], 0, m, smem + rs * Cfg::STAGE_SIZE);
//         }
//     }

//     // -----------------------------------------------------------------------
//     // Epilogue
//     //
//     // Non-splitK path: apply alpha*AB + beta*C_old + bias directly.
//     // SplitK path: beta*C_old is expected to be pre-applied to C by the
//     //   companion scale kernel before the splitK launch, so we only
//     //   atomicAdd alpha*partial + bias_from_sk0.  Bias is added exactly once
//     //   by the sk_idx == 0 block.
//     // -----------------------------------------------------------------------
//     float* dC = C + (long long)batch * strideC;
//     const int ge = lane >> 2, te = lane & 3;

//     // Classify bias mode: 0 = none, 1 = scalar, 2 = per-col (length N),
//     //                     3 = per-(batch)(N) (length M*N, batch-major).
//     const bool bias_active = bias && (!IsSplitK || sk_idx == 0);
//     const int bias_mode = !bias_active                          ? 0
//                         : (bias_stride == 1)                    ? 1
//                         : (bias_stride == (long long)N)         ? 2
//                         : (bias_stride == (long long)M * N)     ? 3
//                         : 0;
//     const float  bias_scalar = (bias_mode == 1) ? bias[0] : 0.f;
//     const float* bias_matrix_row = (bias_mode == 3) ? (bias + (long long)(batch % M) * N) : nullptr;

//     #pragma unroll
//     for (int m = 0; m < Cfg::MMA_M; m++) {
//         const int r0 = by * Cfg::BM + wy * Cfg::WARP_TILE_M + m * 16 + ge;
//         const int r8 = r0 + 8;
//         if (r0 >= M) continue;

//         #pragma unroll
//         for (int n = 0; n < Cfg::MMA_N; n++) {
//             const int c0 = bx * Cfg::BN + wx * Cfg::WARP_TILE_N + n * 8 + te * 2;
//             if (c0 >= N) continue;
//             const int c1 = c0 + 1;

//             float bv0 = 0.f, bv1 = 0.f;
//             switch (bias_mode) {
//                 case 1: bv0 = bias_scalar; bv1 = bias_scalar; break;
//                 case 2:
//                     bv0 = bias[c0];
//                     if (c1 < N) bv1 = bias[c1];
//                     break;
//                 case 3:
//                     bv0 = bias_matrix_row[c0];
//                     if (c1 < N) bv1 = bias_matrix_row[c1];
//                     break;
//                 default: break;
//             }

//             auto st = [&](int r, float v0, float v1) {
//                 if (r >= M) return;
//                 float* p = dC + (long long)r * ldc + c0;
//                 float f0 = alpha * v0 + bv0;
//                 float f1 = alpha * v1 + bv1;

//                 if constexpr (IsSplitK) {
//                     // beta*C_old handled externally by the scale kernel.
//                     atomicAdd(p, f0);
//                     if (c1 < N) atomicAdd(p + 1, f1);
//                 } else {
//                     if (beta != 0.f) {
//                         if (c1 < N) {
//                             float2 o = *(float2*)p;
//                             f0 += beta * o.x;
//                             f1 += beta * o.y;
//                         } else {
//                             f0 += beta * p[0];
//                         }
//                     }
//                     if (c1 < N) *(float2*)p = make_float2(f0, f1);
//                     else        p[0]       = f0;
//                 }
//             };

//             st(r0, acc[m][n][0], acc[m][n][1]);
//             st(r8, acc[m][n][2], acc[m][n][3]);
//         }
//     }
// }

// template <typename Config, bool IsAligned, int SplitK, SgemmLayout Layout>
// __global__ void sgemm_sm89_template_kernel( // Compatibility wrapper for old name if needed
//     int M, int N, int K, float alpha, const float* A, int lda, long long sA, const float* B, int ldb, long long sB, float beta, float* C, int ldc, long long sC, int bc)
// {
//     sgemm_sm89_kernel<Config, IsAligned, SplitK, Layout><<<1, 1, 0, 0>>>(M, N, K, alpha, A, lda, sA, B, ldb, sB, beta, C, ldc, sC, bc, SplitK);
// }

// template <typename Config>
// __global__ void sgemm_sm89_scale_kernel(float* C, float beta, int M, int N, int ldc, long long strideC, int batchCount) {
//     const int r = blockIdx.y * blockDim.y + threadIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x, b = blockIdx.z;
//     if (r < M && c < N && b < batchCount) { float* dst = &C[(long long)b * strideC + (long long)r * ldc + c]; *dst = (beta == 0.f) ? 0.f : (*dst * beta); }
// }








































#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

/**
 * Optimized SM89 SGEMM Core — High Performance & Accuracy
 */

#ifndef MYCUBLAS_LAYOUT_ENUM
#define MYCUBLAS_LAYOUT_ENUM
enum class SgemmLayout { NT, TN, NN };
#endif

// ---------------------------------------------------------------------------
// PTX helpers
// ---------------------------------------------------------------------------
#ifndef MMA_TF32
#define MMA_TF32(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1)                           \
    asm volatile(                                                              \
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "                \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"                  \
        : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)                                \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1))
#endif

#ifndef LDSM_X4
#define LDSM_X4(r0,r1,r2,r3,addr)                                            \
    asm volatile(                                                              \
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];"      \
        : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(addr))
#endif

#ifndef LDSM_X2
#define LDSM_X2(r0,r1,addr)                                                  \
    asm volatile(                                                              \
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1},[%2];"            \
        : "=r"(r0),"=r"(r1) : "r"(addr))
#endif

#ifndef LDSM_X1
#define LDSM_X1(r0,r1,addr)                                                  \
    asm volatile(                                                              \
        "ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0,%1},[%2];"            \
        : "=r"(r0),"=r"(r1) : "r"(addr))
#endif

#ifndef CP_ASYNC_CG
#define CP_ASYNC_CG(dst, src)                                                 \
    asm volatile("cp.async.cg.shared.global [%0],[%1],16;" :: "r"(dst),"l"(src))
#endif

// ---------------------------------------------------------------------------
// Tile config
// ---------------------------------------------------------------------------
template <int BM_, int BN_, int BK_, int STAGES_, int THREADS_>
struct SgemmTileConfigSM89 {
    static constexpr int BM      = BM_;
    static constexpr int BN      = BN_;
    static constexpr int BK      = BK_;
    static constexpr int STAGES  = STAGES_;
    static constexpr int THREADS = THREADS_;

    static constexpr int AS_SIZE    = BM * BK;
    static constexpr int BS_SIZE    = BK * BN;
    static constexpr int STAGE_SIZE = AS_SIZE + BS_SIZE;

    static constexpr int SMEM_BYTES = STAGES * (BM + BN) * BK * sizeof(float);
    static constexpr int MAX_OCC    = (BM * BN >= 256 * 128) ? 1 : 2;

    static constexpr int WARPS_TOTAL = THREADS / 32;
    static constexpr int WARPS_N     = (BN >= 64 && WARPS_TOTAL > 1)
                                         ? ((BN / 64 < WARPS_TOTAL / 2) ? BN / 64 : WARPS_TOTAL / 2)
                                         : 1;
    static constexpr int WARPS_M     = (WARPS_TOTAL > 1) ? (WARPS_TOTAL / WARPS_N) : 1;

    static constexpr int WARP_TILE_M = BM / WARPS_M;
    static constexpr int WARP_TILE_N = BN / WARPS_N;

    static constexpr int MMA_M = WARP_TILE_M / 16;
    static constexpr int MMA_N = WARP_TILE_N / 8;
};

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------
template <typename Cfg, bool IsAligned, bool IsSplitK, SgemmLayout Layout>
__global__ void __launch_bounds__(Cfg::THREADS, Cfg::MAX_OCC)
sgemm_sm89_kernel(
    int M, int N, int K, float alpha,
    const float* __restrict__ A, int lda, long long strideA,
    const float* __restrict__ B, int ldb, long long strideB,
    float beta,
    float* __restrict__ C, int ldc, long long strideC,
    const float* __restrict__ bias, long long bias_stride,
    int batchCount, int splitK)
{
    const int batch = blockIdx.z / splitK;
    const int sk_idx = blockIdx.z % splitK;
    if (batch >= batchCount) return;

    // Robust CTA swizzle for L2 locality (8-wide strips)
    const int sw = 8;
    const int grid_x = gridDim.x, grid_y = gridDim.y;
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
    const int wy   = wid / Cfg::WARPS_N, wx = wid % Cfg::WARPS_N;

    extern __shared__ float smem[];

    float acc[Cfg::MMA_M][Cfg::MMA_N][4];
    #pragma unroll
    for (int i = 0; i < Cfg::MMA_M; i++)
        for (int j = 0; j < Cfg::MMA_N; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    const int k_chunk = ((K + 15) / 16 + splitK - 1) / splitK * 16;
    const int k_start = sk_idx * k_chunk;
    const int k_end   = min(K, k_start + k_chunk);
    const int ktcnt   = (k_end - k_start + 15) / 16;

    const float* A_base = A + (long long)batch * strideA;
    const float* B_base = B + (long long)batch * strideB;

    // Load params
    constexpr int A_STRIDE = (Layout == SgemmLayout::TN) ? Cfg::BM : Cfg::BK;
    constexpr int A_TPR    = A_STRIDE / 4;
    constexpr int A_RPL    = Cfg::THREADS / A_TPR;
    constexpr int A_ITERS  = (Layout == SgemmLayout::TN) ? (Cfg::BK / A_RPL) : (Cfg::BM / A_RPL);
    const int a_row0 = tid / A_TPR, a_col0 = (tid % A_TPR) * 4;

    uint32_t sm_a_off[A_ITERS];
    #pragma unroll
    for (int i = 0; i < A_ITERS; i++) {
        const int r = a_row0 + i * A_RPL;
        if constexpr (Layout == SgemmLayout::TN) sm_a_off[i] = r * Cfg::BM + (a_col0 ^ ((r & 7) << 3));
        else sm_a_off[i] = r * Cfg::BK + (a_col0 ^ ((r & 3) << 2)); // BK=16 swizzle
    }

    constexpr int B_STRIDE = (Layout == SgemmLayout::NT) ? Cfg::BK : Cfg::BN;
    constexpr int B_TPR    = B_STRIDE / 4;
    constexpr int B_RPL    = Cfg::THREADS / B_TPR;
    constexpr int B_ITERS  = (Layout == SgemmLayout::NT) ? (Cfg::BN / B_RPL) : (Cfg::BK / B_RPL);
    const int b_row0 = tid / B_TPR, b_col0 = (tid % B_TPR) * 4;

    uint32_t sm_b_off[B_ITERS];
    #pragma unroll
    for (int i = 0; i < B_ITERS; i++) {
        const int r = b_row0 + i * B_RPL;
        if constexpr (Layout == SgemmLayout::NT) sm_b_off[i] = r * Cfg::BK + (b_col0 ^ ((r & 3) << 2));
        else sm_b_off[i] = r * Cfg::BN + (b_col0 ^ ((r & 7) << 2)); // BN=128 swizzle (8 row)
    }

    auto load_to_smem = [&](int stage, int ko) {
        float* As = smem + stage * Cfg::STAGE_SIZE;
        float* Bs = As   + Cfg::AS_SIZE;
        #pragma unroll
        for (int i = 0; i < A_ITERS; i++) {
            const int r = a_row0 + i * A_RPL, gm = (Layout == SgemmLayout::TN) ? (by * Cfg::BM + a_col0) : (ko + a_col0);
            const int gr = (Layout == SgemmLayout::TN) ? (ko + r) : (by * Cfg::BM + r);
            if (gr < M || (Layout == SgemmLayout::TN && gr < K)) { // Bounds check simplified
                uint32_t sm = (uint32_t)__cvta_generic_to_shared(As + sm_a_off[i]);
                const float* g = A_base + (long long)gr * lda + gm;
                if (IsAligned && ((Layout == SgemmLayout::TN) ? (gm+3 < M) : (gm+3 < K))) CP_ASYNC_CG(sm, g);
                else {
                    float4 val = {0.f,0.f,0.f,0.f};
                    const int lim = (Layout == SgemmLayout::TN) ? M : K;
                    if (gm < lim) val.x = g[0]; if (gm+1 < lim) val.y = g[1]; if (gm+2 < lim) val.z = g[2]; if (gm+3 < lim) val.w = g[3];
                    *(float4*)(As + sm_a_off[i]) = val;
                }
            }
        }
        #pragma unroll
        for (int i = 0; i < B_ITERS; i++) {
            const int r = b_row0 + i * B_RPL, gn = (Layout == SgemmLayout::NT) ? (bx * Cfg::BN + r) : (bx * Cfg::BN + b_col0);
            const int gk = (Layout == SgemmLayout::NT) ? (ko + b_col0) : (ko + r);
            const int row_g = (Layout == SgemmLayout::NT) ? gn : gk, col_g = (Layout == SgemmLayout::NT) ? gk : gn;
            uint32_t sm = (uint32_t)__cvta_generic_to_shared(Bs + sm_b_off[i]);
            const float* g = B_base + (long long)row_g * ldb + col_g;
            const int lim = (Layout == SgemmLayout::NT) ? K : N;
            if ((Layout == SgemmLayout::NT ? gn < N : gk < K) && IsAligned && col_g + 3 < lim) CP_ASYNC_CG(sm, g);
            else if (Layout == SgemmLayout::NT ? gn < N : gk < K) {
                float4 val = {0,0,0,0};
                if (col_g < lim) val.x = g[0]; if (col_g+1 < lim) val.y = g[1]; if (col_g+2 < lim) val.z = g[2]; if (col_g+3 < lim) val.w = g[3];
                *(float4*)(Bs + sm_b_off[i]) = val;
            }
        }
    };

    auto fetch_a = [&](uint32_t reg[4], int ks, int m, const float* As_st) {
        if constexpr (Layout == SgemmLayout::TN) {
            const int m_idx = wy * Cfg::WARP_TILE_M + m * 16 + (lane / 4);
            const int k0 = ks + (lane % 4), k4 = k0 + 4;
            auto la = [&](int cur_k, int cur_m) { return *(const uint32_t*)(&As_st[cur_k * Cfg::BM + (cur_m ^ ((cur_k & 7) << 3))]); };
            reg[0] = la(k0, m_idx); reg[1] = la(k0, m_idx + 8); reg[2] = la(k4, m_idx); reg[3] = la(k4, m_idx + 8);
        } else {
            const int lb = wy * Cfg::WARP_TILE_M + m * 16, lr0 = lb + (lane / 4), lr8 = lr0 + 8, lc = ks + (lane % 4);
            auto ga = [&](int r, int c) { return *(const uint32_t*)(&As_st[r * Cfg::BK + (c ^ ((r & 3) << 2))]); };
            reg[0] = ga(lr0, lc); reg[1] = ga(lr8, lc); reg[2] = ga(lr0, lc + 4); reg[3] = ga(lr8, lc + 4);
        }
    };

    auto fetch_b = [&](uint32_t reg[2], int ks, int n, const float* Bs_st) {
        if constexpr (Layout == SgemmLayout::NT) {
            const int lr = wx * Cfg::WARP_TILE_N + n * 8 + (lane / 4), lc0 = ks + (lane % 4), lc4 = lc0 + 4;
            auto gb = [&](int r, int c) { return *(const uint32_t*)(&Bs_st[r * Cfg::BK + (c ^ ((r & 3) << 2))]); };
            reg[0] = gb(lr, lc0); reg[1] = gb(lr, lc4);
        } else {
            const int lr0 = ks + (lane % 4), lr4 = lr0 + 4, lc = wx * Cfg::WARP_TILE_N + n * 8 + (lane / 4);
            auto gb = [&](int r, int c) { return *(const uint32_t*)(&Bs_st[r * Cfg::BN + (c ^ ((r & 7) << 2))]); };
            reg[0] = gb(lr0, lc); reg[1] = gb(lr4, lc);
        }
    };

    // Warmup
    constexpr int PREFETCH = Cfg::STAGES - 1;
    int rs = 0, ws = 0;
    #pragma unroll
    for (int i = 0; i < PREFETCH; i++) { if (i < ktcnt) load_to_smem(ws, k_start + i * Cfg::BK); asm volatile("cp.async.commit_group;"); ws = (ws + 1) % Cfg::STAGES; }
    asm volatile("cp.async.wait_group %0;" :: "n"(Cfg::STAGES - 2)); __syncthreads();

    uint32_t frA[2][Cfg::MMA_M][4], frB[2][Cfg::MMA_N][2];
    #pragma unroll
    for (int m = 0; m < Cfg::MMA_M; m++) fetch_a(frA[0][m], 0, m, smem + rs * Cfg::STAGE_SIZE);
    #pragma unroll
    for (int n = 0; n < Cfg::MMA_N; n++) fetch_b(frB[0][n], 0, n, smem + rs * Cfg::STAGE_SIZE + Cfg::AS_SIZE);

    for (int kt = 0; kt < ktcnt; kt++) {
        const int kf = k_start + (kt + PREFETCH) * Cfg::BK;
        if (kf < k_end) load_to_smem(ws, kf);
        asm volatile("cp.async.commit_group;");

        #pragma unroll
        for (int m = 0; m < Cfg::MMA_M; m++) {
            #pragma unroll
            for (int n = 0; n < Cfg::MMA_N; n++) {
                float &d0 = acc[m][n][0], &d1 = acc[m][n][1], &d2 = acc[m][n][2], &d3 = acc[m][n][3];
                uint32_t a0 = frA[0][m][0], a1 = frA[0][m][1], a2 = frA[0][m][2], a3 = frA[0][m][3];
                uint32_t b0 = frB[0][n][0], b1 = frB[0][n][1];
                MMA_TF32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
                if (m == 0) fetch_b(frB[1][n], 8, n, smem + rs * Cfg::STAGE_SIZE + Cfg::AS_SIZE);
            }
            fetch_a(frA[1][m], 8, m, smem + rs * Cfg::STAGE_SIZE);
        }
        asm volatile("cp.async.wait_group %0;" :: "n"(Cfg::STAGES - 2)); __syncthreads();
        rs = (rs + 1) % Cfg::STAGES; ws = (ws + 1) % Cfg::STAGES;
        #pragma unroll
        for (int m = 0; m < Cfg::MMA_M; m++) {
            #pragma unroll
            for (int n = 0; n < Cfg::MMA_N; n++) {
                float &d0 = acc[m][n][0], &d1 = acc[m][n][1], &d2 = acc[m][n][2], &d3 = acc[m][n][3];
                uint32_t a0 = frA[1][m][0], a1 = frA[1][m][1], a2 = frA[1][m][2], a3 = frA[1][m][3];
                uint32_t b0 = frB[1][n][0], b1 = frB[1][n][1];
                MMA_TF32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
                if (m == 0 && kt + 1 < ktcnt) fetch_b(frB[0][n], 0, n, smem + rs * Cfg::STAGE_SIZE + Cfg::AS_SIZE);
            }
            if (kt + 1 < ktcnt) fetch_a(frA[0][m], 0, m, smem + rs * Cfg::STAGE_SIZE);
        }
    }

    float* dC = C + (long long)batch * strideC;
    const int ge = lane >> 2, te = lane & 3;
    #pragma unroll
    for (int m = 0; m < Cfg::MMA_M; m++) {
        const int r0 = by * Cfg::BM + wy * Cfg::WARP_TILE_M + m * 16 + ge, r8 = r0 + 8;
        if (r0 >= M) continue;
        #pragma unroll
        for (int n = 0; n < Cfg::MMA_N; n++) {
            const int c0 = bx * Cfg::BN + wx * Cfg::WARP_TILE_N + n * 8 + te * 2;
            if (c0 >= N) continue;
            auto st = [&](int r, float v0, float v1) {
                if (r >= M) return;
                float* p = dC + (long long)r * ldc + c0;
                float f0 = alpha * v0, f1 = alpha * v1;
                if (bias) {
                    if (bias_stride == (long long)N) {
                        f0 += bias[c0]; if (c0 + 1 < N) f1 += bias[c0 + 1];
                    } else if (bias_stride == (long long)M * N) {
                        const float* pb = bias + (long long)(batch % M) * N + c0;
                        f0 += pb[0]; if (c0 + 1 < N) f1 += pb[1];
                    } else if (bias_stride == 1) {
                        f0 += bias[0]; f1 += bias[0];
                    }
                }
                if constexpr (IsSplitK) {
                    atomicAdd(p, f0); if (c0 + 1 < N) atomicAdd(p + 1, f1);
                } else {
                    if (beta != 0.f) { 
                        if (c0 + 1 < N) { float2 o = *(float2*)p; f0 += beta * o.x; f1 += beta * o.y; } 
                        else f0 += beta * p[0]; 
                    }
                    if (c0 + 1 < N) *(float2*)p = make_float2(f0, f1); else p[0] = f0;
                }
            };
            st(r0, acc[m][n][0], acc[m][n][1]); st(r8, acc[m][n][2], acc[m][n][3]);
        }
    }
}

template <typename Config, bool IsAligned, int SplitK, SgemmLayout Layout>
__global__ void sgemm_sm89_template_kernel( // Compatibility wrapper for old name if needed
    int M, int N, int K, float alpha, const float* A, int lda, long long sA, const float* B, int ldb, long long sB, float beta, float* C, int ldc, long long sC, int bc)
{
    sgemm_sm89_kernel<Config, IsAligned, SplitK, Layout><<<1, 1, 0, 0>>>(M, N, K, alpha, A, lda, sA, B, ldb, sB, beta, C, ldc, sC, bc, SplitK);
}

template <typename Config>
__global__ void sgemm_sm89_scale_kernel(float* C, float beta, int M, int N, int ldc, long long strideC, int batchCount) {
    const int r = blockIdx.y * blockDim.y + threadIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x, b = blockIdx.z;
    if (r < M && c < N && b < batchCount) { float* dst = &C[(long long)b * strideC + (long long)r * ldc + c]; *dst = (beta == 0.f) ? 0.f : (*dst * beta); }
}























// #pragma once

// #include <cuda_runtime.h>
// #include <stdint.h>
// #include <stdio.h>

// /**
//  * Optimized SM89 SGEMM Core — High Performance & Accuracy
//  *
//  * Changes vs previous revision (full optimization sweep):
//  *   - ldmatrix.x4 for A in NN / NT layouts (M-major SMEM)
//  *   - ldmatrix.x2.trans for B in NT layout (N-major SMEM)
//  *   - Lane-invariant ldmatrix addressing hoisted out of per-m / per-n calls
//  *   - Epilogue: bias-mode dispatch lifted out of the m,n inner loops
//  *   - splitK correctness: bias is applied only by sk_idx == 0. beta*C_old is
//  *     expected to be pre-applied by the companion scale kernel before the
//  *     splitK launch (that kernel already exists in this file); the splitK
//  *     path in the epilogue ignores beta, matching the scale-then-add pattern.
//  *     Previously the splitK path both ignored beta AND applied bias splitK
//  *     times via atomicAdd, making bias incorrect.
//  *   - TN layout bounds check fixed (previously used an `||` that could let
//  *     out-of-range K rows pass a spurious M-bound test)
//  *   - Grid swizzle guarded against actual_sw <= 0 for degenerate grids
//  *   - Accumulator init fully unrolled
//  *
//  * Layout conventions (SMEM):
//  *   - NN / NT : A is M-major (rows=M, cols=K, stride=BK). fast axis = K
//  *   - TN      : A is K-major (rows=K, cols=M, stride=BM). fast axis = M
//  *   - NT      : B is N-major (rows=N, cols=K, stride=BK). fast axis = K
//  *   - NN / TN : B is K-major (rows=K, cols=N, stride=BN). fast axis = N
//  *
//  * ldmatrix / TF32 compatibility (why A-NN/NT and B-NT use ldmatrix but not
//  * the other cases):
//  *   A TF32 32-bit word's two b16 halves are adjacent along the SMEM fast
//  *   axis. Non-trans ldmatrix pairs b16 halves along the tile-col direction;
//  *   .trans pairs them along the tile-row direction. For a thread's register
//  *   to hold a complete TF32 value the two paired halves must correspond to
//  *   one 32-bit word, i.e. the SMEM fast axis must match the ldmatrix pair
//  *   direction. Crossing that rule silently corrupts values, so we use
//  *   ldmatrix only where the layouts agree.
//  */

// #ifndef MYCUBLAS_LAYOUT_ENUM
// #define MYCUBLAS_LAYOUT_ENUM
// enum class SgemmLayout { NT, TN, NN };
// #endif

// // ---------------------------------------------------------------------------
// // PTX helpers
// // ---------------------------------------------------------------------------
// #ifndef MMA_TF32
// #define MMA_TF32(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1)                           \
//     asm volatile(                                                              \
//         "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "                \
//         "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"                  \
//         : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)                                \
//         : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1))
// #endif

// #ifndef LDSM_X4
// #define LDSM_X4(r0,r1,r2,r3,addr)                                            \
//     asm volatile(                                                              \
//         "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3},[%4];"      \
//         : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "r"(addr))
// #endif

// #ifndef LDSM_X2
// #define LDSM_X2(r0,r1,addr)                                                  \
//     asm volatile(                                                              \
//         "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1},[%2];"            \
//         : "=r"(r0),"=r"(r1) : "r"(addr))
// #endif

// #ifndef LDSM_X2_TRANS
// #define LDSM_X2_TRANS(r0,r1,addr)                                            \
//     asm volatile(                                                              \
//         "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1},[%2];"      \
//         : "=r"(r0),"=r"(r1) : "r"(addr))
// #endif

// #ifndef LDSM_X1
// #define LDSM_X1(r0,r1,addr)                                                  \
//     asm volatile(                                                              \
//         "ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0,%1},[%2];"            \
//         : "=r"(r0),"=r"(r1) : "r"(addr))
// #endif

// #ifndef CP_ASYNC_CG
// #define CP_ASYNC_CG(dst, src)                                                 \
//     asm volatile("cp.async.cg.shared.global [%0],[%1],16;" :: "r"(dst),"l"(src))
// #endif

// // ---------------------------------------------------------------------------
// // Tile config
// // ---------------------------------------------------------------------------
// template <int BM_, int BN_, int BK_, int STAGES_, int THREADS_>
// struct SgemmTileConfigSM89 {
//     static constexpr int BM      = BM_;
//     static constexpr int BN      = BN_;
//     static constexpr int BK      = BK_;
//     static constexpr int STAGES  = STAGES_;
//     static constexpr int THREADS = THREADS_;

//     static constexpr int AS_SIZE    = BM * BK;
//     static constexpr int BS_SIZE    = BK * BN;
//     static constexpr int STAGE_SIZE = AS_SIZE + BS_SIZE;

//     static constexpr int SMEM_BYTES = STAGES * (BM + BN) * BK * sizeof(float);
//     static constexpr int MAX_OCC    = (BM * BN >= 256 * 128) ? 1 : 2;

//     static constexpr int WARPS_TOTAL = THREADS / 32;
//     static constexpr int WARPS_N     = (BN >= 64 && WARPS_TOTAL > 1)
//                                          ? ((BN / 64 < WARPS_TOTAL / 2) ? BN / 64 : WARPS_TOTAL / 2)
//                                          : 1;
//     static constexpr int WARPS_M     = (WARPS_TOTAL > 1) ? (WARPS_TOTAL / WARPS_N) : 1;

//     static constexpr int WARP_TILE_M = BM / WARPS_M;
//     static constexpr int WARP_TILE_N = BN / WARPS_N;

//     static constexpr int MMA_M = WARP_TILE_M / 16;
//     static constexpr int MMA_N = WARP_TILE_N / 8;
// };

// // ---------------------------------------------------------------------------
// // Kernel
// // ---------------------------------------------------------------------------
// template <typename Cfg, bool IsAligned, bool IsSplitK, SgemmLayout Layout>
// __global__ void __launch_bounds__(Cfg::THREADS, Cfg::MAX_OCC)
// sgemm_sm89_kernel(
//     int M, int N, int K, float alpha,
//     const float* __restrict__ A, int lda, long long strideA,
//     const float* __restrict__ B, int ldb, long long strideB,
//     float beta,
//     float* __restrict__ C, int ldc, long long strideC,
//     const float* __restrict__ bias, long long bias_stride,
//     int batchCount, int splitK)
// {
//     const int batch  = blockIdx.z / splitK;
//     const int sk_idx = blockIdx.z % splitK;
//     if (batch >= batchCount) return;

//     // Robust CTA swizzle for L2 locality (8-wide strips).
//     // Guard against degenerate grids where (strip_idx * sw) overshoots grid_x,
//     // which would make `actual_sw <= 0` and break the subsequent modulo.
//     const int sw = 8;
//     const int grid_x = gridDim.x, grid_y = gridDim.y;
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
//     const int wy   = wid / Cfg::WARPS_N, wx = wid % Cfg::WARPS_N;

//     extern __shared__ float smem[];

//     float acc[Cfg::MMA_M][Cfg::MMA_N][4];
//     #pragma unroll
//     for (int i = 0; i < Cfg::MMA_M; i++) {
//         #pragma unroll
//         for (int j = 0; j < Cfg::MMA_N; j++) {
//             acc[i][j][0] = 0.f;
//             acc[i][j][1] = 0.f;
//             acc[i][j][2] = 0.f;
//             acc[i][j][3] = 0.f;
//         }
//     }

//     const int k_chunk = ((K + 15) / 16 + splitK - 1) / splitK * 16;
//     const int k_start = sk_idx * k_chunk;
//     const int k_end   = min(K, k_start + k_chunk);
//     const int ktcnt   = (k_end - k_start + 15) / 16;

//     const float* A_base = A + (long long)batch * strideA;
//     const float* B_base = B + (long long)batch * strideB;

//     // -----------------------------------------------------------------------
//     // Global → shared load setup
//     // -----------------------------------------------------------------------
//     constexpr int A_STRIDE = (Layout == SgemmLayout::TN) ? Cfg::BM : Cfg::BK;
//     constexpr int A_TPR    = A_STRIDE / 4;
//     constexpr int A_RPL    = Cfg::THREADS / A_TPR;
//     constexpr int A_ITERS  = (Layout == SgemmLayout::TN) ? (Cfg::BK / A_RPL) : (Cfg::BM / A_RPL);
//     const int a_row0 = tid / A_TPR, a_col0 = (tid % A_TPR) * 4;

//     uint32_t sm_a_off[A_ITERS];
//     #pragma unroll
//     for (int i = 0; i < A_ITERS; i++) {
//         const int r = a_row0 + i * A_RPL;
//         if constexpr (Layout == SgemmLayout::TN) sm_a_off[i] = r * Cfg::BM + (a_col0 ^ ((r & 7) << 3));
//         else                                     sm_a_off[i] = r * Cfg::BK + (a_col0 ^ ((r & 3) << 2));
//     }

//     constexpr int B_STRIDE = (Layout == SgemmLayout::NT) ? Cfg::BK : Cfg::BN;
//     constexpr int B_TPR    = B_STRIDE / 4;
//     constexpr int B_RPL    = Cfg::THREADS / B_TPR;
//     constexpr int B_ITERS  = (Layout == SgemmLayout::NT) ? (Cfg::BN / B_RPL) : (Cfg::BK / B_RPL);
//     const int b_row0 = tid / B_TPR, b_col0 = (tid % B_TPR) * 4;

//     uint32_t sm_b_off[B_ITERS];
//     #pragma unroll
//     for (int i = 0; i < B_ITERS; i++) {
//         const int r = b_row0 + i * B_RPL;
//         if constexpr (Layout == SgemmLayout::NT) sm_b_off[i] = r * Cfg::BK + (b_col0 ^ ((r & 3) << 2));
//         else                                     sm_b_off[i] = r * Cfg::BN + (b_col0 ^ ((r & 7) << 2));
//     }

//     auto load_to_smem = [&](int stage, int ko) {
//         float* As = smem + stage * Cfg::STAGE_SIZE;
//         float* Bs = As   + Cfg::AS_SIZE;
//         #pragma unroll
//         for (int i = 0; i < A_ITERS; i++) {
//             const int r  = a_row0 + i * A_RPL;
//             const int gm = (Layout == SgemmLayout::TN) ? (by * Cfg::BM + a_col0) : (ko + a_col0);
//             const int gr = (Layout == SgemmLayout::TN) ? (ko + r)                : (by * Cfg::BM + r);
//             // Bounds check on the row dimension (K for TN, M otherwise).
//             const bool row_ok = (Layout == SgemmLayout::TN) ? (gr < K) : (gr < M);
//             if (row_ok) {
//                 uint32_t sm = (uint32_t)__cvta_generic_to_shared(As + sm_a_off[i]);
//                 const float* g = A_base + (long long)gr * lda + gm;
//                 const int lim = (Layout == SgemmLayout::TN) ? M : K;
//                 if (IsAligned && gm + 3 < lim) {
//                     CP_ASYNC_CG(sm, g);
//                 } else {
//                     float4 val = {0.f, 0.f, 0.f, 0.f};
//                     if (gm     < lim) val.x = g[0];
//                     if (gm + 1 < lim) val.y = g[1];
//                     if (gm + 2 < lim) val.z = g[2];
//                     if (gm + 3 < lim) val.w = g[3];
//                     *(float4*)(As + sm_a_off[i]) = val;
//                 }
//             } else {
//                 *(float4*)(As + sm_a_off[i]) = {0.f, 0.f, 0.f, 0.f};
//             }
//         }
//         #pragma unroll
//         for (int i = 0; i < B_ITERS; i++) {
//             const int r  = b_row0 + i * B_RPL;
//             const int gn = (Layout == SgemmLayout::NT) ? (bx * Cfg::BN + r) : (bx * Cfg::BN + b_col0);
//             const int gk = (Layout == SgemmLayout::NT) ? (ko + b_col0)      : (ko + r);
//             const int row_g = (Layout == SgemmLayout::NT) ? gn : gk;
//             const int col_g = (Layout == SgemmLayout::NT) ? gk : gn;
//             const int lim = (Layout == SgemmLayout::NT) ? K : N;
//             const bool row_ok = (Layout == SgemmLayout::NT) ? (gn < N) : (gk < K);
//             if (row_ok) {
//                 uint32_t sm = (uint32_t)__cvta_generic_to_shared(Bs + sm_b_off[i]);
//                 const float* g = B_base + (long long)row_g * ldb + col_g;
//                 if (IsAligned && col_g + 3 < lim) {
//                     CP_ASYNC_CG(sm, g);
//                 } else {
//                     float4 val = {0.f, 0.f, 0.f, 0.f};
//                     if (col_g     < lim) val.x = g[0];
//                     if (col_g + 1 < lim) val.y = g[1];
//                     if (col_g + 2 < lim) val.z = g[2];
//                     if (col_g + 3 < lim) val.w = g[3];
//                     *(float4*)(Bs + sm_b_off[i]) = val;
//                 }
//             } else {
//                 *(float4*)(Bs + sm_b_off[i]) = {0.f, 0.f, 0.f, 0.f};
//             }
//         }
//     };

//     // -----------------------------------------------------------------------
//     // Register fragment loads
//     //
//     // Lane-dependent address terms are computed once outside the k-loop and
//     // combined with per-call (m, n, ks) offsets at the fetch site. The XOR
//     // swizzle of the K/N-column offset uses only the low 2 bits of the row
//     // offset; WARP_TILE_M is a multiple of 16 and m*16 is a multiple of 16,
//     // so neither perturbs the low 2 bits of M — the swizzle can be folded in
//     // as a lane-invariant constant.
//     // -----------------------------------------------------------------------

//     // --- A: lane-invariant parts for NN/NT ldmatrix.x4 ---
//     const int a_ld_row  = ((lane >> 3) >> 1 ? 8 : 0) + (lane & 7);
//     const int a_ld_kbit = ((lane >> 3) & 1) ? 4 : 0;
//     const int a_sw_xor  = (a_ld_row & 3) << 2;
//     const int a_row_abs_base = wy * Cfg::WARP_TILE_M + a_ld_row;  // add m*16 at call site

//     // --- B: lane-invariant parts for NT ldmatrix.x2.trans ---
//     const int b_ld_row_nt  = lane & 7;
//     const int b_ld_kbit_nt = ((lane >> 3) & 1) ? 4 : 0;
//     const int b_sw_xor_nt  = (b_ld_row_nt & 3) << 2;
//     const int b_row_abs_base_nt = wx * Cfg::WARP_TILE_N + b_ld_row_nt;

//     // --- A: lane-invariant parts for TN scalar loads ---
//     const int a_tn_m_base = wy * Cfg::WARP_TILE_M + (lane >> 2);  // + m*16
//     const int a_tn_k_lane = lane & 3;                              // + ks

//     // --- B: lane-invariant parts for NN/TN scalar loads ---
//     const int b_nn_k_lane = lane & 3;                              // + ks
//     const int b_nn_n_base = wx * Cfg::WARP_TILE_N + (lane >> 2);   // + n*8

//     auto fetch_a = [&](uint32_t reg[4], int ks, int m, const float* As_st) {
//         if constexpr (Layout == SgemmLayout::TN) {
//             // TN: A is K-major in SMEM. Manual scalar loads — register
//             // distribution for ldmatrix variants on K-major TF32 storage
//             // cannot match the MMA A-fragment layout without value corruption.
//             const int m_idx = a_tn_m_base + m * 16;
//             const int k0 = ks + a_tn_k_lane, k4 = k0 + 4;
//             auto la = [&](int cur_k, int cur_m) {
//                 return *(const uint32_t*)(&As_st[cur_k * Cfg::BM + (cur_m ^ ((cur_k & 7) << 3))]);
//             };
//             reg[0] = la(k0, m_idx);
//             reg[1] = la(k0, m_idx + 8);
//             reg[2] = la(k4, m_idx);
//             reg[3] = la(k4, m_idx + 8);
//         } else {
//             // NN / NT: A is M-major in SMEM. ldmatrix.x4 loads one A fragment
//             // per call. Register remap (d0, d1, d2, d3) → (a0, a2, a1, a3).
//             const int m_off = a_row_abs_base + m * 16;
//             const int k_sw  = (ks + a_ld_kbit) ^ a_sw_xor;
//             const uint32_t addr = (uint32_t)__cvta_generic_to_shared(
//                                       &As_st[m_off * Cfg::BK + k_sw]);
//             uint32_t d0, d1, d2, d3;
//             LDSM_X4(d0, d1, d2, d3, addr);
//             reg[0] = d0;  // a0 = (T/4,   T%4)
//             reg[1] = d2;  // a1 = (T/4+8, T%4)
//             reg[2] = d1;  // a2 = (T/4,   T%4+4)
//             reg[3] = d3;  // a3 = (T/4+8, T%4+4)
//         }
//     };

//     auto fetch_b = [&](uint32_t reg[2], int ks, int n, const float* Bs_st) {
//         if constexpr (Layout == SgemmLayout::NT) {
//             // NT: B is N-major in SMEM. ldmatrix.x2.trans delivers the exact
//             // MMA B-fragment distribution (reg[0]=d0, reg[1]=d1).
//             const int n_off = b_row_abs_base_nt + n * 8;
//             const int k_sw  = (ks + b_ld_kbit_nt) ^ b_sw_xor_nt;
//             const uint32_t addr = (uint32_t)__cvta_generic_to_shared(
//                                       &Bs_st[n_off * Cfg::BK + k_sw]);
//             uint32_t d0, d1;
//             LDSM_X2_TRANS(d0, d1, addr);
//             reg[0] = d0;
//             reg[1] = d1;
//         } else {
//             // NN / TN: B is K-major in SMEM. Manual scalar loads.
//             const int lr0 = ks + b_nn_k_lane, lr4 = lr0 + 4;
//             const int lc  = b_nn_n_base + n * 8;
//             auto gb = [&](int r, int c) {
//                 return *(const uint32_t*)(&Bs_st[r * Cfg::BN + (c ^ ((r & 7) << 2))]);
//             };
//             reg[0] = gb(lr0, lc);
//             reg[1] = gb(lr4, lc);
//         }
//     };

//     // -----------------------------------------------------------------------
//     // Pipeline warmup
//     // -----------------------------------------------------------------------
//     constexpr int PREFETCH = Cfg::STAGES - 1;
//     int rs = 0, ws = 0;
//     #pragma unroll
//     for (int i = 0; i < PREFETCH; i++) {
//         if (i < ktcnt) load_to_smem(ws, k_start + i * Cfg::BK);
//         asm volatile("cp.async.commit_group;");
//         ws = (ws + 1) % Cfg::STAGES;
//     }
//     asm volatile("cp.async.wait_group %0;" :: "n"(Cfg::STAGES - 2));
//     __syncthreads();

//     uint32_t frA[2][Cfg::MMA_M][4], frB[2][Cfg::MMA_N][2];
//     #pragma unroll
//     for (int m = 0; m < Cfg::MMA_M; m++) fetch_a(frA[0][m], 0, m, smem + rs * Cfg::STAGE_SIZE);
//     #pragma unroll
//     for (int n = 0; n < Cfg::MMA_N; n++) fetch_b(frB[0][n], 0, n, smem + rs * Cfg::STAGE_SIZE + Cfg::AS_SIZE);

//     // -----------------------------------------------------------------------
//     // Main K loop
//     // -----------------------------------------------------------------------
//     for (int kt = 0; kt < ktcnt; kt++) {
//         const int kf = k_start + (kt + PREFETCH) * Cfg::BK;
//         if (kf < k_end) load_to_smem(ws, kf);
//         asm volatile("cp.async.commit_group;");

//         // First inner K-step (ks=0): consume frA[0]/frB[0], prefetch frA[1]/frB[1]
//         #pragma unroll
//         for (int m = 0; m < Cfg::MMA_M; m++) {
//             #pragma unroll
//             for (int n = 0; n < Cfg::MMA_N; n++) {
//                 float &d0 = acc[m][n][0], &d1 = acc[m][n][1];
//                 float &d2 = acc[m][n][2], &d3 = acc[m][n][3];
//                 uint32_t a0 = frA[0][m][0], a1 = frA[0][m][1];
//                 uint32_t a2 = frA[0][m][2], a3 = frA[0][m][3];
//                 uint32_t b0 = frB[0][n][0], b1 = frB[0][n][1];
//                 MMA_TF32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
//                 if (m == 0) fetch_b(frB[1][n], 8, n, smem + rs * Cfg::STAGE_SIZE + Cfg::AS_SIZE);
//             }
//             fetch_a(frA[1][m], 8, m, smem + rs * Cfg::STAGE_SIZE);
//         }

//         asm volatile("cp.async.wait_group %0;" :: "n"(Cfg::STAGES - 2));
//         __syncthreads();
//         rs = (rs + 1) % Cfg::STAGES;
//         ws = (ws + 1) % Cfg::STAGES;

//         // Second inner K-step (ks=8): consume frA[1]/frB[1], prefetch for next kt
//         #pragma unroll
//         for (int m = 0; m < Cfg::MMA_M; m++) {
//             #pragma unroll
//             for (int n = 0; n < Cfg::MMA_N; n++) {
//                 float &d0 = acc[m][n][0], &d1 = acc[m][n][1];
//                 float &d2 = acc[m][n][2], &d3 = acc[m][n][3];
//                 uint32_t a0 = frA[1][m][0], a1 = frA[1][m][1];
//                 uint32_t a2 = frA[1][m][2], a3 = frA[1][m][3];
//                 uint32_t b0 = frB[1][n][0], b1 = frB[1][n][1];
//                 MMA_TF32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
//                 if (m == 0 && kt + 1 < ktcnt) fetch_b(frB[0][n], 0, n, smem + rs * Cfg::STAGE_SIZE + Cfg::AS_SIZE);
//             }
//             if (kt + 1 < ktcnt) fetch_a(frA[0][m], 0, m, smem + rs * Cfg::STAGE_SIZE);
//         }
//     }

//     // -----------------------------------------------------------------------
//     // Epilogue
//     //
//     // Non-splitK path: apply alpha*AB + beta*C_old + bias directly.
//     // SplitK path: beta*C_old is expected to be pre-applied to C by the
//     //   companion scale kernel before the splitK launch, so we only
//     //   atomicAdd alpha*partial + bias_from_sk0.  Bias is added exactly once
//     //   by the sk_idx == 0 block.
//     // -----------------------------------------------------------------------
//     float* dC = C + (long long)batch * strideC;
//     const int ge = lane >> 2, te = lane & 3;

//     // Classify bias mode: 0 = none, 1 = scalar, 2 = per-col (length N),
//     //                     3 = per-(batch)(N) (length M*N, batch-major).
//     const bool bias_active = bias && (!IsSplitK || sk_idx == 0);
//     const int bias_mode = !bias_active                          ? 0
//                         : (bias_stride == 1)                    ? 1
//                         : (bias_stride == (long long)N)         ? 2
//                         : (bias_stride == (long long)M * N)     ? 3
//                         : 0;
//     const float  bias_scalar = (bias_mode == 1) ? bias[0] : 0.f;
//     const float* bias_matrix_row = (bias_mode == 3) ? (bias + (long long)(batch % M) * N) : nullptr;

//     #pragma unroll
//     for (int m = 0; m < Cfg::MMA_M; m++) {
//         const int r0 = by * Cfg::BM + wy * Cfg::WARP_TILE_M + m * 16 + ge;
//         const int r8 = r0 + 8;
//         if (r0 >= M) continue;

//         #pragma unroll
//         for (int n = 0; n < Cfg::MMA_N; n++) {
//             const int c0 = bx * Cfg::BN + wx * Cfg::WARP_TILE_N + n * 8 + te * 2;
//             if (c0 >= N) continue;
//             const int c1 = c0 + 1;

//             float bv0 = 0.f, bv1 = 0.f;
//             switch (bias_mode) {
//                 case 1: bv0 = bias_scalar; bv1 = bias_scalar; break;
//                 case 2:
//                     bv0 = bias[c0];
//                     if (c1 < N) bv1 = bias[c1];
//                     break;
//                 case 3:
//                     bv0 = bias_matrix_row[c0];
//                     if (c1 < N) bv1 = bias_matrix_row[c1];
//                     break;
//                 default: break;
//             }

//             auto st = [&](int r, float v0, float v1) {
//                 if (r >= M) return;
//                 float* p = dC + (long long)r * ldc + c0;
//                 float f0 = alpha * v0 + bv0;
//                 float f1 = alpha * v1 + bv1;

//                 if constexpr (IsSplitK) {
//                     // beta*C_old handled externally by the scale kernel.
//                     atomicAdd(p, f0);
//                     if (c1 < N) atomicAdd(p + 1, f1);
//                 } else {
//                     if (beta != 0.f) {
//                         if (c1 < N) {
//                             float2 o = *(float2*)p;
//                             f0 += beta * o.x;
//                             f1 += beta * o.y;
//                         } else {
//                             f0 += beta * p[0];
//                         }
//                     }
//                     if (c1 < N) *(float2*)p = make_float2(f0, f1);
//                     else        p[0]       = f0;
//                 }
//             };

//             st(r0, acc[m][n][0], acc[m][n][1]);
//             st(r8, acc[m][n][2], acc[m][n][3]);
//         }
//     }
// }

// template <typename Config, bool IsAligned, int SplitK, SgemmLayout Layout>
// __global__ void sgemm_sm89_template_kernel( // Compatibility wrapper for old name if needed
//     int M, int N, int K, float alpha, const float* A, int lda, long long sA, const float* B, int ldb, long long sB, float beta, float* C, int ldc, long long sC, int bc)
// {
//     sgemm_sm89_kernel<Config, IsAligned, SplitK, Layout><<<1, 1, 0, 0>>>(M, N, K, alpha, A, lda, sA, B, ldb, sB, beta, C, ldc, sC, bc, SplitK);
// }

// template <typename Config>
// __global__ void sgemm_sm89_scale_kernel(float* C, float beta, int M, int N, int ldc, long long strideC, int batchCount) {
//     const int r = blockIdx.y * blockDim.y + threadIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x, b = blockIdx.z;
//     if (r < M && c < N && b < batchCount) { float* dst = &C[(long long)b * strideC + (long long)r * ldc + c]; *dst = (beta == 0.f) ? 0.f : (*dst * beta); }
// }