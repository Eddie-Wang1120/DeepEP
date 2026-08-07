// bench_combine_gather_sum.cu -- megakernel-shaped combine gather-and-sum microbenchmark.
//
// This isolates the combine NVL sender multi-hit gather/reduce path while preserving
// megakernel's block / warp / thread shape:
//   - blockDim = 800 threads = (24 combine forwarder warps + 1 coordinator warp) * 32
//   - only the first 8 warps in each block act as NVL sender warps for A-D
//   - E/F use the full 1SM / 800-thread gather-worker shape
//
// Compared configs:
//   A: staged smem gather-sum, approximates current chunked staging before reduce
//   B: direct scalar gather-sum, one int4 position per lane iteration
//   C: direct chunked gather-sum, keeps A's per-lane ILP/chunk shape but skips staging
//   D: direct chunked gather-sum with compile-time NH specialization
//   E: current gather-worker reduce shape, one SM serializes task tokens with 800 threads
//   F: DeepGEMM-style warp/register reduce, all 25 warps parallelize token chunks
//
// Build:
//   nvcc -std=c++17 -gencode=arch=compute_103a,code=sm_103a -O3 \
//        --expt-relaxed-constexpr bench_combine_gather_sum.cu -o bench_combine_gather_sum
//
// Usage:
//   ./bench_combine_gather_sum [tokens] [hidden] [topk] [nh] [blocks] [mode]
//     tokens : default 4096
//     hidden : BF16 elements per token, default 4096, must be a multiple of 8
//     topk   : slot-list width, default 8
//     nh     : local hits per token, default 2, must satisfy 1 <= nh <= topk
//     blocks : simulated SM blocks, default 16; use 1 for 1SM gather-worker validation
//     mode   : any combination of A/B/C/D/E/F, e.g. EF; default * runs all

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#define CHECK_CUDA(call) do { \
  cudaError_t _e = (call); \
  if (_e != cudaSuccess) { \
    std::cerr << "CUDA error " << cudaGetErrorString(_e) << " @ " << __FILE__ << ":" << __LINE__ << std::endl; \
    std::exit(1); \
  } \
} while (0)

static constexpr int kNumSenderWarps = 8;
static constexpr int kNumCombineForwarderWarps = 24;
static constexpr int kMegaKernelThreads = (kNumCombineForwarderWarps + 1) * 32;
static constexpr int kElemsPerInt4 = sizeof(int4) / sizeof(__nv_bfloat16);
static constexpr int kChunkInt4 = 128;
static constexpr int kVecsPerLane = kChunkInt4 / 32;
static constexpr int kPairsPerInt4 = kElemsPerInt4 / 2;

__device__ __forceinline__ int4 ld_nc_int4(const int4* ptr) {
  int4 ret;
  asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
               : "l"(ptr));
  return ret;
}

__device__ __forceinline__ void accum_bf162(float2& acc, __nv_bfloat162 value) {
  float2 v = __bfloat1622float2(value);
  acc.x += v.x;
  acc.y += v.y;
}

__device__ __forceinline__ void accum_bf162_dg(float2& acc, __nv_bfloat162 value) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t packed = *reinterpret_cast<uint32_t*>(&value);
  uint16_t lo = static_cast<uint16_t>(packed & 0xffffu);
  uint16_t hi = static_cast<uint16_t>(packed >> 16);
  asm volatile("add.rn.f32.bf16 %0, %1, %0;" : "+f"(acc.x) : "h"(lo));
  asm volatile("add.rn.f32.bf16 %0, %1, %0;" : "+f"(acc.y) : "h"(hi));
#else
  accum_bf162(acc, value);
#endif
}

__device__ __forceinline__ void accum_int4(float2 (&acc)[kPairsPerInt4], int4 raw) {
  const __nv_bfloat162* bv2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
  for (int p = 0; p < kPairsPerInt4; ++p)
    accum_bf162(acc[p], bv2[p]);
}

__device__ __forceinline__ void accum_int4_dg(float2 (&acc)[kPairsPerInt4], int4 raw) {
  const __nv_bfloat162* bv2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
  for (int p = 0; p < kPairsPerInt4; ++p)
    accum_bf162_dg(acc[p], bv2[p]);
}

__device__ __forceinline__ int4 pack_acc(const float2 (&acc)[kPairsPerInt4]) {
  int4 packed;
  __nv_bfloat162* pv2 = reinterpret_cast<__nv_bfloat162*>(&packed);
#pragma unroll
  for (int p = 0; p < kPairsPerInt4; ++p)
    pv2[p] = __float22bfloat162_rn(acc[p]);
  return packed;
}

__global__ __launch_bounds__(kMegaKernelThreads, 1)
void combine_gather_sum_staged_kernel(const int4* __restrict__ slots,
                                      const int* __restrict__ token_slot_list,
                                      int4* __restrict__ out,
                                      int num_tokens, int hidden_int4,
                                      int topk, int nh) {
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane = tid & 31;
  if (warp_id >= kNumSenderWarps) return;

  __shared__ int4 smem[kNumSenderWarps][kChunkInt4];
  for (int token = blockIdx.x * kNumSenderWarps + warp_id;
       token < num_tokens;
       token += gridDim.x * kNumSenderWarps) {
    int4* dst = out + static_cast<size_t>(token) * hidden_int4;
    for (int chunk_base = 0; chunk_base < hidden_int4; chunk_base += kChunkInt4) {
      const int chunk_end = min(chunk_base + kChunkInt4, hidden_int4);
      const int chunk_int4 = chunk_end - chunk_base;
      float2 acc[kVecsPerLane][kPairsPerInt4];
#pragma unroll
      for (int j = 0; j < kVecsPerLane; ++j) {
#pragma unroll
        for (int p = 0; p < kPairsPerInt4; ++p)
          acc[j][p] = make_float2(0.0f, 0.0f);
      }

      for (int h = 0; h < nh; ++h) {
        const int slot = token_slot_list[token * topk + h];
#pragma unroll
        for (int j = 0; j < kVecsPerLane; ++j) {
          const int local_vi = lane + j * 32;
          if (local_vi < chunk_int4)
            smem[warp_id][local_vi] = ld_nc_int4(slots + static_cast<size_t>(slot) * hidden_int4 + chunk_base + local_vi);
        }
        __syncwarp();
#pragma unroll
        for (int j = 0; j < kVecsPerLane; ++j) {
          const int local_vi = lane + j * 32;
          if (local_vi < chunk_int4)
            accum_int4(acc[j], smem[warp_id][local_vi]);
        }
        __syncwarp();
      }

#pragma unroll
      for (int j = 0; j < kVecsPerLane; ++j) {
        const int vi = chunk_base + lane + j * 32;
        if (vi < chunk_end)
          dst[vi] = pack_acc(acc[j]);
      }
    }
  }
}

__global__ __launch_bounds__(kMegaKernelThreads, 1)
void combine_gather_sum_direct_scalar_kernel(const int4* __restrict__ slots,
                                             const int* __restrict__ token_slot_list,
                                             int4* __restrict__ out,
                                             int num_tokens, int hidden_int4,
                                             int topk, int nh) {
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane = tid & 31;
  if (warp_id >= kNumSenderWarps) return;

  for (int token = blockIdx.x * kNumSenderWarps + warp_id;
       token < num_tokens;
       token += gridDim.x * kNumSenderWarps) {
    int4* dst = out + static_cast<size_t>(token) * hidden_int4;
    for (int vi = lane; vi < hidden_int4; vi += 32) {
      float2 acc[kPairsPerInt4];
#pragma unroll
      for (int p = 0; p < kPairsPerInt4; ++p)
        acc[p] = make_float2(0.0f, 0.0f);
      for (int h = 0; h < nh; ++h) {
        const int slot = token_slot_list[token * topk + h];
        accum_int4(acc, ld_nc_int4(slots + static_cast<size_t>(slot) * hidden_int4 + vi));
      }
      dst[vi] = pack_acc(acc);
    }
  }
}

template <int NH>
__global__ __launch_bounds__(kMegaKernelThreads, 1)
void combine_gather_sum_direct_chunked_nh_kernel(const int4* __restrict__ slots,
                                                 const int* __restrict__ token_slot_list,
                                                 int4* __restrict__ out,
                                                 int num_tokens, int hidden_int4,
                                                 int topk) {
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane = tid & 31;
  if (warp_id >= kNumSenderWarps) return;

  for (int token = blockIdx.x * kNumSenderWarps + warp_id;
       token < num_tokens;
       token += gridDim.x * kNumSenderWarps) {
    int4* dst = out + static_cast<size_t>(token) * hidden_int4;
    for (int chunk_base = 0; chunk_base < hidden_int4; chunk_base += kChunkInt4) {
      const int chunk_end = min(chunk_base + kChunkInt4, hidden_int4);
      float2 acc[kVecsPerLane][kPairsPerInt4];
#pragma unroll
      for (int j = 0; j < kVecsPerLane; ++j) {
#pragma unroll
        for (int p = 0; p < kPairsPerInt4; ++p)
          acc[j][p] = make_float2(0.0f, 0.0f);
      }

#pragma unroll
      for (int h = 0; h < NH; ++h) {
        const int slot = token_slot_list[token * topk + h];
#pragma unroll
        for (int j = 0; j < kVecsPerLane; ++j) {
          const int vi = chunk_base + lane + j * 32;
          if (vi < chunk_end)
            accum_int4(acc[j], ld_nc_int4(slots + static_cast<size_t>(slot) * hidden_int4 + vi));
        }
      }

#pragma unroll
      for (int j = 0; j < kVecsPerLane; ++j) {
        const int vi = chunk_base + lane + j * 32;
        if (vi < chunk_end)
          dst[vi] = pack_acc(acc[j]);
      }
    }
  }
}

__global__ __launch_bounds__(kMegaKernelThreads, 1)
void combine_gather_sum_direct_chunked_kernel(const int4* __restrict__ slots,
                                              const int* __restrict__ token_slot_list,
                                              int4* __restrict__ out,
                                              int num_tokens, int hidden_int4,
                                              int topk, int nh) {
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane = tid & 31;
  if (warp_id >= kNumSenderWarps) return;

  for (int token = blockIdx.x * kNumSenderWarps + warp_id;
       token < num_tokens;
       token += gridDim.x * kNumSenderWarps) {
    int4* dst = out + static_cast<size_t>(token) * hidden_int4;
    for (int chunk_base = 0; chunk_base < hidden_int4; chunk_base += kChunkInt4) {
      const int chunk_end = min(chunk_base + kChunkInt4, hidden_int4);
      float2 acc[kVecsPerLane][kPairsPerInt4];
#pragma unroll
      for (int j = 0; j < kVecsPerLane; ++j) {
#pragma unroll
        for (int p = 0; p < kPairsPerInt4; ++p)
          acc[j][p] = make_float2(0.0f, 0.0f);
      }

      for (int h = 0; h < nh; ++h) {
        const int slot = token_slot_list[token * topk + h];
#pragma unroll
        for (int j = 0; j < kVecsPerLane; ++j) {
          const int vi = chunk_base + lane + j * 32;
          if (vi < chunk_end)
            accum_int4(acc[j], ld_nc_int4(slots + static_cast<size_t>(slot) * hidden_int4 + vi));
        }
      }

#pragma unroll
      for (int j = 0; j < kVecsPerLane; ++j) {
        const int vi = chunk_base + lane + j * 32;
        if (vi < chunk_end)
          dst[vi] = pack_acc(acc[j]);
      }
    }
  }
}

__global__ __launch_bounds__(kMegaKernelThreads, 1)
void gather_reduce_current_1sm_kernel(const int4* __restrict__ slots,
                                      const int* __restrict__ token_slot_list,
                                      int4* __restrict__ out,
                                      int num_tokens, int hidden_int4,
                                      int topk, int nh) {
  const int tid = threadIdx.x;
  for (int token = blockIdx.x; token < num_tokens; token += gridDim.x) {
    int4* dst = out + static_cast<size_t>(token) * hidden_int4;
    for (int vi = tid; vi < hidden_int4; vi += blockDim.x) {
      float2 acc[kPairsPerInt4];
#pragma unroll
      for (int p = 0; p < kPairsPerInt4; ++p)
        acc[p] = make_float2(0.0f, 0.0f);
      for (int h = 0; h < nh; ++h) {
        const int slot = token_slot_list[token * topk + h];
        accum_int4(acc, ld_nc_int4(slots + static_cast<size_t>(slot) * hidden_int4 + vi));
      }
      dst[vi] = pack_acc(acc);
    }
    __syncthreads();
  }
}

__global__ __launch_bounds__(kMegaKernelThreads, 1)
void gather_reduce_deepgemm_1sm_kernel(const int4* __restrict__ slots,
                                       const int* __restrict__ token_slot_list,
                                       int4* __restrict__ out,
                                       int num_tokens, int hidden_int4,
                                       int topk, int nh) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  constexpr int kNumWarps = kMegaKernelThreads / 32;
  const int chunks_per_token = (hidden_int4 + kChunkInt4 - 1) / kChunkInt4;
  const int total_work = num_tokens * chunks_per_token;

  for (int work = blockIdx.x * kNumWarps + warp_id;
       work < total_work;
       work += gridDim.x * kNumWarps) {
    const int token = work / chunks_per_token;
    const int chunk = work - token * chunks_per_token;
    const int chunk_base = chunk * kChunkInt4;
    const int chunk_end = min(chunk_base + kChunkInt4, hidden_int4);
    int4* dst = out + static_cast<size_t>(token) * hidden_int4;

    float2 acc[kVecsPerLane][kPairsPerInt4];
#pragma unroll
    for (int j = 0; j < kVecsPerLane; ++j) {
#pragma unroll
      for (int p = 0; p < kPairsPerInt4; ++p)
        acc[j][p] = make_float2(0.0f, 0.0f);
    }

    for (int h = 0; h < nh; ++h) {
      const int slot = token_slot_list[token * topk + h];
#pragma unroll
      for (int j = 0; j < kVecsPerLane; ++j) {
        const int vi = chunk_base + lane + j * 32;
        if (vi < chunk_end)
          accum_int4_dg(acc[j], ld_nc_int4(slots + static_cast<size_t>(slot) * hidden_int4 + vi));
      }
    }

#pragma unroll
    for (int j = 0; j < kVecsPerLane; ++j) {
      const int vi = chunk_base + lane + j * 32;
      if (vi < chunk_end)
        dst[vi] = pack_acc(acc[j]);
    }
  }
}

template <class F>
float median_ms(F fn, int warm, int it) {
  for (int i = 0; i < warm; ++i) fn();
  CHECK_CUDA(cudaDeviceSynchronize());
  std::vector<float> ts(it);
  cudaEvent_t b, e;
  CHECK_CUDA(cudaEventCreate(&b));
  CHECK_CUDA(cudaEventCreate(&e));
  for (int i = 0; i < it; ++i) {
    CHECK_CUDA(cudaEventRecord(b));
    fn();
    CHECK_CUDA(cudaEventRecord(e));
    CHECK_CUDA(cudaEventSynchronize(e));
    CHECK_CUDA(cudaEventElapsedTime(&ts[i], b, e));
  }
  CHECK_CUDA(cudaEventDestroy(b));
  CHECK_CUDA(cudaEventDestroy(e));
  std::sort(ts.begin(), ts.end());
  return ts[it / 2];
}

static void cpu_ref(const std::vector<__nv_bfloat16>& slots,
                    const std::vector<int>& token_slot_list,
                    std::vector<__nv_bfloat16>& ref,
                    int num_tokens, int hidden, int topk, int nh) {
  for (int t = 0; t < num_tokens; ++t) {
    for (int i = 0; i < hidden; ++i) {
      float acc = 0.0f;
      for (int h = 0; h < nh; ++h) {
        int slot = token_slot_list[t * topk + h];
        acc += __bfloat162float(slots[static_cast<size_t>(slot) * hidden + i]);
      }
      ref[static_cast<size_t>(t) * hidden + i] = __float2bfloat16(acc);
    }
  }
}

static float max_abs_diff(const std::vector<__nv_bfloat16>& a,
                          const std::vector<__nv_bfloat16>& b) {
  float m = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    float da = __bfloat162float(a[i]);
    float db = __bfloat162float(b[i]);
    m = std::max(m, std::fabs(da - db));
  }
  return m;
}

int main(int argc, char** argv) {
  int num_tokens = argc > 1 ? std::atoi(argv[1]) : 4096;
  int hidden = argc > 2 ? std::atoi(argv[2]) : 4096;
  int topk = argc > 3 ? std::atoi(argv[3]) : 8;
  int nh = argc > 4 ? std::atoi(argv[4]) : 2;
  int blocks = argc > 5 ? std::atoi(argv[5]) : 16;
  std::string mode = argc > 6 ? argv[6] : "*";

  if (num_tokens <= 0 || hidden <= 0 || hidden % kElemsPerInt4 != 0 ||
      topk <= 0 || nh <= 0 || nh > topk || blocks <= 0) {
    std::fprintf(stderr, "invalid args: tokens=%d hidden=%d topk=%d nh=%d blocks=%d\n",
                 num_tokens, hidden, topk, nh, blocks);
    return 1;
  }

  const int hidden_int4 = hidden / kElemsPerInt4;
  const int num_slots = num_tokens * topk;
  std::fprintf(stderr,
               "combine gather-sum bench: tokens=%d hidden=%d hidden_int4=%d topk=%d nh=%d blocks=%d blockDim=%d sender_warps=%d mode=%s\n",
               num_tokens, hidden, hidden_int4, topk, nh, blocks, kMegaKernelThreads, kNumSenderWarps, mode.c_str());

  std::mt19937 gen(0);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<__nv_bfloat16> h_slots(static_cast<size_t>(num_slots) * hidden);
  for (auto& x : h_slots) x = __float2bfloat16(dist(gen));

  std::vector<int> h_slot_list(static_cast<size_t>(num_tokens) * topk);
  for (int t = 0; t < num_tokens; ++t) {
    for (int h = 0; h < topk; ++h)
      h_slot_list[t * topk + h] = t * topk + h;
    std::shuffle(h_slot_list.begin() + static_cast<size_t>(t) * topk,
                 h_slot_list.begin() + static_cast<size_t>(t + 1) * topk, gen);
  }

  __nv_bfloat16* d_slots_bf16 = nullptr;
  int* d_slot_list = nullptr;
  int4* d_out_a = nullptr;
  int4* d_out_b = nullptr;
  int4* d_out_c = nullptr;
  int4* d_out_d = nullptr;
  int4* d_out_e = nullptr;
  int4* d_out_f = nullptr;
  CHECK_CUDA(cudaMalloc(&d_slots_bf16, h_slots.size() * sizeof(__nv_bfloat16)));
  CHECK_CUDA(cudaMalloc(&d_slot_list, h_slot_list.size() * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_out_a, static_cast<size_t>(num_tokens) * hidden_int4 * sizeof(int4)));
  CHECK_CUDA(cudaMalloc(&d_out_b, static_cast<size_t>(num_tokens) * hidden_int4 * sizeof(int4)));
  CHECK_CUDA(cudaMalloc(&d_out_c, static_cast<size_t>(num_tokens) * hidden_int4 * sizeof(int4)));
  CHECK_CUDA(cudaMalloc(&d_out_d, static_cast<size_t>(num_tokens) * hidden_int4 * sizeof(int4)));
  CHECK_CUDA(cudaMalloc(&d_out_e, static_cast<size_t>(num_tokens) * hidden_int4 * sizeof(int4)));
  CHECK_CUDA(cudaMalloc(&d_out_f, static_cast<size_t>(num_tokens) * hidden_int4 * sizeof(int4)));
  CHECK_CUDA(cudaMemcpy(d_slots_bf16, h_slots.data(), h_slots.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_slot_list, h_slot_list.data(), h_slot_list.size() * sizeof(int), cudaMemcpyHostToDevice));

  const int4* d_slots = reinterpret_cast<const int4*>(d_slots_bf16);
  auto run_a = [&] {
    combine_gather_sum_staged_kernel<<<blocks, kMegaKernelThreads>>>(d_slots, d_slot_list, d_out_a, num_tokens, hidden_int4, topk, nh);
  };
  auto run_b = [&] {
    combine_gather_sum_direct_scalar_kernel<<<blocks, kMegaKernelThreads>>>(d_slots, d_slot_list, d_out_b, num_tokens, hidden_int4, topk, nh);
  };
  auto run_c = [&] {
    combine_gather_sum_direct_chunked_kernel<<<blocks, kMegaKernelThreads>>>(d_slots, d_slot_list, d_out_c, num_tokens, hidden_int4, topk, nh);
  };
  auto run_d = [&] {
    switch (nh) {
      case 1:
        combine_gather_sum_direct_chunked_nh_kernel<1><<<blocks, kMegaKernelThreads>>>(d_slots, d_slot_list, d_out_d, num_tokens, hidden_int4, topk);
        break;
      case 2:
        combine_gather_sum_direct_chunked_nh_kernel<2><<<blocks, kMegaKernelThreads>>>(d_slots, d_slot_list, d_out_d, num_tokens, hidden_int4, topk);
        break;
      case 3:
        combine_gather_sum_direct_chunked_nh_kernel<3><<<blocks, kMegaKernelThreads>>>(d_slots, d_slot_list, d_out_d, num_tokens, hidden_int4, topk);
        break;
      case 4:
        combine_gather_sum_direct_chunked_nh_kernel<4><<<blocks, kMegaKernelThreads>>>(d_slots, d_slot_list, d_out_d, num_tokens, hidden_int4, topk);
        break;
      default:
        combine_gather_sum_direct_chunked_kernel<<<blocks, kMegaKernelThreads>>>(d_slots, d_slot_list, d_out_d, num_tokens, hidden_int4, topk, nh);
        break;
    }
  };
  auto run_e = [&] {
    gather_reduce_current_1sm_kernel<<<blocks, kMegaKernelThreads>>>(d_slots, d_slot_list, d_out_e, num_tokens, hidden_int4, topk, nh);
  };
  auto run_f = [&] {
    gather_reduce_deepgemm_1sm_kernel<<<blocks, kMegaKernelThreads>>>(d_slots, d_slot_list, d_out_f, num_tokens, hidden_int4, topk, nh);
  };

  run_a();
  run_b();
  run_c();
  run_d();
  run_e();
  run_f();
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  std::vector<__nv_bfloat16> h_ref(static_cast<size_t>(num_tokens) * hidden);
  cpu_ref(h_slots, h_slot_list, h_ref, num_tokens, hidden, topk, nh);
  std::vector<__nv_bfloat16> h_out_a(h_ref.size()), h_out_b(h_ref.size()), h_out_c(h_ref.size()), h_out_d(h_ref.size());
  std::vector<__nv_bfloat16> h_out_e(h_ref.size()), h_out_f(h_ref.size());
  CHECK_CUDA(cudaMemcpy(h_out_a.data(), d_out_a, h_out_a.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_out_b.data(), d_out_b, h_out_b.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_out_c.data(), d_out_c, h_out_c.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_out_d.data(), d_out_d, h_out_d.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_out_e.data(), d_out_e, h_out_e.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_out_f.data(), d_out_f, h_out_f.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
  std::fprintf(stderr,
               "correctness: staged=%.6g direct_scalar=%.6g direct_chunked=%.6g direct_nh=%.6g current_1sm=%.6g dg_1sm=%.6g E_vs_F=%.6g\n",
               max_abs_diff(h_ref, h_out_a), max_abs_diff(h_ref, h_out_b),
               max_abs_diff(h_ref, h_out_c), max_abs_diff(h_ref, h_out_d),
               max_abs_diff(h_ref, h_out_e), max_abs_diff(h_ref, h_out_f),
               max_abs_diff(h_out_e, h_out_f));

  auto wants = [&](char which) {
    if (mode == "*" || mode == "X" || mode == "x")
      return true;
    char lower = static_cast<char>(which + ('a' - 'A'));
    return mode.find(which) != std::string::npos || mode.find(lower) != std::string::npos;
  };
  float ta = 0, tb = 0, tc = 0, td = 0, te = 0, tf = 0;
  if (wants('A')) ta = median_ms(run_a, 10, 50);
  if (wants('B')) tb = median_ms(run_b, 10, 50);
  if (wants('C')) tc = median_ms(run_c, 10, 50);
  if (wants('D')) td = median_ms(run_d, 10, 50);
  if (wants('E')) te = median_ms(run_e, 10, 50);
  if (wants('F')) tf = median_ms(run_f, 10, 50);

  const double bytes_read = static_cast<double>(num_tokens) * nh * hidden * sizeof(__nv_bfloat16);
  const double bytes_write = static_cast<double>(num_tokens) * hidden * sizeof(__nv_bfloat16);
  auto print = [&](const char* name, float ms) {
    if (ms <= 0) return;
    double gbps = (bytes_read + bytes_write) / (ms * 1e-3) / 1e9;
    std::fprintf(stderr, "%-34s median %.4f ms  %.1f GB/s\n", name, ms, gbps);
  };
  print("A staged smem", ta);
  print("B direct scalar", tb);
  print("C direct chunked", tc);
  print("D direct chunked NH", td);
  print("E current gather 1SM", te);
  print("F DeepGEMM-style gather 1SM", tf);

  CHECK_CUDA(cudaFree(d_slots_bf16));
  CHECK_CUDA(cudaFree(d_slot_list));
  CHECK_CUDA(cudaFree(d_out_a));
  CHECK_CUDA(cudaFree(d_out_b));
  CHECK_CUDA(cudaFree(d_out_c));
  CHECK_CUDA(cudaFree(d_out_d));
  CHECK_CUDA(cudaFree(d_out_e));
  CHECK_CUDA(cudaFree(d_out_f));

  return 0;
}
