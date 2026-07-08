// megakernel_compute_umma_fp8.cuh -- FP8 descriptor/state plumbing for MK-v7 UMMA compute.
//
// This header intentionally contains only the FP8 TMA atom containers and host-side
// descriptor builders. The persistent FP8 mainloop will be wired separately after
// the megakernel state can carry FP8 token, weight, and scale pointers end to end.

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>

#include <cstddef>

#include <cutlass/numeric_types.h>

#include "exception.cuh"

namespace deep_ep {
namespace megakernel {
namespace umma_fp8 {

using ElemAB = cutlass::float_e4m3_t;
using ScalePack = uint32_t;

static constexpr int kMaxLocalExperts = 64;
static constexpr uint32_t kDgFp8BlockM = 128;
static constexpr uint32_t kDgFp8BlockN = 128;
static constexpr uint32_t kDgFp8BlockK = 128;
static constexpr uint32_t kDgFp8SwizzleA = 128;
static constexpr uint32_t kDgFp8SwizzleB = 128;
static constexpr uint32_t kDgFp8SwizzleCD = 128;
static constexpr uint32_t kDgFp8GranKA = 128;
static constexpr uint32_t kDgFp8GranKB = 128;
static constexpr uint32_t kDgFp8KAlign = 128;
static constexpr uint32_t kDgFp8StoreBlockM = 128;
static constexpr uint32_t kDgFp8StoreBlockN = kDgFp8SwizzleCD / sizeof(__nv_bfloat16);

// Per-expert FP8 weight descriptors. W_gateup_fp8 keeps the existing pairwise
// interleaved layout [E, 2I, d] with rows [g0,u0,g1,u1,...]. Scale descriptors
// are packed UE8M0/uint32, shaped [outer, ceil(K / (gran_k * 4))].
struct ComputeFp8TmaAtoms {
    int num_experts;
    int intermediate_dim;
    int hidden_dim;
    int gateup_scale_k_packed;
    CUtensorMap wgateup[kMaxLocalExperts];
    CUtensorMap wgateup_sf[kMaxLocalExperts];
};

struct ComputeFp8DownTmaAtoms {
    int num_experts;
    int hidden_dim;
    int intermediate_dim;
    int down_scale_k_packed;
    CUtensorMap wdown[kMaxLocalExperts];
    CUtensorMap wdown_sf[kMaxLocalExperts];
};

// Per-compute-group FP8 activation descriptors. recv/input quantization is not
// performed in this patch; these fields reserve the descriptor contract for the
// later FP8 gather/quantize step and downstream FP8 mainloop.
struct InputFp8TmaAtom_t {
    CUtensorMap a;
    CUtensorMap a_sf;
    CUtensorMap act;
    CUtensorMap act_sf;
    CUtensorMap act_cd;
    CUtensorMap down_cd;
};

inline CUtensorMap dg_make_fp8_tma_2d(const void* ptr, CUtensorMapDataType dtype,
                                      int gmem_inner, int gmem_outer,
                                      int smem_inner, int smem_outer,
                                      int gmem_outer_stride_elems, int elem_size,
                                      int swizzle_bytes) {
    CUtensorMap tm;
    int si = smem_inner;
    if (swizzle_bytes != 0) si = swizzle_bytes / elem_size;
    const cuuint64_t gdims[2] = { (cuuint64_t)gmem_inner, (cuuint64_t)gmem_outer };
    const cuuint32_t sdims[2] = { (cuuint32_t)si, (cuuint32_t)smem_outer };
    const cuuint64_t gstr[1] = { (cuuint64_t)gmem_outer_stride_elems * elem_size };
    const cuuint32_t estr[2] = { 1, 1 };
    CUtensorMapSwizzle sw =
        swizzle_bytes == 128 ? CU_TENSOR_MAP_SWIZZLE_128B :
        swizzle_bytes == 64  ? CU_TENSOR_MAP_SWIZZLE_64B  :
        swizzle_bytes == 32  ? CU_TENSOR_MAP_SWIZZLE_32B  : CU_TENSOR_MAP_SWIZZLE_NONE;
    CUresult r = cuTensorMapEncodeTiled(
        &tm, dtype, 2, (void*)ptr, gdims, gstr, sdims, estr,
        CU_TENSOR_MAP_INTERLEAVE_NONE, sw,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    EP_HOST_ASSERT(r == CUDA_SUCCESS);
    return tm;
}

inline int dg_fp8_scale_k_packed(int k, int gran_k) {
    return (k + gran_k * 4 - 1) / (gran_k * 4);
}

inline CUtensorMap dg_make_fp8_a_desc(const ElemAB* a, int M, int K) {
    return dg_make_fp8_tma_2d(a, CU_TENSOR_MAP_DATA_TYPE_UINT8, K, M,
                              kDgFp8BlockK, kDgFp8BlockM, K,
                              sizeof(ElemAB), kDgFp8SwizzleA);
}

inline CUtensorMap dg_make_fp8_b_desc(const ElemAB* b, int N, int K) {
    return dg_make_fp8_tma_2d(b, CU_TENSOR_MAP_DATA_TYPE_UINT8, K, N,
                              kDgFp8BlockK, kDgFp8BlockN, K,
                              sizeof(ElemAB), kDgFp8SwizzleB);
}

inline CUtensorMap dg_make_fp8_scale_desc(const ScalePack* sf, int outer, int scale_k_packed) {
    return dg_make_fp8_tma_2d(sf, CU_TENSOR_MAP_DATA_TYPE_UINT32, outer, scale_k_packed,
                              kDgFp8BlockM, 1, outer, sizeof(ScalePack), 0);
}

inline CUtensorMap dg_make_fp8_cd_desc(const __nv_bfloat16* d, int M, int N) {
    return dg_make_fp8_tma_2d(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, N, M,
                              kDgFp8StoreBlockN, kDgFp8StoreBlockM, N,
                              sizeof(__nv_bfloat16), kDgFp8SwizzleCD);
}

inline void build_compute_fp8_tma_atoms(ComputeFp8TmaAtoms& atoms,
                                        const ElemAB* W_gateup_fp8,
                                        const ScalePack* W_gateup_sf,
                                        int num_local_experts,
                                        int intermediate_dim,
                                        int hidden_dim) {
    EP_HOST_ASSERT(num_local_experts <= kMaxLocalExperts);
    atoms.num_experts = num_local_experts;
    atoms.intermediate_dim = intermediate_dim;
    atoms.hidden_dim = hidden_dim;
    atoms.gateup_scale_k_packed = dg_fp8_scale_k_packed(hidden_dim, kDgFp8GranKB);
    for (int e = 0; e < num_local_experts; ++e) {
        atoms.wgateup[e] = dg_make_fp8_b_desc(
            W_gateup_fp8 + (size_t)e * 2 * intermediate_dim * hidden_dim,
            2 * intermediate_dim, hidden_dim);
        atoms.wgateup_sf[e] = dg_make_fp8_scale_desc(
            W_gateup_sf + (size_t)e * 2 * intermediate_dim * atoms.gateup_scale_k_packed,
            2 * intermediate_dim, atoms.gateup_scale_k_packed);
    }
}

inline void build_compute_fp8_down_tma_atoms(ComputeFp8DownTmaAtoms& atoms,
                                             const ElemAB* W_down_fp8,
                                             const ScalePack* W_down_sf,
                                             int num_local_experts,
                                             int hidden_dim,
                                             int intermediate_dim) {
    EP_HOST_ASSERT(num_local_experts <= kMaxLocalExperts);
    atoms.num_experts = num_local_experts;
    atoms.hidden_dim = hidden_dim;
    atoms.intermediate_dim = intermediate_dim;
    atoms.down_scale_k_packed = dg_fp8_scale_k_packed(intermediate_dim, kDgFp8GranKB);
    for (int e = 0; e < num_local_experts; ++e) {
        atoms.wdown[e] = dg_make_fp8_b_desc(
            W_down_fp8 + (size_t)e * hidden_dim * intermediate_dim,
            hidden_dim, intermediate_dim);
        atoms.wdown_sf[e] = dg_make_fp8_scale_desc(
            W_down_sf + (size_t)e * hidden_dim * atoms.down_scale_k_packed,
            hidden_dim, atoms.down_scale_k_packed);
    }
}

inline InputFp8TmaAtom_t make_input_fp8_group_atoms(const ElemAB* input_fp8,
                                                    const ScalePack* input_sf,
                                                    const ElemAB* act_fp8,
                                                    const ScalePack* act_sf,
                                                    const __nv_bfloat16* act_bf16,
                                                    const __nv_bfloat16* down_bf16,
                                                    int M,
                                                    int hidden_dim,
                                                    int intermediate_dim) {
    InputFp8TmaAtom_t atom{};
    const int input_scale_k_packed = dg_fp8_scale_k_packed(hidden_dim, kDgFp8GranKA);
    const int act_scale_k_packed = dg_fp8_scale_k_packed(intermediate_dim, kDgFp8GranKA);
    atom.a = dg_make_fp8_a_desc(input_fp8, M, hidden_dim);
    atom.a_sf = dg_make_fp8_scale_desc(input_sf, M, input_scale_k_packed);
    atom.act = dg_make_fp8_a_desc(act_fp8, M, intermediate_dim);
    atom.act_sf = dg_make_fp8_scale_desc(act_sf, M, act_scale_k_packed);
    atom.act_cd = dg_make_fp8_cd_desc(act_bf16, M, intermediate_dim);
    atom.down_cd = dg_make_fp8_cd_desc(down_bf16, M, hidden_dim);
    return atom;
}

}  // namespace umma_fp8
}  // namespace megakernel
}  // namespace deep_ep
