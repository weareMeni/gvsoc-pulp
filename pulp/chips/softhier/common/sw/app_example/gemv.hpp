
#pragma once

#include <cstddef>
#include <cstdint>
#include <riscv_vector.h>

template<typename C>
inline int ClusterPipeline<C>::run(
    size_t in_slot,
    size_t out_slot,
    bool first
) {
    using T = typename C::data_type;
    static_assert(sizeof(T) == sizeof(_Float16),
                  "Expected 16-bit FP storage");

    const T* A = A_tiles[in_slot].ptr();
    const T* x = B_tiles[in_slot].ptr();
    T* y = C_tiles[out_slot].ptr();

    // Each vector lane computes a different output row.
    for (size_t m = 0; m < C::M;) {
        const size_t vl = __riscv_vsetvl_e16m1(C::M - m);

        vfloat32m2_t acc;

        if (first) {
            acc = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        } else {
            // Restore previous FP16 partial results as FP32.
            const vfloat16m1_t previous =
                __riscv_vreinterpret_v_u16m1_f16m1(
                    __riscv_vle16_v_u16m1(y + m, vl));

            const vfloat16m1_t ones =
                __riscv_vfmv_v_f_f16m1((_Float16)1.0f, vl);

            acc = __riscv_vfwmul_vv_f32m2(
                previous, ones, vl);
        }

        // A is column-major: A[k*C::M + m].
        // Accumulate across K without horizontal reductions.
        for (size_t k = 0; k < C::K; ++k) {
            _Float16 xk;
            __builtin_memcpy(&xk, x + k, sizeof(xk));

            const vfloat16m1_t va =
                __riscv_vreinterpret_v_u16m1_f16m1(
                    __riscv_vle16_v_u16m1(
                        A + k*C::M + m, vl));

            acc = __riscv_vfwmacc_vf_f32m2(
                acc, xk, va, vl);
        }

        // Store accumulated results back into the C tile.
        const vfloat16m1_t result =
            __riscv_vfncvt_f_f_w_f16m1(acc, vl);

        __riscv_vse16_v_u16m1(
            y + m,
            __riscv_vreinterpret_v_f16m1_u16m1(result),
            vl);

        m += vl;
    }

    return 0;
}
