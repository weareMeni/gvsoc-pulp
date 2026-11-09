#include <vector>
#include <algorithm>
#include <iostream>
#include <string>

#include <type_traits>
#include <cstdint>
#include <cstdio>

#include "softhier.hpp"
#include "benchmark_layout.hpp"

template<typename T>
inline void correctness_check(
    size_t N, size_t M, size_t K,
    const T* A, const T* B, const T* C,
    bool a_row_major = true, bool b_row_major = true
) {
    constexpr float ATOL = 0.01f, RTOL = 0.01f;
    constexpr size_t BLOCK = 1024;
    static float acc[BLOCK];  // Ordinary .bss, not .l1

    auto f32 = [](T v) -> float {
        if constexpr (std::is_same_v<T, uint16_t>) {
            _Float16 h;
            __builtin_memcpy(&h, &v, sizeof(h));
            return (float)h;
        }
        return (float)v;
    };

    size_t errors = 0;

    auto check = [&](size_t i, size_t j, float ref) {
        float got = f32(C[i*N + j]);
        float diff = __builtin_fabsf(got - ref);
        float tol = ATOL + RTOL * __builtin_fabsf(ref);

        if (!(diff <= tol)) {
            if (errors < 10)
                printf("Mismatch (%u,%u): actual=%d expected=%d "
                       "error=%d tolerance=%d (all x10000)\n",
                       (unsigned)i, (unsigned)j,
                       (int)(got * 10000.0f),
                       (int)(ref * 10000.0f),
                       (int)(diff * 10000.0f),
                       (int)(tol * 10000.0f));
            ++errors;
        }
    };

    printf("CHECK START M=%u N=%u K=%u\n",
           (unsigned)M, (unsigned)N, (unsigned)K);

    if (a_row_major) {
        for (size_t i = 0; i < M; ++i)
            for (size_t j = 0; j < N; ++j) {
                float sum = 0;
                for (size_t k = 0; k < K; ++k)
                    sum += f32(A[i*K+k]) *
                           f32(B[b_row_major ? k*N+j : j*K+k]);
                check(i, j, sum);
            }
    } else {
        for (size_t j = 0; j < N; ++j)
            for (size_t base = 0; base < M; base += BLOCK) {
                size_t rows = (M-base < BLOCK) ? M-base : BLOCK;

                for (size_t r = 0; r < rows; ++r) acc[r] = 0;

                for (size_t k = 0; k < K; ++k) {
                    float b = f32(B[b_row_major ? k*N+j : j*K+k]);
                    const T* a = A + k*M + base;

                    for (size_t r = 0; r < rows; ++r)
                        acc[r] += f32(a[r]) * b;

                    if ((k+1) % 128 == 0)
                        printf("CHECK K %u/%u\n",
                               (unsigned)(k+1), (unsigned)K);
                }

                for (size_t r = 0; r < rows; ++r)
                    check(base+r, j, acc[r]);
            }
    }

    printf("Verification: %u errors\n", (unsigned)errors);
}

template<typename C, typename Func>
void benchmark(size_t N, size_t M, size_t K, Func F, const char* name, bool a_row_major = true){
    hwpe_cg_enable();

    using T = typename C::data_type;

    const uint32_t cid = flex_get_cluster_id();
    const uint32_t core = flex_get_core_id();
    const bool master = (cid == 0 && flex_is_first_core());
    size_t hbm_offset = 0;

    T* A     = hbm_alloc<T>(cid, M * K, hbm_offset);
    T* B     = hbm_alloc<T>(cid, K * N, hbm_offset);
    T* C_out = hbm_alloc<T>(cid, M * N, hbm_offset);

    if (!A || !B || !C_out) {
        if (master) flex_print("[HBM ERROR] Allocation failed\n");
        return;
    }

    hbm_random_init(A, M * K, 42u + cid);
    hbm_random_init(B, K * N, 1234u + cid);

    if (master){
        flex_print("-------------------- M=");
        flex_print_int(M);
        flex_print(" N=");
        flex_print_int(N);
        flex_print(" K=");
        flex_print_int(K);
        flex_print(" --------------------\n");

        flex_print("Template: Mb=");
        flex_print_int(C::M);
        flex_print(" Nb=");
        flex_print_int(C::N);
        flex_print(" Kb=");
        flex_print_int(C::K);
        flex_print(" LOAD_DEPTH=");
        flex_print_int(C::LOAD_DEPTH);
        flex_print(" OVERLAP_STORE=");
        flex_print_int(C::OVERLAP_STORE);
        flex_print("\n");

        flex_print("Number of iterations per task: ");
        flex_print_int(K);
        flex_print(" / ");
        flex_print_int(C::K);
        flex_print("\n");

        flex_print("Function being benchmarked: ");
        flex_print(name);
        flex_print("\n");
    }

    if (core == 1)
        hwpe_soft_clear();

    flex_global_barrier_polling();

    // Start Timer
    flex_timer_start();

    // Run Function
    F(A, B, C_out, M, N, K);

    // Stop Timer
    flex_timer_end();

    mm_gemv_threaded_dump_trace<C>();

    flex_global_barrier_polling();

    if (master){
        flex_print("Running Correctness Test\n");
        correctness_check(N, M, K, A, B, C_out, a_row_major);
    }

    flex_global_barrier_polling();

    hwpe_cg_disable();
}

int main()
{
    flex_global_barrier_init();
    redmule_irq_init();
    
    using C = config<256, 1, 512, 3, 4, 4, true, Float16>;

    benchmark<C>(1, 1024, 1024, mm_gemv_threaded<C>, "mm_gemv_threaded", false);

    flex_eoc_all(0);
    
    return 0;
}