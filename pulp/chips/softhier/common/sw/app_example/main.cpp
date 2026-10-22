#include <vector>
#include <algorithm>
#include <iostream>
#include <string>

#include "softhier.hpp"
#include "benchmark_layout.hpp"

template<typename T>
inline void correctness_check(size_t N, size_t M, size_t K, T* A, T* B, T* C_out){
    size_t errors = 0;
    double max_error = 0.0;

    constexpr double ATOL = 1e-3;
    constexpr double RTOL = 1e-3;

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            double expected = 0.0;

            for (size_t k = 0; k < K; ++k) {
                expected += static_cast<double>(A[i * K + k]) * static_cast<double>(B[k * N + j]);
            }

            const double actual = static_cast<double>(C_out[i * N + j]);
            const double diff = __builtin_fabs(actual - expected);
            const double tolerance = ATOL + RTOL * __builtin_fabs(expected);

            if (diff > max_error)
                max_error = diff;

            if (!(diff <= tolerance)) {
                if (errors < 10) {
                    flex_print("Mismatch: expected=");
                    flex_print_int(expected);
                    flex_print(" actual=");
                    flex_print_int(actual);
                    flex_print("\n");
                }
                ++errors;
            }
        }
    }

    flex_print("GEMM verification: ");
    flex_print_int(errors);
    flex_print(" errors, max error=");
    flex_print_int(max_error);
    flex_print("\n");
}

template<typename C, typename Func>
void benchmark(size_t N, size_t M, size_t K, Func F, const char* name){
    hwpe_cg_enable();

    using T = typename C::data_type;

    const uint32_t cid = flex_get_cluster_id();
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

    hwpe_soft_clear();

    flex_global_barrier_polling();

    // Start Timer
    flex_timer_start();

    // Run Function
    F(A, B, C_out, M, N, K);

    // Stop Timer
    flex_timer_end();

    flex_global_barrier_polling();

    if (master){
        flex_print("Running Correctness Test\n");
        correctness_check(N, M, K, A, B, C_out);
    }

    hwpe_cg_disable();
}

int main()
{
    flex_global_barrier_init();
    redmule_irq_init();
    
    using C = config<256, 256, 128, 3, 4, 4, true, Float16>;

    benchmark<C>(1024, 1024, 512, mm_summa<C>, "mm_singledev");

    flex_eoc_all(0);
    
    return 0;
}