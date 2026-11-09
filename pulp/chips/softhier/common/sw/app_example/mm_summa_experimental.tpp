
#pragma once
#include "gemm.hpp"
#include <cstddef>
#include <cstdint>
#include <cstdio>

#ifndef MM_SUMMA_TRACE
#define MM_SUMMA_TRACE 1
#endif
#ifndef MM_SUMMA_TRACE_CLUSTER
#define MM_SUMMA_TRACE_CLUSTER 0
#endif
#ifndef MM_SUMMA_TRACE_CAPACITY
#define MM_SUMMA_TRACE_CAPACITY 128
#endif

namespace summa_detail {

template<class C> struct Ring {
    volatile uint32_t ready_a[C::LOAD_DEPTH] = {}, ready_b[C::LOAD_DEPTH] = {};
    volatile uint32_t pulled_a[C::LOAD_DEPTH] = {}, pulled_b[C::LOAD_DEPTH] = {};
};
struct Shared { uint32_t base = 0, retired = 0; };

template<class C> static ClusterPipeline<C> gemm __attribute__((section(".l1")));
template<class C> static Ring<C> ring __attribute__((section(".l1")));
template<class C> static Shared shared __attribute__((section(".l1")));

enum Event : uint8_t {
    A_LOAD=1, B_LOAD, A_SHIFT, B_SHIFT,
    A_DONE, B_DONE, INPUT_READY, REDMULE_SUBMIT,
    C_STORE, C_DONE, WAIT_IN, WAIT_C
};

#if MM_SUMMA_TRACE
struct Record {
    uint32_t cycle, gen, info;
    uint16_t event, slot;
};

template<class C> struct Trace {
    Record data[2][MM_SUMMA_TRACE_CAPACITY];
    uint32_t count[2] = {}, dropped[2] = {};
};

template<class C> static Trace<C> trace __attribute__((section(".l1")));

template<class C>
inline void log(uint32_t cid, uint32_t core, Event event,
                uint32_t gen, size_t slot, uint32_t info = 0) {
    if (cid != MM_SUMMA_TRACE_CLUSTER) return;
    auto& t = trace<C>;
    const uint32_t i = t.count[core];
    if (i >= MM_SUMMA_TRACE_CAPACITY) { ++t.dropped[core]; return; }

    uint32_t cycle;
    asm volatile("csrr %0, mcycle" : "=r"(cycle));
    t.data[core][i] = {cycle, gen, info, (uint16_t)event, (uint16_t)slot};
    t.count[core] = i + 1;
}
#else
template<class C>
inline void log(uint32_t, uint32_t, Event, uint32_t, size_t, uint32_t = 0) {}
#endif

inline void fence() { asm volatile("fence rw, rw" ::: "memory"); }
inline void pause() { asm volatile("" ::: "memory"); }

template<class C> inline uint32_t off(const void* p) {
    return (uint32_t)((uintptr_t)p - (uintptr_t)local(0));
}
template<class C> inline volatile uint32_t* remote(FlexPosition2D peer, const Ring<C>& r, size_t field, size_t slot) {
    return (volatile uint32_t*)remote_pos(peer, off<C>(&r) + field + slot * sizeof(uint32_t));
}

} // namespace summa_detail

template<class C>
void mm_summa_threaded(typename C::data_type* A,
                       typename C::data_type* B,
                       typename C::data_type* C_out,
                       size_t M, size_t N, size_t K) {
    using namespace summa_detail;
    using T = typename C::data_type;

    static_assert(C::LOAD_DEPTH >= 2 && C::OUTPUT_DEPTH >= 1,
                  "Insufficient pipeline buffering");

    const uint32_t cid = flex_get_cluster_id(), core = flex_get_core_id();
    const size_t mt = M/C::M, nt = N/C::N, kt = K/C::K, G = C::GRID_ROWS;

    if (!mt || !nt || !kt || M%C::M || N%C::N || K%C::K ||
        mt%G || nt%G || kt%G) {
        if (cid == 0 && core == 0) flex_print("SUMMA: invalid dimensions\n");
        return;
    }

    const size_t mg_count = mt/G, ng_count = nt/G, kr_count = kt/G;
    const size_t total = mg_count * ng_count;
    const auto pos = get_pos(cid), left = left_pos(pos), bottom = bottom_pos(pos);
    const size_t aligned = (pos.y + G - pos.x)%G;

    auto& p = gemm<C>;
    auto& s = p.sync;
    auto& r = ring<C>;
    auto& w = shared<C>;

    auto task = [&](size_t mg, size_t ng) -> TileTask<T> {
        const size_t m = mg*G + pos.y, n = ng*G + pos.x;
        return {A + m*C::M*K, B + n*C::N,
                C_out + m*C::M*N + n*C::N, K, N, N, kt};
    };
    auto rw = [&](FlexPosition2D peer, size_t field, size_t slot) {
        return remote<C>(peer, r, field, slot);
    };
    constexpr size_t RA = offsetof(Ring<C>, ready_a);
    constexpr size_t RB = offsetof(Ring<C>, ready_b);
    constexpr size_t PA = offsetof(Ring<C>, pulled_a);
    constexpr size_t PB = offsetof(Ring<C>, pulled_b);

    if (core == 0) {
        r = Ring<C>{};
        s = PipelineSync<C>{};
        w.base = redmule_completed_jobs;
        w.retired = 0;
#if MM_SUMMA_TRACE
        if (cid == MM_SUMMA_TRACE_CLUSTER) trace<C> = Trace<C>{};
#endif
    }

    flex_global_barrier_polling(); // Both cores participate.

    if (core == 0) {
        // RedMulE worker
        size_t gen = 0, in = 0, tile = 0;

        for (size_t mg = 0; mg < mg_count; ++mg)
            for (size_t ng = 0; ng < ng_count; ++ng, ++tile) {
                const size_t out = tile % C::OUTPUT_DEPTH;

                if (tile >= C::OUTPUT_DEPTH) {
                    const uint32_t required =
                        (uint32_t)(tile - C::OUTPUT_DEPTH + 1);

                    if (__atomic_load_n(&w.retired, __ATOMIC_ACQUIRE) < required)
                        log<C>(cid, core, WAIT_C, (uint32_t)tile+1, out, required);

                    while (__atomic_load_n(&w.retired, __ATOMIC_ACQUIRE) < required)
                        pause();
                }

                for (size_t k = 0; k < kt; ++k) {
                    ++gen;

                    if (__atomic_load_n(&s.inputs_arrived[in].completed,
                                        __ATOMIC_ACQUIRE) < gen)
                        log<C>(cid, core, WAIT_IN, (uint32_t)gen, in);

                    while (__atomic_load_n(&s.inputs_arrived[in].completed,
                                           __ATOMIC_ACQUIRE) < gen)
                        pause();

                    p.run(in, out, k == 0);
                    log<C>(cid, core, REDMULE_SUBMIT, (uint32_t)gen, in, (uint32_t)out);
                    in = (in + 1)%C::LOAD_DEPTH;
                }
            }

    } else if (core == 1) {
        // Single cooperative DMA scheduler
        const uint32_t base = w.base;
        size_t mg = 0, ng = 0, kr = 0, hop = 0, in = 0, src = 0;
        size_t next_store = 0, next_done = 0;
        uint32_t gen = 0;
        bool ap = false, bp = false, ad = false, bd = false, finished = false;

        auto reusable = [&](bool is_a) {
            size_t old = __atomic_load_n(&s.inputs_arrived[in].completed, __ATOMIC_ACQUIRE);
            return !old || (redmule_completed_jobs >= base + old &&
                 (old%G == 0 || (is_a ? r.pulled_a[in] : r.pulled_b[in]) >= old));
        };

        while (!finished || next_done < total) {
            // Output polls never block input progress.
            if (next_done < next_store) {
                size_t out = next_done % C::OUTPUT_DEPTH;
                if (bare_dma_done(s.output_dma_id[out])) {
                    fence();
                    ++next_done;

                    log<C>(cid, core, C_DONE, (uint32_t)next_done, out, s.output_dma_id[out]);

                    __atomic_store_n(&w.retired, (uint32_t)next_done, __ATOMIC_RELEASE);
                }
            }

            if (next_store < total &&
                next_store-next_done < C::OUTPUT_DEPTH &&
                redmule_completed_jobs >= base + (next_store+1)*kt) {
                size_t out = next_store % C::OUTPUT_DEPTH;
                auto t = task(next_store/ng_count, next_store%ng_count);

                p.submit_output(t.C, N, out);
                log<C>(cid, core, C_STORE, (uint32_t)next_store+1, out, s.output_dma_id[out]);
                ++next_store;
            }

            if (finished) continue;

            const uint32_t next = gen+1;
            const bool shift = hop != 0;
            const auto t = task(mg, ng);
            const size_t k = (kr*G+aligned)*C::K;

            // Submit A, independently of B.
            if (!ap && reusable(true)) {
                if (!shift) {
                    s.a_dma_id[in] = bare_dma_start_2d(
                        dma_addr(p.A_tiles[in].ptr()), dma_addr(t.A+k),
                        C::K*sizeof(T), C::K*sizeof(T),
                        t.lda*sizeof(T), C::M);

                    ap = true;
                    log<C>(cid, core, A_LOAD, next, in, s.a_dma_id[in]);

                } else if (*rw(left, RA, src) >= gen) {
                    s.a_dma_id[in] = flex_dma_async_pattern_round_shift_right(
                        off<C>(p.A_tiles[in].ptr()),
                        off<C>(p.A_tiles[src].ptr()),
                        C::M*C::K*sizeof(T));

                    ap = true;
                    log<C>(cid, core, A_SHIFT, next, in, s.a_dma_id[in]);
                }
            }

            // Submit B, independently of A.
            if (!bp && reusable(false)) {
                if (!shift) {
                    s.b_dma_id[in] = bare_dma_start_2d(
                        dma_addr(p.B_tiles[in].ptr()),
                        dma_addr(t.B+k*t.ldb),
                        C::N*sizeof(T), C::N*sizeof(T),
                        t.ldb*sizeof(T), C::K);

                    bp = true;
                    log<C>(cid, core, B_LOAD, next, in, s.b_dma_id[in]);

                } else if (*rw(bottom, RB, src) >= gen) {
                    s.b_dma_id[in] = flex_dma_async_pattern_round_shift_up(
                        off<C>(p.B_tiles[in].ptr()),
                        off<C>(p.B_tiles[src].ptr()),
                        C::K*C::N*sizeof(T));

                    bp = true;
                    log<C>(cid, core, B_SHIFT, next, in, s.b_dma_id[in]);
                }
            }

            // Retire A and B independently.
            if (ap && !ad && bare_dma_done(s.a_dma_id[in])) {
                fence();
                if (shift) *rw(left, PA, src) = gen;
                fence();
                r.ready_a[in] = next;
                ad = true;
                log<C>(cid, core, A_DONE, next, in, s.a_dma_id[in]);
            }

            if (bp && !bd && bare_dma_done(s.b_dma_id[in])) {
                fence();
                if (shift) *rw(bottom, PB, src) = gen;
                fence();
                r.ready_b[in] = next;
                bd = true;
                log<C>(cid, core, B_DONE, next, in, s.b_dma_id[in]);
            }

            if (!ad || !bd) continue;

            // Both operands are ready for RedMulE.
            __atomic_store_n(&s.inputs_arrived[in].completed,
                             (size_t)next, __ATOMIC_RELEASE);
            log<C>(cid, core, INPUT_READY, next, in);

            gen = next;
            src = in;
            in = (in+1)%C::LOAD_DEPTH;
            ap = bp = ad = bd = false;

            if (++hop == G) {
                hop = 0;
                if (++kr == kr_count) {
                    kr = 0;
                    if (++ng == ng_count) {
                        ng = 0;
                        if (++mg == mg_count) finished = true;
                    }
                }
            }
        }
    }

    flex_global_barrier_polling();
}


template<class C>
void mm_summa_threaded_dump_trace() {
#if MM_SUMMA_TRACE
    using namespace summa_detail;

    if (flex_get_cluster_id() != MM_SUMMA_TRACE_CLUSTER ||
        flex_get_core_id() != 0)
        return;

    const auto& t = trace<C>;

    printf("SUMMA,core,cycle,event,gen,slot,info\n");

    for (uint32_t core = 0; core < 2; ++core) {
        uint32_t n = t.count[core];

        printf("SUMMA_COUNT,core=%u,count=%u,dropped=%u\n",
               (unsigned)core,
               (unsigned)n,
               (unsigned)t.dropped[core]);

        // Protect against corrupted record counts.
        if (n > MM_SUMMA_TRACE_CAPACITY)
            n = MM_SUMMA_TRACE_CAPACITY;

        for (uint32_t i = 0; i < n; ++i) {
            const auto& e = t.data[core][i];

            // Numeric event IDs: no string pointer lookup.
            printf("SUMMA,%u,%u,%u,%u,%u,%u\n",
                   (unsigned)core,
                   (unsigned)e.cycle,
                   (unsigned)e.event,
                   (unsigned)e.gen,
                   (unsigned)e.slot,
                   (unsigned)e.info);
        }
    }
#endif
}

