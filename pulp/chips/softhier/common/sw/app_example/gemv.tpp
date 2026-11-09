
#pragma once

// Threaded GEMV: A/X DMA overlap, compute/DMA overlap, and asynchronous Y DMA.
// Y DMA is kept separate from an unfinished A/X pair (the verified schedule).
// This is the .hpp implementing mm_gemv_threaded(), NOT gemv.tpp.
#include "softhier.hpp"
#include "gemv.hpp"
#include <cstddef>
#include <cstdint>
#include <cstdio>

#ifndef GEMV_TRACE
#define GEMV_TRACE 1
#endif
#ifndef GEMV_TRACE_CLUSTER
#define GEMV_TRACE_CLUSTER 0
#endif
#ifndef GEMV_TRACE_CAPACITY
#define GEMV_TRACE_CAPACITY 128
#endif
#ifndef GEMV_TRACE_AUTO_DUMP
#define GEMV_TRACE_AUTO_DUMP 1
#endif
#ifndef GEMV_PARTITION_ROWS
// Original benchmark contract: each cluster processes its own complete GEMV.
#define GEMV_PARTITION_ROWS 0
#endif

namespace gemv_detail {

struct Shared {
    uint32_t consumed = 0, computed = 0, retired = 0;
};

template<class C>
static ClusterPipeline<C> pipeline __attribute__((section(".l1")));

template<class C>
static Shared shared __attribute__((section(".l1")));

enum Event : uint8_t {
    A_LOAD=1, X_LOAD, A_DONE, X_DONE, INPUT_READY,
    VPU_BEGIN, VPU_END, INPUT_CONSUMED, OUTPUT_READY,
    Y_STORE, Y_DONE, WAIT_IN, WAIT_OUT, WAIT_SLOT
};

inline void fence() { asm volatile("fence rw, rw" ::: "memory"); }
inline void spin() { asm volatile("" ::: "memory"); }

#if GEMV_TRACE
struct Record {
    uint32_t cycle, gen, info;
    uint16_t event, slot;
};

template<class C>
struct Trace {
    Record data[2][GEMV_TRACE_CAPACITY];
    uint32_t count[2] = {}, dropped[2] = {};
    bool printed = false;
};

template<class C>
static Trace<C> trace __attribute__((section(".l1")));

inline const char* event_name(uint16_t event) {
    switch (event) {
        case A_LOAD: return "A_LOAD";
        case X_LOAD: return "X_LOAD";
        case A_DONE: return "A_DONE";
        case X_DONE: return "X_DONE";
        case INPUT_READY: return "INPUT_READY";
        case VPU_BEGIN: return "VPU_BEGIN";
        case VPU_END: return "VPU_END";
        case INPUT_CONSUMED: return "INPUT_CONSUMED";
        case OUTPUT_READY: return "OUTPUT_READY";
        case Y_STORE: return "Y_STORE";
        case Y_DONE: return "Y_DONE";
        case WAIT_IN: return "WAIT_IN";
        case WAIT_OUT: return "WAIT_OUT";
        case WAIT_SLOT: return "WAIT_SLOT";
        default: return "UNKNOWN";
    }
}

template<class C>
inline void log(uint32_t cid, uint32_t core, Event event,
                uint32_t gen, size_t slot, uint32_t info = 0) {
    if (cid != GEMV_TRACE_CLUSTER || core >= 2) return;

    auto& t = trace<C>;
    const uint32_t i = t.count[core];
    if (i >= GEMV_TRACE_CAPACITY) {
        ++t.dropped[core];
        return;
    }

    uint32_t cycle;
    asm volatile("csrr %0, mcycle" : "=r"(cycle));
    t.data[core][i] = {cycle, gen, info, (uint16_t)event, (uint16_t)slot};
    t.count[core] = i + 1;
}

// Merge both workers' records by mcycle, rather than dumping core 0 and
// core 1 in separate blocks. The kernel's final barrier precedes this dump.
template<class C>
inline void dump(uint32_t cid, uint32_t core) {
    if (cid != GEMV_TRACE_CLUSTER || core != 0) return;
    auto& t = trace<C>;
    if (t.printed) return;  // Avoid duplicate output if caller also dumps.
    t.printed = true;

    uint32_t n0 = t.count[0], n1 = t.count[1];
    if (n0 > GEMV_TRACE_CAPACITY) n0 = GEMV_TRACE_CAPACITY;
    if (n1 > GEMV_TRACE_CAPACITY) n1 = GEMV_TRACE_CAPACITY;

    printf("GEMV_TRACE_BEGIN,cluster=%u,core0=%u,core1=%u,dropped0=%u,dropped1=%u\n",
           (unsigned)cid, (unsigned)n0, (unsigned)n1,
           (unsigned)t.dropped[0], (unsigned)t.dropped[1]);
    printf("GEMV_TRACE,cycle,core,event,gen,slot,info\n");

    uint32_t i0 = 0, i1 = 0;
    while (i0 < n0 || i1 < n1) {
        const bool choose0 = (i1 == n1) ||
            (i0 < n0 && (int32_t)(t.data[0][i0].cycle -
                                       t.data[1][i1].cycle) <= 0);
        const uint32_t worker = choose0 ? 0 : 1;
        const Record& r = choose0 ? t.data[0][i0++] : t.data[1][i1++];
        printf("GEMV_TRACE,%u,%u,%s,%u,%u,%u\n",
               (unsigned)r.cycle, (unsigned)worker, event_name(r.event),
               (unsigned)r.gen, (unsigned)r.slot, (unsigned)r.info);
    }
    printf("GEMV_TRACE_END\n");
}
#else
template<class C>
inline void log(uint32_t, uint32_t, Event, uint32_t, size_t,
                uint32_t = 0) {}
template<class C>
inline void dump(uint32_t, uint32_t) {}
#endif

} // namespace gemv_detail

template<class C>
void mm_gemv_threaded(typename C::data_type* A,
                      typename C::data_type* B,
                      typename C::data_type* C_out,
                      size_t M, size_t N, size_t K) {
    using namespace gemv_detail;
    using T = typename C::data_type;
    static_assert(C::LOAD_DEPTH >= 2 && C::OUTPUT_DEPTH >= 1,
                  "GEMV requires >=2 input and >=1 output buffers");

    const uint32_t cid = flex_get_cluster_id();
    const uint32_t core = flex_get_core_id();
    const size_t mt = M / C::M;
    const size_t kt = K / C::K;

    if (N != 1 || !mt || !kt || M % C::M || K % C::K) {
        if (cid == 0 && core == 0) flex_print("GEMV: invalid dimensions\n");
        return;
    }

    size_t first_tile = 0, total = mt;
#if GEMV_PARTITION_ROWS
    constexpr size_t P = C::GRID_ROWS * C::GRID_ROWS;
    const size_t each = mt / P, extra = mt % P;
    first_tile = cid * each + (cid < extra ? cid : extra);
    total = each + (cid < extra);
#endif

    auto& p = pipeline<C>;
    auto& s = p.sync;
    auto& w = shared<C>;

    if (core == 0) {
        s = PipelineSync<C>{};
        w = Shared{};
#if GEMV_TRACE
        if (cid == GEMV_TRACE_CLUSTER) trace<C> = Trace<C>{};
#endif
    }
    flex_global_barrier_polling();

    if (core == 0) {
        // Compute worker. One worker owns input-consumed / output-ready counters.
        uint32_t gen = 0;
        size_t in = 0;
        for (size_t tile = 0; tile < total; ++tile) {
            const size_t out = tile % C::OUTPUT_DEPTH;
            if (tile >= C::OUTPUT_DEPTH) {
                const uint32_t need = (uint32_t)(tile - C::OUTPUT_DEPTH + 1);
                if (__atomic_load_n(&w.retired, __ATOMIC_ACQUIRE) < need)
                    log<C>(cid, core, WAIT_OUT, tile+1, out, need);
                while (__atomic_load_n(&w.retired, __ATOMIC_ACQUIRE) < need)
                    spin();
            }
            for (size_t kb = 0; kb < kt; ++kb) {
                ++gen;
                if (__atomic_load_n(&s.inputs_arrived[in].completed, __ATOMIC_ACQUIRE) < gen)
                    log<C>(cid, core, WAIT_IN, gen, in);
                while (__atomic_load_n(&s.inputs_arrived[in].completed, __ATOMIC_ACQUIRE) < gen)
                    spin();

                log<C>(cid, core, VPU_BEGIN, gen, in, (uint32_t)out);
                p.run(in, out, kb == 0);
                log<C>(cid, core, VPU_END, gen, in, (uint32_t)out);
                __atomic_store_n(&w.consumed, gen, __ATOMIC_RELEASE);
                log<C>(cid, core, INPUT_CONSUMED, gen, in);
                in = (in + 1) % C::LOAD_DEPTH;
            }
            fence();
            __atomic_store_n(&w.computed, (uint32_t)(tile+1), __ATOMIC_RELEASE);
            log<C>(cid, core, OUTPUT_READY, tile+1, out);
        }
    } else if (core == 1) {
        // DMA worker: asynchronous A/X pair; asynchronous Y between pairs.
        uint32_t gen = 0;
        size_t tile = 0, kb = 0, in = 0, next_store = 0, next_done = 0;
        bool ap = false, xp = false, ad = false, xd = false;
        bool finished = (total == 0), waiting = false;

        while (!finished || next_done < total) {
            // Retire the oldest outstanding output DMA if it has completed.
            if (next_done < next_store) {
                const size_t out = next_done % C::OUTPUT_DEPTH;
                if (bare_dma_done(s.output_dma_id[out])) {
                    fence();
                    ++next_done;
                    log<C>(cid, core, Y_DONE, next_done, out, s.output_dma_id[out]);
                    __atomic_store_n(&w.retired, (uint32_t)next_done,
                                     __ATOMIC_RELEASE);
                }
            }

            // A finished output may be stored between A/X input pairs.
            // No new input is issued until this output DMA has retired.
            if (next_store < total && next_store == next_done &&
                !ap && !xp &&
                __atomic_load_n(&w.computed, __ATOMIC_ACQUIRE) >= next_store+1) {
                const size_t out = next_store % C::OUTPUT_DEPTH;
                const size_t row = (first_tile + next_store) * C::M;
                fence();
                s.output_dma_id[out] = bare_dma_start_2d(
                    dma_addr(C_out + row), dma_addr(p.C_tiles[out].ptr()),
                    C::M*sizeof(T), C::M*sizeof(T), C::M*sizeof(T), 1);
                log<C>(cid, core, Y_STORE, next_store+1, out, s.output_dma_id[out]);
                ++next_store;
            }
            if (finished) continue;
            if (next_done != next_store) continue; // Y outstanding.

            const size_t old = __atomic_load_n(
                &s.inputs_arrived[in].completed, __ATOMIC_ACQUIRE);
            const bool reusable = !old ||
                __atomic_load_n(&w.consumed, __ATOMIC_ACQUIRE) >= old;
            if (!reusable && !waiting) {
                log<C>(cid, core, WAIT_SLOT, gen+1, in, (uint32_t)old);
                waiting = true;
            }

            const size_t row = (first_tile + tile)*C::M;
            const size_t k = kb*C::K;
            if (reusable && !ap) {
                s.a_dma_id[in] = bare_dma_start_2d(
                    dma_addr(p.A_tiles[in].ptr()), dma_addr(A + k*M + row),
                    C::M*sizeof(T), C::M*sizeof(T), M*sizeof(T), C::K);
                ap = true;
                log<C>(cid, core, A_LOAD, gen+1, in, s.a_dma_id[in]);
            }
            if (reusable && !xp) {
                // No wait between A and X submissions: this is input DMA overlap.
                s.b_dma_id[in] = bare_dma_start_2d(
                    dma_addr(p.B_tiles[in].ptr()), dma_addr(B + k),
                    C::K*sizeof(T), C::K*sizeof(T), C::K*sizeof(T), 1);
                xp = true;
                log<C>(cid, core, X_LOAD, gen+1, in, s.b_dma_id[in]);
            }
            if (ap && !ad && bare_dma_done(s.a_dma_id[in])) {
                ad = true;
                log<C>(cid, core, A_DONE, gen+1, in, s.a_dma_id[in]);
            }
            if (xp && !xd && bare_dma_done(s.b_dma_id[in])) {
                xd = true;
                log<C>(cid, core, X_DONE, gen+1, in, s.b_dma_id[in]);
            }
            if (!ad || !xd) continue;

            fence();
            ++gen;
            __atomic_store_n(&s.inputs_arrived[in].completed,
                             (size_t)gen, __ATOMIC_RELEASE);
            log<C>(cid, core, INPUT_READY, gen, in);
            in = (in+1) % C::LOAD_DEPTH;
            ap = xp = ad = xd = waiting = false;
            if (++kb == kt) {
                kb = 0;
                if (++tile == total) finished = true;
            }
        }
    }

    flex_global_barrier_polling(); // All input and output work is complete.
#if GEMV_TRACE && GEMV_TRACE_AUTO_DUMP
    dump<C>(cid, core);           // Only one core on one cluster prints.
    flex_global_barrier_polling(); // Avoid benchmark prints interleaving trace.
#endif
}

// Retained for callers that already invoke it. With auto-dump enabled,
// the 'printed' guard avoids dumping the same run twice.
template<class C>
void mm_gemv_threaded_dump_trace() {
    gemv_detail::dump<C>(flex_get_cluster_id(), flex_get_core_id());
}
