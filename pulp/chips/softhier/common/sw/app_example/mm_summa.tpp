
#include "gemm.hpp"

#define DEBUG_SUMMA

#ifdef DEBUG_SUMMA
#define SUMMA_LOG(...) printf(__VA_ARGS__)

enum : uint16_t {
    COMPUTE_SUBMIT,
    OUTPUT_SUBMIT,
    OUTPUT_DONE,
    OUTPUT_REUSE_WAIT,
    OUTPUT_REUSE_DONE,
    INPUT_A_SUBMIT,
    INPUT_B_SUBMIT,
    INPUT_A_DONE,
    INPUT_B_DONE,
    SHIFT_A_SUBMIT,
    SHIFT_B_SUBMIT,
    SHIFT_A_DONE,
    SHIFT_B_DONE,
};

struct SummaEvent {
    uint32_t cycle;
    uint16_t type;
    uint16_t slot;
    uint32_t generation;
};

static constexpr size_t MAX_EVENTS = 2048;
static SummaEvent events[MAX_EVENTS] __attribute__((section(".l1")));
static size_t event_count = 0;

static inline uint32_t summa_cycles() {
    uint32_t cycles;
    asm volatile("csrr %0, mcycle" : "=r"(cycles));
    return cycles;
}

static inline void summa_event(uint16_t type, uint16_t slot, uint32_t gen) {
    if (event_count < MAX_EVENTS)
        events[event_count++] = {summa_cycles(), type, slot, gen};
}

#define SUMMA_EVENT(type, slot, gen) \
    do { \
        if (flex_get_cluster_id() == 0 && \
            flex_is_first_core()) { \
            summa_event(type, slot, gen); \
        } \
    } while (0)

#else
#define SUMMA_EVENT(...) ((void)0)
#define SUMMA_LOG(...) ((void)0)
#endif

template<typename C>
struct RingSync {
    volatile uint32_t ready_a[C::LOAD_DEPTH] = {};
    volatile uint32_t ready_b[C::LOAD_DEPTH] = {};
    volatile uint32_t pulled_a[C::LOAD_DEPTH] = {};
    volatile uint32_t pulled_b[C::LOAD_DEPTH] = {};
};

template<typename C>
static ClusterPipeline<C> gemm_storage __attribute__((section(".l1")));

template<typename C>
static RingSync<C> ring_sync __attribute__((section(".l1")));

template<typename C>
void mm_summa(
    typename C::data_type* A,
    typename C::data_type* B,
    typename C::data_type* C_out,
    size_t M, size_t N, size_t K
) {
    using T = typename C::data_type;

    const uint32_t cid = flex_get_cluster_id();
    const bool master = (cid == 0 && flex_is_first_core());

    if (M % C::M || N % C::N || K % C::K) {
        if (master)
            flex_print("Matrix dimension not a multiple of tile dimensions, exiting\n");
        return;
    }

    const size_t m_tiles = M / C::M;
    const size_t n_tiles = N / C::N;
    const size_t k_tiles = K / C::K;

    if (!m_tiles || !n_tiles || !k_tiles) {
        if (master)
            flex_print("Matrix size smaller than tile size, exiting\n");
        return;
    }

    const size_t m_groups = (m_tiles + C::GRID_ROWS - 1) / C::GRID_ROWS;
    const size_t n_groups = (n_tiles + C::GRID_ROWS - 1) / C::GRID_ROWS;
    const size_t k_rounds = (k_tiles + C::GRID_ROWS - 1) / C::GRID_ROWS;

    if (m_tiles % C::GRID_ROWS || n_tiles % C::GRID_ROWS) {
        if (master)
            flex_print("Number of tiles cannot be equally distributed across cluster grid, exiting\n");
        return;
    }

    const FlexPosition2D pos = get_pos(cid);
    const FlexPosition2D left = left_pos(pos);
    const FlexPosition2D bottom = bottom_pos(pos);

    const size_t aligned = (pos.y + C::GRID_ROWS - pos.x) % C::GRID_ROWS;

    auto& gemm = gemm_storage<C>;
    auto& sync = gemm.sync;
    auto& net = ring_sync<C>;

    net = RingSync<C>{};
    sync = PipelineSync<C>{};

    auto offset = [&](const void* ptr) -> uint32_t {
        return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(ptr) - static_cast<uintptr_t>(local(0)));
    };

    auto remote_word = [&](FlexPosition2D p, size_t field, size_t slot) -> volatile uint32_t* {
        return reinterpret_cast<volatile uint32_t*>(remote_pos(p, offset(&net) + field + slot * sizeof(uint32_t)));
    };

    constexpr size_t READY_A = offsetof(RingSync<C>, ready_a);
    constexpr size_t READY_B = offsetof(RingSync<C>, ready_b);
    constexpr size_t ACK_A = offsetof(RingSync<C>, pulled_a);
    constexpr size_t ACK_B = offsetof(RingSync<C>, pulled_b);

    flex_global_barrier_polling();

    const uint32_t base = redmule_completed_jobs;
    uint32_t generation = 0;
    size_t output_generation = 0;

    enum class OutputPhase : uint8_t {
        Free,
        Computing,
        Storing
    };

    struct OutputWork {
        T* destination = nullptr;
        uint32_t target = 0;
        OutputPhase phase = OutputPhase::Free;
    };

    OutputWork outputs[C::OUTPUT_DEPTH] = {};

    auto progress_outputs = [&]() {
        const uint32_t done = redmule_completed_jobs;

        for (size_t s = 0; s < C::OUTPUT_DEPTH; ++s) {
            auto& w = outputs[s];

            if (w.phase == OutputPhase::Computing && done >= w.target) {
                gemm.submit_output(w.destination, N, s);
                SUMMA_EVENT(OUTPUT_SUBMIT, s, w.target);
                w.phase = OutputPhase::Storing;
            }
        }
    };

    auto acquire_output = [&](size_t slot) {
        auto& w = outputs[slot];
        progress_outputs();

        while (w.phase == OutputPhase::Computing) {
            progress_outputs();
            asm volatile("" ::: "memory");
        }

        if (w.phase == OutputPhase::Storing) {
            SUMMA_EVENT(OUTPUT_REUSE_WAIT, slot, w.target);
            gemm.wait_output(slot);
            SUMMA_EVENT(OUTPUT_REUSE_DONE, slot, w.target);
            SUMMA_EVENT(OUTPUT_DONE, slot, w.target);
            w.phase = OutputPhase::Free;
        }
    };

    uint32_t incoming_gen[C::LOAD_DEPTH] = {};
    size_t incoming_src[C::LOAD_DEPTH] = {};
    bool incoming_shift[C::LOAD_DEPTH] = {};

    bool a_posted[C::LOAD_DEPTH] = {};
    bool b_posted[C::LOAD_DEPTH] = {};

    bool a_done[C::LOAD_DEPTH] = {};
    bool b_done[C::LOAD_DEPTH] = {};

    auto can_reuse_a = [&](size_t slot) -> bool {
        const uint32_t old =sync.inputs_arrived[slot].completed;
        return !old || (redmule_completed_jobs >= base + old && (old % C::GRID_ROWS == 0 || net.pulled_a[slot] >= old));
    };

    auto can_reuse_b = [&](size_t slot) -> bool {
        const uint32_t old = sync.inputs_arrived[slot].completed;
        return !old || (redmule_completed_jobs >= base + old && (old % C::GRID_ROWS == 0 || net.pulled_b[slot] >= old));
    };

    auto finish_a = [&](size_t slot) {
        if (a_done[slot])
            return;

        bare_dma_wait(sync.a_dma_id[slot]);
        asm volatile("fence rw, rw" ::: "memory");
        const uint32_t gen = incoming_gen[slot];

        if (incoming_shift[slot]) {
            *remote_word(left, ACK_A, incoming_src[slot]) = gen - 1;
            SUMMA_EVENT(SHIFT_A_DONE, slot, gen);
        } else {
            SUMMA_EVENT(INPUT_A_DONE, slot, gen);
        }

        asm volatile("fence rw, rw" ::: "memory");
        net.ready_a[slot] = gen;
        a_done[slot] = true;
    };

    auto finish_b = [&](size_t slot) {
        if (b_done[slot])
            return;

        bare_dma_wait(sync.b_dma_id[slot]);
        asm volatile("fence rw, rw" ::: "memory");
        const uint32_t gen = incoming_gen[slot];

        if (incoming_shift[slot]) {
            *remote_word(bottom, ACK_B, incoming_src[slot]) = gen - 1;
            SUMMA_EVENT(SHIFT_B_DONE, slot, gen);
        } else {
            SUMMA_EVENT(INPUT_B_DONE, slot, gen);
        }

        asm volatile("fence rw, rw" ::: "memory");
        net.ready_b[slot] = gen;
        b_done[slot] = true;
    };

    auto load_round = [&](
        const TileTask<T>& task,
        size_t kr,
        size_t slot,
        uint32_t gen
    ) {
        const size_t kt = kr * C::GRID_ROWS + aligned;
        const size_t k = kt * C::K;

        incoming_gen[slot] = gen;
        incoming_shift[slot] = false;

        a_posted[slot] = false;
        b_posted[slot] = false;
        a_done[slot] = false;
        b_done[slot] = false;

        while (!a_posted[slot] || !b_posted[slot]) {
            if (!a_posted[slot] && can_reuse_a(slot)) {
                if (kt < k_tiles) {
                    sync.a_dma_id[slot] =
                        bare_dma_start_2d(
                            dma_addr(gemm.A_tiles[slot].ptr()),
                            dma_addr(task.A + k),
                            C::K * sizeof(T),
                            C::K * sizeof(T),
                            task.lda * sizeof(T),
                            C::M);
                } else {
                    T* a = gemm.A_tiles[slot].ptr();
                    asm volatile("fence rw, rw" ::: "memory");
                    net.ready_a[slot] = gen;
                    a_done[slot] = true;
                }

                a_posted[slot] = true;
                SUMMA_EVENT(INPUT_A_SUBMIT, slot, gen);
            }

            if (!b_posted[slot] && can_reuse_b(slot)) {
                if (kt < k_tiles) {
                    sync.b_dma_id[slot] =
                        bare_dma_start_2d(
                            dma_addr(gemm.B_tiles[slot].ptr()),
                            dma_addr(task.B + k * task.ldb),
                            C::N * sizeof(T),
                            C::N * sizeof(T),
                            task.ldb * sizeof(T),
                            C::K
                        );
                } else {
                    T* b = gemm.B_tiles[slot].ptr();
                    asm volatile("fence rw, rw" ::: "memory");
                    net.ready_b[slot] = gen;
                    b_done[slot] = true;
                }

                b_posted[slot] = true;
                SUMMA_EVENT(INPUT_B_SUBMIT, slot, gen);
            }

            if (!a_posted[slot] && b_posted[slot] && !b_done[slot])
                finish_b(slot);

            if (!b_posted[slot] && a_posted[slot] && !a_done[slot])
                finish_a(slot);

            progress_outputs();
            asm volatile("" ::: "memory");
        }
    };

    auto receive = [&](size_t slot, uint32_t gen, bool shifted) {
        finish_a(slot);
        finish_b(slot);
        sync.inputs_arrived[slot].signal(gen);
    };

    auto shift = [&](
        size_t src,
        size_t dst,
        uint32_t gen
    ) {
        const uint32_t next_gen = gen + 1;

        incoming_gen[dst] = next_gen;
        incoming_src[dst] = src;
        incoming_shift[dst] = true;

        a_posted[dst] = false;
        b_posted[dst] = false;
        a_done[dst] = false;
        b_done[dst] = false;

        volatile uint32_t* left_ready = remote_word(left, READY_A, src);
        volatile uint32_t* bottom_ready = remote_word(bottom, READY_B, src);

        while (!a_posted[dst] || !b_posted[dst]) {
            if (!a_posted[dst] && can_reuse_a(dst) && *left_ready >= gen) {
                sync.a_dma_id[dst] =
                    flex_dma_async_pattern_round_shift_right(
                        offset(gemm.A_tiles[dst].ptr()),
                        offset(gemm.A_tiles[src].ptr()),
                        C::M * C::K * sizeof(T));

                a_posted[dst] = true;
                SUMMA_EVENT(SHIFT_A_SUBMIT, dst, next_gen);
            }

            if (!b_posted[dst] && can_reuse_b(dst) && *bottom_ready >= gen) {
                sync.b_dma_id[dst] =
                    flex_dma_async_pattern_round_shift_up(
                        offset(gemm.B_tiles[dst].ptr()),
                        offset(gemm.B_tiles[src].ptr()),
                        C::K * C::N * sizeof(T));

                b_posted[dst] = true;
                SUMMA_EVENT(SHIFT_B_SUBMIT,dst,next_gen);
            }

            if (!a_posted[dst] && b_posted[dst] && !b_done[dst])
                finish_b(dst);

            if (!b_posted[dst] && a_posted[dst] && !a_done[dst])
                finish_a(dst);

            progress_outputs();
            asm volatile("" ::: "memory");
        }
    };

    size_t slot = 0;

    for (size_t mg = 0; mg < m_groups; ++mg) {
        for (size_t ng = 0; ng < n_groups; ++ng) {
            const size_t m = mg * C::GRID_ROWS + pos.y;
            const size_t n = ng * C::GRID_ROWS + pos.x;
            const size_t out = output_generation % C::OUTPUT_DEPTH;

            TileTask<T> task = {
                A + m * C::M * K,
                B + n * C::N,
                C_out + m * C::M * N + n * C::N,
                K, N, N, k_tiles
            };

            const uint32_t tile_base = generation;

            progress_outputs();
            load_round(task, 0, slot, tile_base + 1);

            for (size_t kr = 0; kr < k_rounds; ++kr) {
                for (size_t hop = 0; hop < C::GRID_ROWS; ++hop) {
                    const uint32_t gen = ++generation;
                    const size_t next = (slot + 1) % C::LOAD_DEPTH;
                    receive(slot, gen, hop != 0);

                    if (kr == 0 && hop == 0)
                        acquire_output(out);

                    progress_outputs();
                    gemm.run(slot, out, kr == 0 && hop == 0);
                    SUMMA_EVENT(COMPUTE_SUBMIT, slot, gen);
                    progress_outputs();

                    if (hop + 1 < C::GRID_ROWS) {
                        shift(slot, next, gen);
                    } else if (kr + 1 < k_rounds) {
                        const uint32_t next_gen = tile_base + (kr + 1) * C::GRID_ROWS + 1;
                        load_round(task, kr + 1, next, next_gen);
                    }
                    slot = next;
                }
            }

            outputs[out] = {task.C, base + generation, OutputPhase::Computing};
            progress_outputs();
            ++output_generation;
        }
    }

    for (size_t s = 0; s < C::OUTPUT_DEPTH; ++s) {
        auto& w = outputs[s];

        while (w.phase == OutputPhase::Computing) {
            progress_outputs();
            asm volatile("" ::: "memory");
        }

        if (w.phase == OutputPhase::Storing) {
            gemm.wait_output(s);
            SUMMA_EVENT(OUTPUT_DONE, s, w.target);
            w.phase = OutputPhase::Free;
        }
    }

#ifdef DEBUG_SUMMA
    if (master) {
        for (size_t i = 0; i < event_count; ++i) {
            SUMMA_LOG(
                "[cluster %u] cycle=%u type=%u slot=%u gen=%u\n",
                (unsigned)cid,
                (unsigned)events[i].cycle,
                (unsigned)events[i].type,
                (unsigned)events[i].slot,
                (unsigned)events[i].generation
            );
        }
    }
#endif
}