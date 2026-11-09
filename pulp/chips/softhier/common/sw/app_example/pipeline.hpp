#define W_COLS_OFFSET 0

template<size_t WIDTH, size_t LENGTH, typename T, bool row_major = true>
struct tile {
    static constexpr size_t BANK_WIDTH = 4;
    static constexpr size_t BANK_COUNT = 128;

    static constexpr size_t width = WIDTH;
    static constexpr size_t length = LENGTH;
    static constexpr size_t num_elements = LENGTH * WIDTH;

    alignas(BANK_WIDTH) T data
        [row_major ? LENGTH : WIDTH]
        [row_major ? WIDTH : LENGTH];

    T* ptr() { return &data[0][0]; }
    const T* ptr() const { return &data[0][0]; }

    static constexpr size_t offset(size_t row, size_t col) {
        return row_major ? row * WIDTH + col
                         : col * LENGTH + row;
    }

    T& operator()(size_t row, size_t col) {
        return ptr()[offset(row, col)];
    }

    const T& operator()(size_t row, size_t col) const {
        return ptr()[offset(row, col)];
    }
};

struct Semaphore {
    size_t completed = 0;

    void signal(size_t generation) {
        completed = generation;
    }

    bool ready(size_t generation) const {
        return completed >= generation;
    }
};

// Describes one output tile and its entire K reduction.
// A, B, C point to the beginning of the relevant submatrices.
template<typename T>
struct TileTask {
    const T* A;
    const T* B;
    T* C;

    size_t lda;
    size_t ldb;
    size_t ldc;

    size_t k_tiles;
};

extern volatile uint32_t redmule_completed_jobs;

template<typename C>
struct PipelineSync {
    Semaphore inputs_arrived[C::LOAD_DEPTH];

    uint32_t a_dma_id[C::LOAD_DEPTH] = {};
    uint32_t b_dma_id[C::LOAD_DEPTH] = {};
    uint32_t output_dma_id[C::OUTPUT_DEPTH] = {};
};

template<typename C>
struct ClusterPipeline {
    using T = typename C::data_type;

    tile<C::K, C::M, T> A_tiles[C::LOAD_DEPTH];
    tile<C::N, C::K, T> B_tiles[C::LOAD_DEPTH];
    tile<C::N, C::M, T> C_tiles[C::OUTPUT_DEPTH];

    PipelineSync<C> sync{};

    int run(size_t in_slot, size_t out_slot, bool first);

    void submit_input(const TileTask<T>& task, size_t k_tile, size_t slot);

    void wait_input(size_t slot, size_t generation);

    int submit_redmule(size_t in_slot, size_t out_slot, uint8_t op);

    void wait_redmule(uint32_t target);

    void submit_output(T* destination, size_t ldc, size_t slot);

    void wait_output(size_t slot);
};

template<typename C>
inline void ClusterPipeline<C>::submit_input(
    const TileTask<T>& task,
    size_t k_tile,
    size_t slot
) {
    const size_t k = k_tile * C::K;

    sync.a_dma_id[slot] = bare_dma_start_2d(
        dma_addr(A_tiles[slot].ptr()),
        dma_addr(task.A + k),
        C::K * sizeof(T),
        C::K * sizeof(T),
        task.lda * sizeof(T),
        C::M
    );

    // Last ID in the A/B pair, assuming ordered DMA completion.
    sync.b_dma_id[slot] = bare_dma_start_2d(
        dma_addr(B_tiles[slot].ptr()),
        dma_addr(task.B + k * task.ldb),
        C::N * sizeof(T),
        C::N * sizeof(T),
        task.ldb * sizeof(T),
        C::K
    );
}

template<typename C>
inline void ClusterPipeline<C>::wait_input(
    size_t slot,
    size_t generation
) {
    bare_dma_wait(sync.a_dma_id[slot]);
    bare_dma_wait(sync.b_dma_id[slot]);
    sync.inputs_arrived[slot].signal(generation);
}

template<typename C>
inline int ClusterPipeline<C>::submit_redmule(
    size_t in_slot,
    size_t out_slot,
    uint8_t op
) {

    int job_id = hwpe_acquire_job();

    while (job_id < 0)
        job_id = hwpe_acquire_job();

    redmule_cfg(
        reinterpret_cast<uintptr_t>(A_tiles[in_slot].ptr()),
        reinterpret_cast<uintptr_t>(B_tiles[in_slot].ptr()),
        reinterpret_cast<uintptr_t>(C_tiles[out_slot].ptr()),
        C::M, C::K, C::N,
        static_cast<uint16_t>(W_COLS_OFFSET),
        op,
        C::format
    );

    asm volatile("fence iorw, iorw" ::: "memory");

    hwpe_trigger_job();

    return job_id;
}

template<typename C>
inline void ClusterPipeline<C>::wait_redmule(uint32_t target) {
    while (redmule_completed_jobs < target)
        asm volatile("" ::: "memory");
}

template<typename C>
inline void ClusterPipeline<C>::submit_output(
    T* destination,
    size_t ldc,
    size_t slot
) {
    sync.output_dma_id[slot] = bare_dma_start_2d(
        dma_addr(destination),
        dma_addr(C_tiles[slot].ptr()),
        C::N * sizeof(T),
        ldc * sizeof(T),
        C::N * sizeof(T),
        C::M
    );
}

template<typename C>
inline void ClusterPipeline<C>::wait_output(
    size_t slot
) {
    bare_dma_wait(sync.output_dma_id[slot]);
}