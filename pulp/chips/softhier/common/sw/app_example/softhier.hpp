#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <cstring>

#include "archi_redmule.h"
#include "hal_redmule.h"

#include "softhier_runtime.h"
#include "softhier_printf.h"
#include "softhier_dma_pattern.h"

#define ARCH_HBM_NUM_CHANNELS     16U


static inline uint64_t dma_addr(const void* ptr) {
    return static_cast<uint64_t>(
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(ptr))
    );
}

#include "pipeline.hpp"

static inline uint32_t random_next(uint32_t& state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

template<typename T>
void hbm_random_init(T* ptr, size_t count, uint32_t seed)
{
    uint32_t state = seed ? seed : 1u;
    volatile T* dst = ptr;

    const bool master =
        (flex_get_cluster_id() == 0 && flex_is_first_core());

    constexpr size_t LOG_INTERVAL = 65536;

    flex_global_barrier_polling();


    for (size_t i = 0; i < count; ++i) {
        const uint32_t r = random_next(state);

        const uint16_t bits = static_cast<uint16_t>(
            ((r & 1u) << 15) |
            0x3C00u |
            ((r >> 1) & 0x03FFu)
        );

        dst[i] = static_cast<T>(bits);

        if (master && ((i + 1) % LOG_INTERVAL == 0)) {
            flex_print("HBM initialized: ");
            flex_print_int(static_cast<uint32_t>(i + 1));
            flex_print(" / ");
            flex_print_int(static_cast<uint32_t>(count));
            flex_print("\n");
        }
    }

    if (master) {
        flex_print("HBM initialization complete: ");
        flex_print_int(static_cast<uint32_t>(count));
        flex_print(" elements\n");
    }
}

template<typename T>
T* hbm_alloc(
    uint32_t channel,
    size_t count,
    size_t& offset
) {
    constexpr uintptr_t HBM_BASE = ARCH_HBM_START_BASE;
    constexpr size_t CHANNEL_SIZE = ARCH_HBM_NODE_ADDR_SPACE;

    constexpr size_t alignment = alignof(T);

    // Align the next allocation.
    offset = (offset + alignment - 1) & ~(alignment - 1);

    const size_t bytes = count * sizeof(T);

    // Check available channel capacity.
    if (bytes > CHANNEL_SIZE || offset > CHANNEL_SIZE - bytes) {
        return nullptr;
    }

    const uintptr_t address = HBM_BASE + static_cast<uintptr_t>(channel) * CHANNEL_SIZE + offset;

    offset += bytes;

    return reinterpret_cast<T*>(address);
}

template<
    uint16_t Mb,
    uint16_t Nb,
    uint16_t Kb,
    uint16_t LoadDepth,
    uint16_t GridRows,
    uint16_t GridCols,
    bool OverlapStore = false,
    uint8_t FORMAT = Float16
>
struct config {
    static_assert(Mb > 0 && Mb % 16 == 0);
    static_assert(Nb > 0 && Nb % 16 == 0);
    static_assert(Kb > 0);

    static_assert(FORMAT == Float8 || FORMAT == Float16 || FORMAT == Float8Alt || FORMAT == Float16Alt);

    static_assert(LoadDepth >= 2, "Ring rotation requires two input slots");

    using data_type = std::conditional_t< FORMAT == Float8 || FORMAT == Float8Alt, uint8_t, uint16_t>;

    static constexpr size_t M = Mb;
    static constexpr size_t N = Nb;
    static constexpr size_t K = Kb;

    static constexpr size_t GRID_ROWS = GridRows;
    static constexpr size_t GRID_COLS = GridCols;

    static constexpr size_t LOAD_DEPTH = LoadDepth;
    static constexpr bool OVERLAP_STORE = OverlapStore;
    static constexpr size_t OUTPUT_DEPTH = OverlapStore ? 2 : 1;

    static constexpr uint8_t format = FORMAT;

    static constexpr size_t TCDM_BYTES =
        (LOAD_DEPTH * (M * K + K * N) +
         OUTPUT_DEPTH * M * N) * sizeof(data_type);

    static_assert(TCDM_BYTES <= 1048576, "Tiles must fit in TCDM");
};

template<typename C>
void mm_summa(typename C::data_type* A, typename C::data_type* B, typename C::data_type* C_out, size_t M, size_t N, size_t K);

#include "mm_summa.tpp"