#pragma once

#include <cstddef>
#include <cstdint>

#include <boost/unordered/concurrent_flat_map.hpp>

#include "ak_ann_cache_config.hh"
#include "ak_vector_slot.hh"
#include "ak_write_log.hh"

namespace aker
{
    /**
     * @brief Cache entry ownership kind.
     *
     * - PREPARED: Entry created by makeCEntry() before insertion.
     * - INTERNAL: Entry stored in the cache (vector slots are pool-managed).
     * - RETURNED_COPY: Deep-copied entry returned to callers (owns slot VectorSlot objects).
     */
    typedef std::uint8_t result_cache_entry_kind_t;

    enum
    {
        RESULT_CACHE_ENTRY_KIND_PREPARED      = 0,
        RESULT_CACHE_ENTRY_KIND_INTERNAL      = 1,
        RESULT_CACHE_ENTRY_KIND_RETURNED_COPY = 2
    };

    /**
     * @brief Cache entry status for CAS-based entry-level locking.
     */
    typedef std::uint8_t result_cache_entry_status_t;

    enum
    {
        RESULT_CACHE_ENTRY_STATUS_VALID = 0,
        RESULT_CACHE_ENTRY_STATUS_INMOD
    };

    /**
     * @brief Result cache entry.
     *
     * This is a legacy POD style structure that is used across C/C++ boundaries.
     * Refactoring should preserve the layout and semantics of this structure.
     */
    typedef struct ResultCacheLookupEntry2
    {
        result_cache_entry_status_t     entry_status;
        result_cache_entry_kind_t       entry_kind;
        std::int32_t                    version;

        VectorSlot*                        query_vector;
        std::uint32_t                   vector_list_size;
        VectorSlot**                       vector_slot_ref_list;

        float                           thresh;
        float                           min_distance;
        float                           max_distance;

        struct ResultCacheLookupEntry2* prev;
        struct ResultCacheLookupEntry2* next;

        float                           risk_factor;
        write_log_checkpoint_t      checkpoint;

    } result_cache_entry_t;

    /**
     * @brief Lookup table for cache entries.
     */
    typedef struct ResultCacheLookupTable2
    {
        boost::concurrent_flat_map<vector_id_t, result_cache_entry_t*> map;

    } result_cache_table2_t;

    /**
     * @brief Alignment for pointer-array allocations used in cache entries.
     */
    inline constexpr std::size_t k_cache_entry_slot_alignment = 8;
}
