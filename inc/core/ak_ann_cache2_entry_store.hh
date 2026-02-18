#pragma once

#include <cstddef>

#include "core/ak_ann_cache2_context.hh"

namespace aker
{
    /**
     * @brief EntryStore module for ANNCache2.
     *
     * This module centralizes lookup table access and per-entry CAS locks.
     */
    class ANNCache2EntryStore
    {
    public:
        /**
         * @brief Constructs the module with the shared cache context.
         */
        explicit ANNCache2EntryStore(ANNCache2Context* context) noexcept;

        /**
         * @brief Resolves the given vector id to the root cache entry.
         */
        result_cache_entry_t* getCEntry(vector_id_t vector_id) noexcept;

        /**
         * @brief Creates a deep copy of a cache entry for external consumption.
         */
        result_cache_entry_t* copyCacheEntry(result_cache_entry_t* entry) noexcept;

        /**
         * @brief Attempts to acquire the per-entry CAS lock.
         */
        bool tryLockCEntry(result_cache_entry_t* entry) noexcept;

        /**
         * @brief Releases the per-entry CAS lock.
         */
        bool unlockCEntry(result_cache_entry_t* entry) noexcept;

        /**
         * @brief Links a dummy entry to an existing root entry.
         */
        bool linkCEntryLocked(result_cache_entry_t* allocated_entry, vector_id_t found_id) noexcept;

    private:
        ANNCache2Context* context_;
    };
}
