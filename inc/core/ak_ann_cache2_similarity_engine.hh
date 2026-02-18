#pragma once

#include "ak_approx_filter.hh"
#include "core/ak_ann_cache2_context.hh"

namespace aker
{
    class ANNCache2EntryStore;

    /**
     * @brief SimilarityEngine module for ANNCache2.
     *
     * This module implements exact-hit and sim-hit logic and invalid handling.
     */
    class ANNCache2SimilarityEngine
    {
    public:
        /**
         * @brief Constructs the module with the shared cache context.
         */
        ANNCache2SimilarityEngine(ANNCache2Context* context, ANNCache2EntryStore* entry_store) noexcept;

        /**
         * @brief Looks up a cache entry by exact id or by approximate similarity.
         */
        result_cache_entry_t* simGetCEntryLocked(
            vector_view_t query_vector_data,
            bool& similar_entry,
            bool& is_invalid,
            distance_function_t distance_function) noexcept;

        /**
         * @brief Validates an entry by pushing invalid vectors to the tail.
         */
        bool handleInvalidCEntryLocked(result_cache_entry_t* entry) noexcept;

    private:
        ANNCache2Context*  context_;
        ANNCache2EntryStore* entry_store_;
    };
}
