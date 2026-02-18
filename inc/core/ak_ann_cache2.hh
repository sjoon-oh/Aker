#pragma once

#include <cstddef>
#include <cstdint>

#include <memory>
#include <string>
#include <vector>

#include "ak_approx_filter.hh"
#include "core/ak_ann_cache_stats.hh"
#include "core/ak_ann_cache2_types.hh"

namespace aker
{
    // Forward declarations of internal modules.
    class ANNCache2EntryStore;
    class ANNCache2SimilarityEngine;
    class ANNCache2Maintenance;
    class ANNCache2Telemetry;

    // Forward declaration for the internal shared context.
    struct ANNCache2Context;

    /**
     * @brief Geometry-aware result cache sitting on top of an ANN index / external DB.
     *
     * This class exposes a stable public API while delegating implementation details
     * to internal modules (EntryStore / SimilarityEngine / Maintenance / Telemetry).
     *
     * Lock policy: public API acquires the global cache lock and invokes *Locked()
     * module methods. This refactor keeps the lock policy unchanged.
     */
    class ANNCache2
    {
    public:
        /**
         * @brief Constructs a cache instance with the provided parameter set.
         */
        explicit ANNCache2(ann_cache_config_t& parameter_info) noexcept;

        /**
         * @brief Destructor. Exports a final telemetry snapshot if the cache was used.
         */
        virtual ~ANNCache2();

        ANNCache2(const ANNCache2&) = delete;
        ANNCache2& operator=(const ANNCache2&) = delete;

        /**
         * @brief Creates a prepared cache entry before insertion.
         */
        result_cache_entry_t* makeCEntry(
            VectorSlot* query_vector,
            std::uint32_t list_size,
            VectorSlot** vector_local_reference_list) noexcept;

        /**
         * @brief Frees an externally-held cache entry.
         */
        void freeCEntry(result_cache_entry_t* entry) noexcept;

        /**
         * @brief Checks exact hit first, and then performs approximate lookup.
         */
        result_cache_entry_t* simGetCEntry(
            vector_view_t query_vector_data,
            bool& similar_entry,
            bool& is_invalid,
            distance_function_t distance_function) noexcept;

        /**
         * @brief Inserts a prepared entry into the cache.
         */
        bool insertCEntry2(
            vector_id_t vector_id,
            result_cache_entry_t* entry,
            vector_view_t query_vector_data) noexcept;

        /**
         * @brief Links an allocated dummy entry to an existing root entry.
         */
        bool linkCEntry(result_cache_entry_t* allocated_entry, vector_id_t found_id) noexcept;

        /**
         * @brief Marks a vector as deleted in the vector pool.
         */
        void markVectorDeleted(vector_id_t vector_id) noexcept;

        /**
         * @brief Inserts a write-log entry to reflect an underlying DB mutation.
         */
        void insertWLEntry3(
            vector_view_t write_vector,
            distance_function_t distance_function,
            result_conversion_function_t result_conversion_function = nullptr) noexcept;

        /**
         * @brief Consumes aged write-log entries and performs maintenance work.
         */
        void consumeAgedWLEntry(
            distance_function_t distance_function,
            result_conversion_function_t result_conversion_function = nullptr) noexcept;

        /**
         * @brief Clears cache state.
         */
        void resetCache() noexcept;

        /**
         * @brief Stress test helper that invalidates a random portion of pooled vectors.
         */
        void stressTestInvalidateRandom(float percent) noexcept;

        /**
         * @brief Collects pooled vectors into a provided list.
         */
        void collectPooledVectors(std::vector<VectorSlot*>& pooled_list) noexcept;

        /**
         * @brief Returns a human-readable status string.
         */
        std::string getStatusText() noexcept;

        /**
         * @brief Returns a concise CSV summary (key,value).
         */
        std::string getSummaryCsv() noexcept;

        /**
         * @brief Exports telemetry traces under /tmp/aker_trace_<timestamp>/.
         */
        void exportTraceToFiles() noexcept;

    private:
        std::unique_ptr<ANNCache2Context> context_;

        std::unique_ptr<ANNCache2EntryStore> entry_store_;
        std::unique_ptr<ANNCache2SimilarityEngine> similarity_engine_;
        std::unique_ptr<ANNCache2Maintenance> maintenance_;
        std::unique_ptr<ANNCache2Telemetry> telemetry_;
    };
}
