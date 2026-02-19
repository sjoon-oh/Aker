#pragma once

#include <cstddef>
#include <cstdint>

#include <memory>

#include "ak_approx_filter.hh"
#include "ak_eviction_strategy.hh"
#include "ak_vector_slot_pool.hh"
#include "ak_write_log.hh"

#include "core/ak_anns_cache_stats.hh"
#include "core/ak_anns_cache_types.hh"
#include "utils/ak_spin_mutex.hh"

namespace aker
{
    /**
     * @brief Internal shared context for ANNSCache modules.
     *
     * This structure owns the cache's core state and is shared across internal modules.
     * It exists to keep ANNSCache as a thin facade while allowing modules to access
     * the same state without friend relationships.
     */
    struct ANNSCacheContext
    {
        /**
         * @brief Cache configuration parameters.
         */
        anns_cache_config_t parameter;

        /**
         * @brief Telemetry statistics.
         */
        anns_cache_stats_t stats;

        /**
         * @brief Current representative entry count.
         */
        size_t repr_entry_count;

        /**
         * @brief Query-id to cache-entry lookup table.
         */
        std::unique_ptr<anns_cache_table_t> lookup_table;

        /**
         * @brief Vector pool storing shared result vectors.
         */
        std::unique_ptr<VectorSlotPool> vector_pool;

        /**
         * @brief Eviction strategy used to select candidates.
         */
        std::unique_ptr<EvictionStrategy> eviction_strategy;

        /**
         * @brief Approximate filter storing representative query vectors.
         */
        std::unique_ptr<ApproxFilterDualHNSW2> apprx_filter;

        /**
         * @brief Count of evicted entries.
         */
        std::uint32_t evict_entry_count;

        /**
         * @brief Risk-aware write log.
         */
        std::unique_ptr<RiskAwareWriteLog> write_log;

        /**
         * @brief Approximate read count used for deciding write-log maintenance.
         */
        std::uint32_t try_read_count;

        /**
         * @brief Global cache lock.
         */
        SpinMutex cache_lock;

        /**
         * @brief Tracks whether the cache observed any activity.
         */
        bool has_activity;

        /**
         * @brief Constructs a context and initializes owned components.
         */
        explicit ANNSCacheContext(const anns_cache_config_t& parameter_info) noexcept;

        /**
         * @brief Destructor.
         */
        ~ANNSCacheContext() noexcept;

        ANNSCacheContext(const ANNSCacheContext&) = delete;
        ANNSCacheContext& operator=(const ANNSCacheContext&) = delete;
    };
}
