#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <ctime>

#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

#include "ak_logger.hh"

#include "core/ak_cache_vector_copy.hh"
#include "core/ak_anns_cache.hh"
#include "core/ak_anns_cache_context.hh"
#include "core/ak_anns_cache_modules.hh"
#include "utils/ak_malloc_ptr.hh"

namespace aker
{
    namespace
    {
        static constexpr size_t k_write_log_scan_thresh = 64;
    }

    ANNSCacheContext::ANNSCacheContext(const anns_cache_config_t& parameter_info) noexcept
        : parameter(parameter_info),
          stats(),
          repr_entry_count(0),
          lookup_table(std::make_unique<anns_cache_table_t>()),
          vector_pool(std::make_unique<VectorSlotPool>(parameter_info.capacity.slot_pool_size, parameter_info.vector_format.vector_data_size)),
          eviction_strategy(std::make_unique<EvictionStrategyFifo>()),
          apprx_filter(std::make_unique<ApproxFilterDualHNSW2>(parameter_info)),
          evict_entry_count(0),
          write_log(std::make_unique<RiskAwareWriteLog>(parameter_info.capacity.vector_in_topk, k_write_log_scan_thresh, parameter_info.tuning.risk_thresh)),
          try_read_count(0),
          has_activity(false)
    {
    }

    ANNSCacheContext::~ANNSCacheContext() noexcept = default;

    ANNSCache::ANNSCache(anns_cache_config_t& parameter_info) noexcept
        : context_(std::make_unique<ANNSCacheContext>(parameter_info))
    {
        /* Initializes the cache with the provided parameters.
         */
        entry_store_ = std::make_unique<ANNSCacheEntryStore>(context_.get());
        similarity_engine_ = std::make_unique<ANNSCacheSimilarityEngine>(context_.get(), entry_store_.get());
        maintenance_ = std::make_unique<ANNSCacheMaintenance>(context_.get(), entry_store_.get());
        telemetry_ = std::make_unique<ANNSCacheTelemetry>(context_.get());

        /* Emit a configuration snapshot for debugging.
         */
        AKER_LOG_INFO << "[ANNSCache] parameters";
        AKER_LOG_INFO << "  vector_dim=" << context_->parameter.vector_format.vector_dim;
        AKER_LOG_INFO << "  slot_pool_size=" << context_->parameter.capacity.slot_pool_size;
        AKER_LOG_INFO << "  slot_list_size=" << context_->parameter.capacity.slot_list_size;
        AKER_LOG_INFO << "  vector_data_size=" << context_->parameter.vector_format.vector_data_size;
        AKER_LOG_INFO << "  vector_in_topk=" << context_->parameter.capacity.vector_in_topk;
        AKER_LOG_INFO << "  vector_extras=" << context_->parameter.capacity.vector_extras;
        AKER_LOG_INFO << "  similar_match=" << static_cast<int>(context_->parameter.tuning.similar_match);
        AKER_LOG_INFO << "  use_fixed_thresh=" << static_cast<int>(context_->parameter.tuning.use_fixed_thresh);
        AKER_LOG_INFO << "  fixed_thresh=" << context_->parameter.tuning.fixed_thresh;
        AKER_LOG_INFO << "  start_thresh=" << context_->parameter.tuning.start_thresh;
        AKER_LOG_INFO << "  risk_thresh=" << context_->parameter.tuning.risk_thresh;
        AKER_LOG_INFO << "  alpha_tighten=" << context_->parameter.tuning.alpha_tighten;
        AKER_LOG_INFO << "  alpha_loosen=" << context_->parameter.tuning.alpha_loosen;

        assert(context_->parameter.capacity.slot_list_size ==
               (context_->parameter.capacity.vector_in_topk + context_->parameter.capacity.vector_extras));
    }

    ANNSCache::~ANNSCache()
    {
        /* Export a final trace snapshot on destruction.
         */
        if (context_ == nullptr || !context_->has_activity)
        {
            return;
        }

        std::lock_guard<SpinMutex> cache_guard(context_->cache_lock);

        telemetry_->exportTraceToFiles();

        const std::string summary_csv = telemetry_->buildSummaryCsv();
        const std::string& trace_directory_path = context_->stats.getTraceDirectoryPath();

        if (!trace_directory_path.empty())
        {
            AKER_LOG_INFO << "[ANNSCache] exported telemetry to " << trace_directory_path;
        }

        AKER_LOG_INFO << "[ANNSCache] final state summary (CSV)";
        AKER_LOG_INFO << "\n" << summary_csv;
    }

    anns_cache_entry_t*
    ANNSCache::createCacheEntry(
        VectorSlot* query_vector,
        std::uint32_t list_size,
        VectorSlot** vector_local_reference_list) noexcept
    {
        /* Prepares a cache entry.
         * The returned entry is not inserted until insertCacheEntry() succeeds.
         */
        context_->has_activity = true;

        anns_cache_entry_t* entry = new anns_cache_entry_t();

        const size_t vec_data_size = context_->vector_pool->getPayloadSize();
        entry->query_vector = cloneVectorBasic(query_vector, vec_data_size);

        entry->entry_kind = ANNS_CACHE_ENTRY_KIND_PREPARED;
        entry->entry_status = ANNS_CACHE_ENTRY_STATUS_VALID;
        entry->version = 0;

        entry->vector_list_size = list_size;

        if (vector_local_reference_list != nullptr)
        {
            entry->vector_slot_ref_list = static_cast<VectorSlot**>(
                aligned_alloc(k_cache_entry_slot_alignment, sizeof(VectorSlot*) * list_size));
            std::memcpy(entry->vector_slot_ref_list, vector_local_reference_list, sizeof(VectorSlot*) * list_size);

            entry->min_distance = entry->vector_slot_ref_list[0]->getDistance();
            entry->max_distance = entry->vector_slot_ref_list[list_size - 1]->getDistance();
        }
        else
        {
            entry->vector_slot_ref_list = nullptr;
            entry->min_distance = std::numeric_limits<float>::max();
            entry->max_distance = std::numeric_limits<float>::min();
        }

        entry->thresh = entry->min_distance;
        entry->thresh *= context_->parameter.tuning.start_thresh;

        if (context_->parameter.tuning.similar_match == 0)
            entry->thresh = 0;

        entry->prev = nullptr;
        entry->next = nullptr;
        entry->checkpoint = nullptr;
        entry->risk_factor = 0.0f;

        return entry;
    }

    void
    ANNSCache::destroyCacheEntry(anns_cache_entry_t* entry) noexcept
    {
        /* Releases an externally held entry.
         * - Always deletes the owned query_vector.
         * - Frees the slot pointer array.
         * - Deletes slot VectorSlot objects only for RETURNED_COPY entries.
         */
        context_->has_activity = true;

        if (entry == nullptr)
            return;

        if (entry->entry_kind == ANNS_CACHE_ENTRY_KIND_RETURNED_COPY)
        {
            if (entry->vector_slot_ref_list != nullptr)
            {
                for (size_t i = 0; i < entry->vector_list_size; i++)
                    delete entry->vector_slot_ref_list[i];
            }
        }

        delete entry->query_vector;
        entry->query_vector = nullptr;

        free(entry->vector_slot_ref_list);
        entry->vector_slot_ref_list = nullptr;

        delete entry;
    }

    anns_cache_entry_t*
    ANNSCache::getCacheEntry(
        vector_view_t query_vector_data,
        bool& similar_entry,
        bool& is_invalid,
        distance_function_t distance_function) noexcept
    {
        /* Public API wrapper.
         * Measures end-to-end latency and records a concise history sample.
         */
        context_->has_activity = true;

        ElapsedLatencyPair latency;
        latency.start();

        anns_cache_entry_t* result_entry = nullptr;
        size_t cache_entry_count = 0;
        size_t pool_size = 0;
        size_t approx_repr = 0;

        /* Convert the query vector before entering the global cache critical section.
         * This reduces the lock hold time without changing cache semantics.
         */
        MallocPtr<float> query_vector_float = makeMallocPtr<float>(query_vector_data.vector_dim);
        bool convert_success = query_vector_data.conversion_function(
            query_vector_data.vector_data,
            query_vector_data.vector_data_size,
            query_vector_data.vector_dim,
            query_vector_float.get(),
            query_vector_data.aux);
        (void)convert_success;

        {
            std::lock_guard<SpinMutex> cache_guard(context_->cache_lock);

            result_entry = similarity_engine_->getCacheEntryLocked(
                query_vector_data, similar_entry, is_invalid, distance_function, query_vector_float.get());

            cache_entry_count = context_->eviction_strategy->getCurrSize();
            pool_size = context_->vector_pool->getSize();
            approx_repr = context_->apprx_filter->getRepresentativeVectorNumber();
        }

        latency.end();
        latency.elapsedMs();

        context_->stats.appendLatencySample(ANNSCacheStats::LatencyMetric::k_get_cache_entry, latency);

        context_->stats.recordCacheHistorySample("getCacheEntry", latency.getElapsedMs(), cache_entry_count, pool_size, approx_repr);

        return result_entry;
    }

    bool
    ANNSCache::insertCacheEntry(
        vector_id_t vector_id,
        anns_cache_entry_t* entry,
        vector_view_t query_vector_data) noexcept
    {
        context_->has_activity = true;

        ElapsedLatencyPair latency;
        latency.start();

        bool inserted = false;
        size_t cache_entry_count = 0;
        size_t pool_size = 0;
        size_t approx_repr = 0;

        {
            std::lock_guard<SpinMutex> cache_guard(context_->cache_lock);

            inserted = maintenance_->insertCacheEntryLocked(vector_id, entry, query_vector_data);

            cache_entry_count = context_->eviction_strategy->getCurrSize();
            pool_size = context_->vector_pool->getSize();
            approx_repr = context_->apprx_filter->getRepresentativeVectorNumber();
        }

        latency.end();
        latency.elapsedMs();

        context_->stats.appendLatencySample(ANNSCacheStats::LatencyMetric::k_insert_cache_entry, latency);

        context_->stats.recordCacheHistorySample("insertCacheEntry", latency.getElapsedMs(), cache_entry_count, pool_size, approx_repr);

        return inserted;
    }

    bool
    ANNSCache::linkCacheEntry(anns_cache_entry_t* allocated_entry, vector_id_t found_id) noexcept
    {
        context_->has_activity = true;

        ElapsedLatencyPair latency;
        latency.start();

        bool linked = false;
        size_t cache_entry_count = 0;
        size_t pool_size = 0;
        size_t approx_repr = 0;

        {
            std::lock_guard<SpinMutex> cache_guard(context_->cache_lock);

            linked = entry_store_->linkCacheEntryLocked(allocated_entry, found_id);

            cache_entry_count = context_->eviction_strategy->getCurrSize();
            pool_size = context_->vector_pool->getSize();
            approx_repr = context_->apprx_filter->getRepresentativeVectorNumber();
        }

        latency.end();
        latency.elapsedMs();

        context_->stats.appendLatencySample(ANNSCacheStats::LatencyMetric::k_link_cache_entry, latency);

        context_->stats.recordCacheHistorySample("linkCacheEntry", latency.getElapsedMs(), cache_entry_count, pool_size, approx_repr);

        return linked;
    }

    void
    ANNSCache::markVectorDeleted(vector_id_t vector_id) noexcept
    {
        context_->has_activity = true;

        ElapsedLatencyPair latency;
        latency.start();

        size_t cache_entry_count = 0;
        size_t pool_size = 0;
        size_t approx_repr = 0;

        {
            std::lock_guard<SpinMutex> cache_guard(context_->cache_lock);

            maintenance_->markVectorDeletedLocked(vector_id);

            cache_entry_count = context_->eviction_strategy->getCurrSize();
            pool_size = context_->vector_pool->getSize();
            approx_repr = context_->apprx_filter->getRepresentativeVectorNumber();
        }

        latency.end();
        latency.elapsedMs();

        context_->stats.appendLatencySample(ANNSCacheStats::LatencyMetric::k_mark_vector_deleted, latency);

        context_->stats.recordCacheHistorySample("markVectorDeleted", latency.getElapsedMs(), cache_entry_count, pool_size, approx_repr);
    }

    void
    ANNSCache::insertWriteLogEntry(
        vector_view_t write_vector,
        distance_function_t distance_function,
        result_conversion_function_t result_conversion_function) noexcept
    {
        context_->has_activity = true;

        ElapsedLatencyPair latency;
        latency.start();

        size_t cache_entry_count = 0;
        size_t pool_size = 0;
        size_t approx_repr = 0;

        /* Convert the write vector before entering the global cache critical section.
         * This is safe because conversion uses only the caller-provided vector buffer.
         */
        const size_t vector_dim = context_->parameter.vector_format.vector_dim;
        MallocPtr<float> write_vector_float = makeMallocPtr<float>(vector_dim);
        bool convert_success = write_vector.conversion_function(
            write_vector.vector_data,
            write_vector.vector_data_size,
            vector_dim,
            write_vector_float.get(),
            write_vector.aux);
        (void)convert_success;

        {
            std::lock_guard<SpinMutex> cache_guard(context_->cache_lock);

            maintenance_->insertWriteLogEntryLocked(
                write_vector,
                distance_function,
                result_conversion_function,
                write_vector_float.get());

            cache_entry_count = context_->eviction_strategy->getCurrSize();
            pool_size = context_->vector_pool->getSize();
            approx_repr = context_->apprx_filter->getRepresentativeVectorNumber();
        }

        latency.end();
        latency.elapsedMs();

        context_->stats.appendLatencySample(ANNSCacheStats::LatencyMetric::k_insert_write_log_entry, latency);

        context_->stats.recordCacheHistorySample("insertWriteLogEntry", latency.getElapsedMs(), cache_entry_count, pool_size, approx_repr);
    }

    void
    ANNSCache::processWriteLogEntries(
        distance_function_t distance_function,
        result_conversion_function_t result_conversion_function) noexcept
    {
        context_->has_activity = true;

        ElapsedLatencyPair latency;
        latency.start();

        size_t cache_entry_count = 0;
        size_t pool_size = 0;
        size_t approx_repr = 0;

        {
            std::lock_guard<SpinMutex> cache_guard(context_->cache_lock);

            maintenance_->processWriteLogEntriesLocked(distance_function, result_conversion_function);

            cache_entry_count = context_->eviction_strategy->getCurrSize();
            pool_size = context_->vector_pool->getSize();
            approx_repr = context_->apprx_filter->getRepresentativeVectorNumber();
        }

        latency.end();
        latency.elapsedMs();

        context_->stats.appendLatencySample(ANNSCacheStats::LatencyMetric::k_process_write_log_entries, latency);

        context_->stats.recordCacheHistorySample("processWriteLogEntries", latency.getElapsedMs(), cache_entry_count, pool_size, approx_repr);
    }

    void
    ANNSCache::resetCache() noexcept
    {
        context_->has_activity = true;

        std::lock_guard<SpinMutex> cache_guard(context_->cache_lock);
        maintenance_->resetCacheLocked();
    }

    void
    ANNSCache::stressTestInvalidateRandom(float percent) noexcept
    {
        std::lock_guard<SpinMutex> cache_guard(context_->cache_lock);
        maintenance_->stressTestInvalidateRandomLocked(percent);
    }

    void
    ANNSCache::collectPooledVectors(std::vector<VectorSlot*>& pooled_list) noexcept
    {
        std::lock_guard<SpinMutex> cache_guard(context_->cache_lock);
        maintenance_->collectPooledVectorsLocked(pooled_list);
    }

    std::string
    ANNSCache::getStatusText() noexcept
    {
        std::lock_guard<SpinMutex> cache_guard(context_->cache_lock);
        return telemetry_->buildStatusText();
    }

    std::string
    ANNSCache::getSummaryCsv() noexcept
    {
        std::lock_guard<SpinMutex> cache_guard(context_->cache_lock);
        return telemetry_->buildSummaryCsv();
    }

    void
    ANNSCache::exportTraceToFiles() noexcept
    {
        std::lock_guard<SpinMutex> cache_guard(context_->cache_lock);
        telemetry_->exportTraceToFiles();
    }
}
