#include "core/ak_anns_cache_telemetry.hh"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace aker
{
    namespace
    {
        /**
         * @brief Cache summary file name.
         */
        static constexpr const char* k_cache_summary_file = "aker_trace_cache_summary.csv";

        /**
         * @brief Cache status file name.
         */
        static constexpr const char* k_cache_status_file = "aker_trace_cache_status.txt";

        /**
         * @brief Cache parameter snapshot file name.
         */
        static constexpr const char* k_cache_parameters_file = "aker_trace_parameters.csv";

        /**
         * @brief Write-log metric snapshot file name.
         */
        static constexpr const char* k_write_log_metrics_file = "aker_trace_write_log.csv";
    }

    ANNSCacheTelemetry::ANNSCacheTelemetry(ANNSCacheContext* context) noexcept
        : context_(context)
    {
        assert(context_ != nullptr);
    }

    std::string ANNSCacheTelemetry::buildStatusText() noexcept
    {
        /* Build a human-readable cache status string.
         */
        std::ostringstream oss;

        const size_t total_entry_count = context_->lookup_table->map.size();
        size_t virtual_entry_count = 0;
        size_t valid_entry_count = 0;

        for (const auto& pair : context_->lookup_table->map)
        {
            anns_cache_entry_t* entry = pair.second;
            if (entry == nullptr)
                continue;

            if (entry->neighbors_list == nullptr)
                continue;

            if (entry->next != nullptr)
            {
                size_t linked_count = 0;
                anns_cache_entry_t* next_entry = entry->next;
                while (next_entry != nullptr)
                {
                    ++linked_count;
                    next_entry = next_entry->next;
                }
                virtual_entry_count += linked_count;
            }

            if (entry->version != -1)
                ++valid_entry_count;
        }

        const size_t physical_entry_count = (total_entry_count >= virtual_entry_count)
                                               ? (total_entry_count - virtual_entry_count)
                                               : 0;

        const float denom = static_cast<float>(context_->stats.cache_hit + context_->stats.cache_miss + context_->stats.cache_sim_hit);
        const float exact_hit_ratio = (denom == 0.0f) ? 0.0f : (static_cast<float>(context_->stats.cache_hit) / denom);
        const float total_hit_ratio = (denom == 0.0f)
                                          ? 0.0f
                                          : (static_cast<float>(context_->stats.cache_hit + context_->stats.cache_sim_hit) / denom);

        oss << "ANNSCache Status\n";
        oss << "  Total entries: " << total_entry_count << "\n";
        oss << "  Physical entries: " << physical_entry_count << "\n";
        oss << "  Virtual entries: " << virtual_entry_count << "\n";
        oss << "  Valid entries: " << valid_entry_count << "\n";
        oss << "\n";

        oss << "  Vector pool: " << context_->vector_pool->getStatusText() << "\n";
        oss << "  Write log: " << context_->write_log->getStatusText() << "\n";
        oss << "\n";

        oss << "  Cache hit: " << context_->stats.cache_hit << "\n";
        oss << "  Cache sim-hit: " << context_->stats.cache_sim_hit << "\n";
        oss << "  Cache miss: " << context_->stats.cache_miss << "\n";
        oss << "  Cache evict: " << context_->stats.cache_evict << "\n";
        oss << "  Cache invalid-detect: " << context_->stats.cache_invalid_detect << "\n";
        oss << "  Cache dropout: " << context_->stats.cache_dropout << "\n";
        oss << "  Exact hit ratio: " << exact_hit_ratio << "\n";
        oss << "  Total hit ratio: " << total_hit_ratio << "\n";
        oss << "\n";

        oss << "  Eviction queue size: " << context_->eviction_strategy->getCurrSize() << "\n";
        oss << "  Approx repr vectors: " << context_->apprx_filter->getRepresentativeVectorNumber() << "\n";
        oss << "  Approx added vectors: " << context_->apprx_filter->getAddedCounts() << "\n";

        return oss.str();
    }

    std::string ANNSCacheTelemetry::buildSummaryCsv() noexcept
    {
        /* Build a concise CSV key/value snapshot.
         */
        std::ostringstream oss;
        oss << "metric,value\n";

        const size_t total_entry_count = context_->lookup_table->map.size();
        size_t virtual_entry_count = 0;
        size_t valid_entry_count = 0;

        for (const auto& pair : context_->lookup_table->map)
        {
            anns_cache_entry_t* entry = pair.second;
            if (entry == nullptr)
                continue;

            if (entry->neighbors_list == nullptr)
                continue;

            if (entry->next != nullptr)
            {
                size_t linked_count = 0;
                anns_cache_entry_t* next_entry = entry->next;
                while (next_entry != nullptr)
                {
                    ++linked_count;
                    next_entry = next_entry->next;
                }
                virtual_entry_count += linked_count;
            }

            if (entry->version != -1)
                ++valid_entry_count;
        }

        const size_t physical_entry_count = (total_entry_count >= virtual_entry_count)
                                               ? (total_entry_count - virtual_entry_count)
                                               : 0;

        size_t checkpoint_count = 0;
        for (const auto& pair : context_->lookup_table->map)
        {
            anns_cache_entry_t* entry = pair.second;
            if (entry == nullptr)
                continue;
            if (entry->checkpoint != nullptr)
                ++checkpoint_count;
        }

        oss << "TotalEntryCount," << total_entry_count << "\n";
        oss << "PhysicalEntryCount," << physical_entry_count << "\n";
        oss << "VirtualEntryCount," << virtual_entry_count << "\n";
        oss << "ValidEntryCount," << valid_entry_count << "\n";
        oss << "CheckpointCount," << checkpoint_count << "\n";

        oss << "VectorPoolSize," << context_->vector_pool->getSize() << "\n";
        oss << "VectorPoolCapacity," << context_->vector_pool->getCapacity() << "\n";
        oss << "VectorPoolShards," << context_->vector_pool->getShardCount() << "\n";

        /* Export a few write-log health metrics.
         */
        const WriteLogMetrics write_log_metrics = context_->write_log->getMetrics();
        oss << "WriteLogEntryCount," << write_log_metrics.log_entry_count << "\n";
        oss << "WriteLogCurrentRisk," << write_log_metrics.current_risk << "\n";
        oss << "WriteLogTotalUnseen," << write_log_metrics.total_unseen << "\n";
        oss << "WriteLogRefreshCount," << write_log_metrics.refresh_count << "\n";

        oss << "CacheHit," << context_->stats.cache_hit << "\n";
        oss << "CacheSimHit," << context_->stats.cache_sim_hit << "\n";
        oss << "CacheMiss," << context_->stats.cache_miss << "\n";
        oss << "CacheEvict," << context_->stats.cache_evict << "\n";
        oss << "CacheInvalidDetect," << context_->stats.cache_invalid_detect << "\n";
        oss << "CacheDropout," << context_->stats.cache_dropout << "\n";

        const float denom = static_cast<float>(context_->stats.cache_hit + context_->stats.cache_miss + context_->stats.cache_sim_hit);
        const float exact_hit_ratio = (denom == 0.0f) ? 0.0f : (static_cast<float>(context_->stats.cache_hit) / denom);
        const float total_hit_ratio = (denom == 0.0f)
                                          ? 0.0f
                                          : (static_cast<float>(context_->stats.cache_hit + context_->stats.cache_sim_hit) / denom);

        oss << "ExactHitRatio," << exact_hit_ratio << "\n";
        oss << "TotalHitRatio," << total_hit_ratio << "\n";

        oss << "EvictionQueueSize," << context_->eviction_strategy->getCurrSize() << "\n";
        oss << "ApproxReprCount," << context_->apprx_filter->getRepresentativeVectorNumber() << "\n";
        oss << "ApproxAddedCount," << context_->apprx_filter->getAddedCounts() << "\n";

        return oss.str();
    }

    void ANNSCacheTelemetry::exportTraceToFiles() noexcept
    {
        /* Export all telemetry under a single /tmp timestamped directory.
         */
        if (!context_->has_activity)
        {
            return;
        }

        const std::string directory_path = context_->stats.exportTraceToFiles();
        if (directory_path.empty())
        {
            return;
        }

        {
            const std::string file_name = directory_path + std::string(k_cache_summary_file);
            std::ofstream file(file_name, std::ios::out);
            if (file.is_open())
            {
                file << buildSummaryCsv();
                file.close();
            }
        }

        {
            const std::string file_name = directory_path + std::string(k_cache_status_file);
            std::ofstream file(file_name, std::ios::out);
            if (file.is_open())
            {
                file << buildStatusText();
                file.close();
            }
        }

        {
            /* Export write-log metrics as a concise CSV snapshot.
             */
            const std::string file_name = directory_path + std::string(k_write_log_metrics_file);
            std::ofstream file(file_name, std::ios::out);
            if (file.is_open())
            {
                file << context_->write_log->buildMetricsCsv();
                file.close();
            }
        }

        {
            /* Export cache parameter snapshot for reproducibility.
             */
            const std::string file_name = directory_path + std::string(k_cache_parameters_file);
            std::ofstream file(file_name, std::ios::out);
            if (file.is_open())
            {
                file << "parameter,value\n";
                file << "dimension," << context_->parameter.vector_format.dimension << "\n";
                file << "pool_size," << context_->parameter.capacity.pool_size << "\n";
                file << "vector_in_bytes," << context_->parameter.vector_format.vector_in_bytes << "\n";
                file << "in_topk," << context_->parameter.capacity.in_topk << "\n";
                file << "top_delta," << context_->parameter.capacity.top_delta << "\n";

                file << "global_thresh," << context_->parameter.tuning.global_thresh << "\n";
                file << "dropout," << context_->parameter.tuning.dropout << "\n";
                file << "risk_thresh," << context_->parameter.tuning.risk_thresh << "\n";
                file << "alpha_tighten," << context_->parameter.tuning.alpha_tighten << "\n";
                file << "alpha_loosen," << context_->parameter.tuning.alpha_loosen << "\n";

                file << "distance_metric," << static_cast<int>(context_->parameter.distance_metric) << "\n";
                file.close();
            }
        }
    }
}
