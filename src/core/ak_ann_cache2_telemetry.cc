#include "core/ak_ann_cache2_telemetry.hh"

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

    ANNCache2Telemetry::ANNCache2Telemetry(ANNCache2Context* context) noexcept
        : context_(context)
    {
        assert(context_ != nullptr);
    }

    std::string ANNCache2Telemetry::buildStatusText() noexcept
    {
        /* Build a human-readable cache status string.
         */
        std::ostringstream oss;

        const size_t total_entry_count = context_->lookup_table->map.size();
        size_t virtual_entry_count = 0;
        size_t valid_entry_count = 0;

        context_->lookup_table->map.visit_all(
            [&](const auto& pair)
            {
                if (pair.second->vector_slot_ref_list == nullptr)
                {
                    return;
                }

                if (pair.second->next != nullptr)
                {
                    size_t linked_count = 0;
                    result_cache_entry_t* next_entry = pair.second->next;
                    while (next_entry != nullptr)
                    {
                        ++linked_count;
                        next_entry = next_entry->next;
                    }
                    virtual_entry_count += linked_count;
                }

                if (pair.second->version != -1)
                {
                    ++valid_entry_count;
                }
            });

        const size_t physical_entry_count = (total_entry_count >= virtual_entry_count)
                                               ? (total_entry_count - virtual_entry_count)
                                               : 0;

        const float denom = static_cast<float>(context_->stats.cache_hit + context_->stats.cache_miss + context_->stats.cache_sim_hit);
        const float exact_hit_ratio = (denom == 0.0f) ? 0.0f : (static_cast<float>(context_->stats.cache_hit) / denom);
        const float total_hit_ratio = (denom == 0.0f)
                                          ? 0.0f
                                          : (static_cast<float>(context_->stats.cache_hit + context_->stats.cache_sim_hit) / denom);

        oss << "ANNCache2 Status\n";
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
        oss << "  Exact hit ratio: " << exact_hit_ratio << "\n";
        oss << "  Total hit ratio: " << total_hit_ratio << "\n";
        oss << "\n";

        oss << "  Eviction queue size: " << context_->eviction_strategy->getCurrSize() << "\n";
        oss << "  Approx repr vectors: " << context_->apprx_filter->getRepresentativeVectorNumber() << "\n";
        oss << "  Approx added vectors: " << context_->apprx_filter->getAddedCounts() << "\n";

        return oss.str();
    }

    std::string ANNCache2Telemetry::buildSummaryCsv() noexcept
    {
        /* Build a concise CSV key/value snapshot.
         */
        std::ostringstream oss;
        oss << "metric,value\n";

        const size_t total_entry_count = context_->lookup_table->map.size();
        size_t virtual_entry_count = 0;
        size_t valid_entry_count = 0;

        context_->lookup_table->map.visit_all(
            [&](const auto& pair)
            {
                if (pair.second->vector_slot_ref_list == nullptr)
                {
                    return;
                }

                if (pair.second->next != nullptr)
                {
                    size_t linked_count = 0;
                    result_cache_entry_t* next_entry = pair.second->next;
                    while (next_entry != nullptr)
                    {
                        ++linked_count;
                        next_entry = next_entry->next;
                    }
                    virtual_entry_count += linked_count;
                }

                if (pair.second->version != -1)
                {
                    ++valid_entry_count;
                }
            });

        const size_t physical_entry_count = (total_entry_count >= virtual_entry_count)
                                               ? (total_entry_count - virtual_entry_count)
                                               : 0;

        size_t checkpoint_count = 0;
        context_->lookup_table->map.visit_all(
            [&](const auto& pair)
            {
                if (pair.second->checkpoint != nullptr)
                {
                    ++checkpoint_count;
                }
            });

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

    void ANNCache2Telemetry::exportTraceToFiles() noexcept
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
                file << "vector_dim," << context_->parameter.vector_format.vector_dim << "\n";
                file << "vector_pool_size," << context_->parameter.capacity.slot_pool_size << "\n";
                file << "vector_list_size," << context_->parameter.capacity.slot_list_size << "\n";
                file << "vector_data_size," << context_->parameter.vector_format.vector_data_size << "\n";
                file << "vector_intopk," << context_->parameter.capacity.vector_in_topk << "\n";
                file << "vector_extras," << context_->parameter.capacity.vector_extras << "\n";

                file << "similar_match," << static_cast<int>(context_->parameter.tuning.similar_match) << "\n";
                file << "use_fixed_thresh," << static_cast<int>(context_->parameter.tuning.use_fixed_thresh) << "\n";
                file << "fixed_thresh," << context_->parameter.tuning.fixed_thresh << "\n";
                file << "start_thresh," << context_->parameter.tuning.start_thresh << "\n";
                file << "risk_thresh," << context_->parameter.tuning.risk_thresh << "\n";
                file << "alpha_tighten," << context_->parameter.tuning.alpha_tighten << "\n";
                file << "alpha_loosen," << context_->parameter.tuning.alpha_loosen << "\n";

                file << "distance_type," << static_cast<int>(context_->parameter.distance_type) << "\n";
                file.close();
            }
        }
    }
}
