#include <algorithm>
#include <array>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include "core/ak_ann_cache2_maintenance.hh"
#include "core/ak_ann_cache2_entry_store.hh"
#include "ak_logger.hh"
#include "utils/ak_malloc_ptr.hh"

namespace aker
{
    /* Helper: Frees a cache entry object itself (query vector + slot pointer array).
     * Slot VectorSlot elements are intentionally not deleted here.
     */
    static inline void
    destroyCacheEntryObject(result_cache_entry_t* entry) noexcept
    {
        if (entry == nullptr)
            return;

        delete entry->query_vector;
        entry->query_vector = nullptr;

        free(entry->vector_slot_ref_list);
        entry->vector_slot_ref_list = nullptr;

        delete entry;
    }

    ANNCache2Maintenance::ANNCache2Maintenance(ANNCache2Context* context, ANNCache2EntryStore* entry_store) noexcept
        : context_(context),
          entry_store_(entry_store)
    {
        assert(context_ != nullptr);
        assert(entry_store_ != nullptr);
    }

    bool
    ANNCache2Maintenance::needEvictLocked(size_t vector_list_size) noexcept
    {
        /* Conservative eviction decision based on pool size and incoming list size.
         */
        size_t current_pool_size = context_->vector_pool->getSize();
        return (current_pool_size + vector_list_size) > context_->parameter.capacity.slot_pool_size;
    }

    void
    ANNCache2Maintenance::evictVectorsLocked(size_t to_evicts, std::vector<vector_id_t>& evicted_list) noexcept
    {
        /* Evicts entries using the configured eviction strategy.
         * This preserves original semantics, including releasing pool vectors and unlinking child entries.
         */
        size_t eviction_size = 0;

        /* Emit eviction request for debugging.
         */
        AKER_LOG_DEBUG << "[ANNCacheMaintenance] eviction requested: to_evicts=" << to_evicts;

        for (size_t i = 0; i < to_evicts; i++)
        {
            vector_id_t evict_candidate_id = context_->eviction_strategy->nextEvictCandidate();
            if (evict_candidate_id == 0)
                break;

            result_cache_entry_t* evict_candidate_entry = entry_store_->getCEntry(evict_candidate_id);
            if (evict_candidate_entry == nullptr)
                continue;

            AKER_LOG_DEBUG << "[ANNCacheMaintenance] evicting entry: vector_id=" << evict_candidate_id
                          << " list_size=" << evict_candidate_entry->vector_list_size;

            /* Release all pooled vectors referenced by this entry.
             */
            for (size_t j = 0; j < evict_candidate_entry->vector_list_size; j++)
            {
                VectorSlot* vector = evict_candidate_entry->vector_slot_ref_list[j];
                if (vector == nullptr)
                    continue;

                bool deleted = context_->vector_pool->releaseVectorReference(vector->getVectorId());
                if (deleted)
                    eviction_size++;
            }

            /* Remove linked child entries.
             */
            result_cache_entry_t* child_entry = evict_candidate_entry->next;
            while (child_entry != nullptr)
            {
                vector_id_t child_vector_id = child_entry->query_vector->getVectorId();
                result_cache_entry_t* next = child_entry->next;

                context_->lookup_table->map.erase(child_vector_id);
                destroyCacheEntryObject(child_entry);

                child_entry = next;
            }

            /* Update write-log statistics based on checkpoint distance.
             */
            epoch_t unseen = 0;
            if (evict_candidate_entry->checkpoint != nullptr)
            {
                unseen = context_->write_log->getUnseenDistance(evict_candidate_entry->checkpoint);
                context_->write_log->releaseCheckpoint(evict_candidate_entry->checkpoint);
                evict_candidate_entry->checkpoint = nullptr;
            }

            context_->write_log->removeCacheEntryFromRoundRobin(static_cast<void*>(evict_candidate_entry));

            context_->lookup_table->map.erase(evict_candidate_id);
            context_->repr_entry_count = context_->eviction_strategy->getCurrSize();

            context_->write_log->removeCacheEntryRisk(evict_candidate_entry->risk_factor, unseen, context_->repr_entry_count);

            context_->evict_entry_count++;
            destroyCacheEntryObject(evict_candidate_entry);

            evicted_list.push_back(evict_candidate_id);
        }

        context_->stats.cache_evict += eviction_size;
        context_->write_log->trimUnreferencedHeadEntries();

        if (!evicted_list.empty())
        {
            AKER_LOG_INFO << "[ANNCacheMaintenance] evicted entries: count=" << evicted_list.size();
        }
    }

    bool
    ANNCache2Maintenance::insertCEntryLocked(
        vector_id_t vector_id,
        result_cache_entry_t* entry,
        vector_view_t query_vector_data) noexcept
    {
        /* Inserts a prepared entry into the cache.
         * This preserves the original insertion semantics.
         */
        /* Step-level latency is recorded as separate series.
         */

        ElapsedLatencyPair latency_int_1;
        ElapsedLatencyPair latency_int_2;
        ElapsedLatencyPair latency_int_3;

        result_cache_entry_t* allocated_entry = entry;

        int inserted = context_->lookup_table->map.try_emplace_or_visit(
            vector_id,
            entry,
            [&](const auto& pair)
            {
                (void)pair;
            });

        if (inserted == 0)
            return false;

        /* Optional eviction path.
         */
        if (needEvictLocked(allocated_entry->vector_list_size))
        {
            latency_int_1.start();

            std::vector<vector_id_t> evicted_list;
            evictVectorsLocked(allocated_entry->vector_list_size, evicted_list);

            if (!evicted_list.empty())
            {
                context_->apprx_filter->deleteVectors(evicted_list);
                AKER_LOG_DEBUG << "[ANNCacheMaintenance] approx filter deleted vectors: count=" << evicted_list.size();
                if (context_->apprx_filter->needSwitch())
                {
                    AKER_LOG_INFO << "[ANNCacheMaintenance] approx filter generation rotate triggered";
                    context_->apprx_filter->rotateGeneration();
                }
            }

            latency_int_1.end();
            latency_int_1.setAux1(evicted_list.size());
            context_->stats.appendLatencySample(ANNCacheStats::LatencyMetric::k_insert_cache_entry_step_1, latency_int_1);
        }

        /* Setup metadata for the inserted entry.
         */
        allocated_entry->checkpoint = context_->write_log->acquireTailCheckpoint();

        allocated_entry->prev = nullptr;
        allocated_entry->next = nullptr;
        allocated_entry->version = 0;

        /* Register representative vector in the approximate filter.
         */
        latency_int_2.start();
        context_->apprx_filter->addVector(query_vector_data);
        AKER_LOG_DEBUG << "[ANNCacheMaintenance] approx filter added representative vector: query_id="
                      << query_vector_data.vector_id;
        latency_int_2.end();
        context_->stats.appendLatencySample(ANNCacheStats::LatencyMetric::k_insert_cache_entry_step_2, latency_int_2);

        /* Transfer vector slots into the pool.
         */
        latency_int_3.start();

        float dist_topk = allocated_entry->vector_slot_ref_list[context_->parameter.capacity.vector_in_topk - 1]->getDistance();
        float dist_max = allocated_entry->max_distance;

        for (size_t i = 0; i < allocated_entry->vector_list_size; i++)
        {
            vector_id_t result_vector_id = allocated_entry->vector_slot_ref_list[i]->getVectorId();
            vector_data_t* result_vector_data = allocated_entry->vector_slot_ref_list[i]->getVectorData();
            float dist = allocated_entry->vector_slot_ref_list[i]->getDistance();

            VectorSlot* pooled_vector = context_->vector_pool->acquireOrCreateVector(result_vector_id, result_vector_data);
            allocated_entry->vector_slot_ref_list[i] = pooled_vector;

            float old_dist = pooled_vector->getDistance();
            if (old_dist > dist)
                pooled_vector->setDistance(dist);
        }

        allocated_entry->entry_kind = RESULT_CACHE_ENTRY_KIND_INTERNAL;

        assert(allocated_entry->vector_list_size >= context_->parameter.capacity.vector_in_topk);
        assert(dist_max >= dist_topk);

        if (dist_max == 0.0f)
        {
            dist_max = std::numeric_limits<float>::max();
            dist_topk = std::numeric_limits<float>::max();
        }

        allocated_entry->risk_factor = dist_topk / dist_max;

        context_->eviction_strategy->addEvictCandidate(vector_id);
        context_->repr_entry_count = context_->eviction_strategy->getCurrSize();

        AKER_LOG_DEBUG << "[ANNCacheMaintenance] inserted cache entry finalized: vector_id=" << vector_id
                      << " risk_factor=" << allocated_entry->risk_factor
                      << " repr_entry_count=" << context_->repr_entry_count;

        context_->write_log->addCacheEntryToRoundRobin(static_cast<void*>(allocated_entry));
        context_->write_log->addCacheEntryRisk(allocated_entry->risk_factor, 0, context_->repr_entry_count);

        latency_int_3.end();
        context_->stats.appendLatencySample(ANNCacheStats::LatencyMetric::k_insert_cache_entry_step_3, latency_int_3);

        context_->stats.approx_added_counts.push_back(context_->apprx_filter->getAddedCounts());
        context_->stats.approx_representative_counts.push_back(context_->apprx_filter->getRepresentativeVectorNumber());

        return true;
    }

    void
    ANNCache2Maintenance::consumeAgedWLEntryLocked(
        distance_function_t distance_function,
        result_conversion_function_t result_conversion_function) noexcept
    {
        /* Triggers the slow-path write-log scan if the read-to-repr ratio is high enough.
         */
        static constexpr double k_try_read_ratio_thresh = 0.25;

        if (context_->try_read_count > (k_try_read_ratio_thresh * context_->repr_entry_count))
        {
            if (context_->write_log->shouldRunSlowPath())
                incrBatchUpdateWLog2Locked(distance_function, result_conversion_function);
            context_->try_read_count = 0;
        }
    }

    void

    void
    ANNCache2Maintenance::updateWLEntryFastPathLocked(
        vector_view_t write_vector,
        distance_function_t distance_function,
        result_conversion_function_t result_conversion_function) noexcept
    {
        /* Fast-path write-log update that opportunistically improves a nearby entry.
         */
        static constexpr faiss::idx_t k_search_per_filter = 1;
        static constexpr size_t k_candidate_count = 2;

        MallocPtr<float> query_vector = makeMallocPtr<float>(context_->parameter.vector_format.vector_dim);

        std::array<float, k_candidate_count> distances{};
        std::array<faiss::idx_t, k_candidate_count> labels{};

        bool convert_success = write_vector.conversion_function(
            write_vector.vector_data,
            write_vector.vector_data_size,
            write_vector.vector_dim,
            query_vector.get(),
            write_vector.aux);
        (void)convert_success;

        context_->apprx_filter->searchSimilarVectors(
            query_vector.get(),
            k_search_per_filter,
            distances.data(),
            labels.data());

        for (size_t i = 0; i < static_cast<size_t>(k_search_per_filter); i++)
        {
            if (distances[i] == k_invalid_distance)
                continue;

            if (labels[i] < 0)
                continue;

            vector_id_t vector_id = static_cast<vector_id_t>(labels[i]);
            result_cache_entry_t* found_entry = entry_store_->getCEntry(vector_id);
            if (found_entry == nullptr)
                continue;

            float query_distance = distance_function(
                write_vector.vector_data,
                found_entry->query_vector->getVectorData(),
                write_vector.vector_dim);

            if (query_distance < found_entry->max_distance)
            {
                for (int j = static_cast<int>(found_entry->vector_list_size - 1); j >= 0; j--)
                {
                    if (!found_entry->vector_slot_ref_list[j]->isValid())
                        continue;

                    vector_id_t delete_vector_id = found_entry->vector_slot_ref_list[j]->getVectorId();
                    float old_distance = found_entry->vector_slot_ref_list[j]->getDistance();

                    vector_id_t write_vector_id = write_vector.vector_id;
                    vector_data_t* write_vector_data = write_vector.vector_data;

                    if (old_distance > query_distance)
                    {
                        VectorSlot* sub_vec = context_->vector_pool->replaceVectorReference(
                            delete_vector_id,
                            write_vector_id,
                            write_vector_data);

                        assert(sub_vec != nullptr);

                        std::uint64_t aux_data_1 = found_entry->query_vector->getAuxData1();
                        std::uint64_t aux_data_2 = found_entry->query_vector->getAuxData2();

                        if (result_conversion_function != nullptr)
                            result_conversion_function(
                                write_vector_id,
                                sub_vec->getVectorData(),
                                context_->vector_pool->getPayloadSize(),
                                aux_data_1,
                                aux_data_2);

                        found_entry->vector_slot_ref_list[j] = sub_vec;

                        if (j == static_cast<int>(found_entry->vector_list_size - 1))
                        {
                            found_entry->max_distance = query_distance;
                            found_entry->vector_slot_ref_list[j]->setDistance(query_distance);
                        }

                        break;
                    }
                }
            }
        }
    }

    void
    ANNCache2Maintenance::incrBatchUpdateWLog2Locked(
        distance_function_t distance_function,
        result_conversion_function_t result_conversion_function) noexcept
    {
        /* Slow-path write-log update.
         * This preserves the snapshot implementation (sweep + topk refresh).
         */
        /* Step-level latency is recorded as separate series.
         */

        ElapsedLatencyPair latency_int_1;
        ElapsedLatencyPair latency_int_2;

        cache_entry_handle_t rr_handle = context_->write_log->getNextCacheEntryFromRoundRobin();
        result_cache_entry_t* saved_entry = static_cast<result_cache_entry_t*>(rr_handle);
        if (saved_entry == nullptr)
            return;

        vector_id_t vector_id = saved_entry->query_vector->getVectorId();

        bool found = false;
        context_->lookup_table->map.visit(vector_id, [&](const auto& pair) { (void)pair; found = true; });
        assert(found == true);

        WriteLogScanResult scan_result;

        latency_int_1.start();
        scan_result = context_->write_log->scanLogWindow(
            saved_entry->query_vector->getVectorData(),
            context_->parameter.vector_format.vector_dim,
            saved_entry->max_distance,
            distance_function,
            saved_entry->checkpoint);

        std::sort(
            scan_result.candidates.begin(),
            scan_result.candidates.end(),
            [](const WriteLogCandidate& a, const WriteLogCandidate& b)
            {
                return a.distance > b.distance;
            });

        latency_int_1.end();
        context_->stats.appendLatencySample(ANNCacheStats::LatencyMetric::k_incr_batch_update_write_log_step_1, latency_int_1);

        latency_int_2.start();

        if (!scan_result.candidates.empty())
        {
            /* Merge refreshed candidates into the top-k list.
             */
            VectorSlot** new_slot_ref_list = new VectorSlot*[context_->parameter.capacity.vector_in_topk];
            std::memset(new_slot_ref_list, 0, sizeof(VectorSlot*) * context_->parameter.capacity.vector_in_topk);

            int new_slot_ref_index = 0;
            size_t vector_data_size = context_->vector_pool->getPayloadSize();

            for (int k = 0; k < static_cast<int>(context_->parameter.capacity.vector_in_topk); k++)
            {
                float current_distance = saved_entry->vector_slot_ref_list[k]->getDistance();
                if (!scan_result.candidates.empty())
                {
                    WriteLogCandidate& current_candidate = scan_result.candidates.back();
                    if (current_distance > current_candidate.distance)
                    {
                        vector_id_t delete_vector_id = saved_entry->vector_slot_ref_list[k]->getVectorId();

                        vector_id_t found_vector_id = current_candidate.vector_id;
                        const vector_data_t* found_vector_data = current_candidate.vector_data;

                        std::uint64_t aux_data_1 = current_candidate.aux_data_1;
                        std::uint64_t aux_data_2 = current_candidate.aux_data_2;

                        VectorSlot* sub_vec = context_->vector_pool->replaceVectorReference(
                            delete_vector_id,
                            found_vector_id,
                            found_vector_data);
                        sub_vec->setDistance(current_candidate.distance);

                        assert(sub_vec != nullptr);

                        if (result_conversion_function != nullptr)
                            result_conversion_function(
                                found_vector_id,
                                sub_vec->getVectorData(),
                                vector_data_size,
                                aux_data_1,
                                aux_data_2);

                        new_slot_ref_list[new_slot_ref_index] = sub_vec;
                        new_slot_ref_index++;

                        context_->write_log->recordRefresh();
                        scan_result.candidates.pop_back();
                    }
                    else
                    {
                        new_slot_ref_list[new_slot_ref_index] = saved_entry->vector_slot_ref_list[k];
                        new_slot_ref_index++;
                    }
                }
                else
                {
                    new_slot_ref_list[new_slot_ref_index] = saved_entry->vector_slot_ref_list[k];
                    new_slot_ref_index++;
                }
            }

            std::memcpy(
                saved_entry->vector_slot_ref_list,
                new_slot_ref_list,
                sizeof(VectorSlot*) * context_->parameter.capacity.vector_in_topk);

            delete[] new_slot_ref_list;
        }

        latency_int_2.end();
        context_->stats.appendLatencySample(ANNCacheStats::LatencyMetric::k_incr_batch_update_write_log_step_2, latency_int_2);

        context_->write_log->replaceCheckpoint(saved_entry->checkpoint, scan_result.new_checkpoint);
        context_->write_log->consumeUnseenDistance(scan_result.advanced_epoch_distance, context_->repr_entry_count);
    }

    void
    ANNCache2Maintenance::insertWLEntry3Locked(
        vector_view_t write_vector,
        distance_function_t distance_function,
        result_conversion_function_t result_conversion_function) noexcept
    {
        /* Write-log insertion is staged with latency sub-counters.
         */
        /* Step-level latency is recorded as separate series.
         */

        AKER_LOG_DEBUG << "[ANNCacheMaintenance] write-log insert begin: vector_id=" << write_vector.vector_id;

        ElapsedLatencyPair latency_int_1;
        ElapsedLatencyPair latency_int_2;
        ElapsedLatencyPair latency_int_3;
        ElapsedLatencyPair latency_int_4;

        latency_int_1.start();
        context_->write_log->insertLogEntry(
            write_vector.vector_id,
            write_vector.vector_data,
            static_cast<size_t>(write_vector.vector_data_size),
            write_vector.aux_data_1,
            write_vector.aux_data_2);
        latency_int_1.end();
        context_->stats.appendLatencySample(ANNCacheStats::LatencyMetric::k_insert_write_log_entry_step_1, latency_int_1);

        latency_int_2.start();
        updateWLEntryFastPathLocked(write_vector, distance_function, result_conversion_function);
        latency_int_2.end();
        context_->stats.appendLatencySample(ANNCacheStats::LatencyMetric::k_insert_write_log_entry_step_2, latency_int_2);

        latency_int_3.start();
        consumeAgedWLEntryLocked(distance_function, result_conversion_function);
        latency_int_3.end();
        context_->stats.appendLatencySample(ANNCacheStats::LatencyMetric::k_insert_write_log_entry_step_3, latency_int_3);

        latency_int_4.start();
        context_->write_log->trimUnreferencedHeadEntries();
        latency_int_4.end();
        context_->stats.appendLatencySample(ANNCacheStats::LatencyMetric::k_insert_write_log_entry_step_4, latency_int_4);

        AKER_LOG_DEBUG << "[ANNCacheMaintenance] write-log insert end";
    }

    void
    ANNCache2Maintenance::markVectorDeletedLocked(vector_id_t vector_id) noexcept
    {
        AKER_LOG_INFO << "[ANNCacheMaintenance] markVectorDeleted: vector_id=" << vector_id;
        context_->vector_pool->invalidateVector(vector_id);
    }

    void
    ANNCache2Maintenance::collectPooledVectorsLocked(std::vector<VectorSlot*>& pooled_list) noexcept
    {
        context_->vector_pool->collectPooledVectors(pooled_list);
    }

    void
    ANNCache2Maintenance::resetCacheLocked() noexcept
    {
        /* Reset write-log and eviction structures first to avoid stale pointers.
         */
        AKER_LOG_INFO << "[ANNCacheMaintenance] resetCacheLocked";
        context_->write_log->clear();

        context_->eviction_strategy = std::make_unique<EvictionStrategyFifo>();
        context_->repr_entry_count = 0;
        context_->evict_entry_count = 0;
        context_->try_read_count = 0;

        /* Clear cache storage and representative filter.
         */
        context_->lookup_table->map.clear();
        context_->vector_pool->clear();
        context_->apprx_filter->rotateGeneration();
    }

    void
    ANNCache2Maintenance::stressTestInvalidateRandomLocked(float percent) noexcept
    {
        if (percent < 0.0f || percent > 100.0f)
            return;

        size_t pool_size = context_->vector_pool->getSize();
        size_t target_invalidate = static_cast<size_t>(pool_size * (percent));

        std::vector<VectorSlot*> pooled_list;
        context_->vector_pool->collectPooledVectors(pooled_list);

        std::random_shuffle(pooled_list.begin(), pooled_list.end());

        for (auto& vec : pooled_list)
        {
            if (target_invalidate == 0)
                break;

            context_->vector_pool->invalidateVector(vec->getVectorId());
            target_invalidate--;
        }

        context_->stats.cache_hit = 0;
        context_->stats.cache_miss = 0;
        context_->stats.cache_invalid_detect = 0;
        context_->stats.cache_evict = 0;
        context_->stats.cache_sim_hit = 0;
    }
}
