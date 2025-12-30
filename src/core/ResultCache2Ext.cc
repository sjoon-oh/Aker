
#include <vector>
#include <string>
#include <limits>
#include <algorithm>

#include <cstdio>

#include "core/ResultCache2.hh"

// 
// For reference: https://github.com/boostorg/unordered/blob/develop/test/cfoa/interprocess_concurrency_tests.cpp

// ResultCache::
namespace topkache
{
    
    static inline void
    test_validity_reflist(result_cache_entry_t* entry)
    {
        if (entry == nullptr)
            return;

        int count = 0;
        for (int i = 0; i < entry->vector_list_size; i++)
        {
            if (entry->vector_slot_ref_list[i] != nullptr)
                continue;

            else
                count++;
        }

        if (count > 0)
        {
            printf("Invalid vectors in the entry: %d\n", count);
            assert(count == 0);
        }
    }

#pragma region EXTERNAL

    /* Make a cache entry
     *  All data should be prepared before calling this function outside of the caller.
     *  makeCEntry does not allocate any memory for the vectors, but just for the entry.
     */
    result_cache_entry_t*
    ResultCache2::makeCEntry(
        Vector2* query_vector, std::uint32_t list_size, Vector2** vector_local_reference_list) noexcept
    {
        used = true;
        
        result_cache_entry_t* entry = new result_cache_entry_t();

        entry->query_vector                 = new Vector2(vector_pool->getVecDataSize());
        entry->query_vector->setVecVersion(query_vector->getVecVersion());
        entry->query_vector->setVecId(query_vector->getVecId());
        std::memcpy(entry->query_vector->getVecData(), 
            query_vector->getVecData(), vector_pool->getVecDataSize());

        entry->version                      = 0;

        entry->vector_list_size             = list_size;
        entry->entry_status                 = RESULT_CACHE_ENTRY_STATUS_VALID;

        entry->vector_slot_ref_list   = (Vector2**)aligned_alloc(8, sizeof(Vector2*) * list_size);
        if (vector_local_reference_list != nullptr)
        {
            std::memcpy(entry->vector_slot_ref_list, 
                vector_local_reference_list, sizeof(Vector2*) * list_size);
            entry->min_distance
                = entry->vector_slot_ref_list[0]->getDistance();
            entry->max_distance
                = entry->vector_slot_ref_list[list_size - 1]->getDistance();
        }
        else
        {
            entry->vector_slot_ref_list = nullptr;
            entry->min_distance = std::numeric_limits<float>::max();
            entry->max_distance = std::numeric_limits<float>::min();
        }

        entry->threshold        = entry->min_distance;
        entry->threshold        *= parameter.start_threshold;

        if (parameter.similar_match == 0)
            entry->threshold = 0;           // Forcefully set to 0, since we do not need this.

        entry->prev                         = nullptr;
        entry->next                         = nullptr;
        entry->checkpoint                   = nullptr;

        return entry;
    }
    
    /* Free the cache entry
     *  This function frees the memory allocated for the vector_slot_ref_list 
     *  and the entry itself.
     */
    void
    ResultCache2::freeCEntry(result_cache_entry_t* entry) noexcept
    {
        used = true;

        free(entry->vector_slot_ref_list);
        delete entry;
    }

    /* Get the cache entry with the given vector_id
     *  When the entry is returned, the entry is locked and should be unlocked by releaseCEntry.
     */
    result_cache_entry_t*
    ResultCache2::exGetCEntryPassive(
        vector_id_t vector_id) noexcept
    {
        used = true;

        std::vector<ElapsedPair>& latency_measure 
            = stats.stat_level_0["exGetCEntryPassive"];
        ElapsedPair latency_0;
        
        latency_0.start();

        _lock();
        result_cache_entry_t* entry = _getCEntry(vector_id);

        if (entry != nullptr)
        {
            stats.cache_hit++;
            stats.recordHitHistory();
        }
        else
        {
            stats.cache_miss++;
            stats.recordHitHistory();
        }

        latency_0.end();
        latency_measure.push_back(latency_0);

        _unlock();

        return entry;
    }

    result_cache_entry_t*
    ResultCache2::exGetCEntryActive(
        vector_id_t vector_id,
        bool& is_invalid
    ) noexcept
    {

        used = true;

        std::vector<ElapsedPair>& latency_measure 
            = stats.stat_level_0["exGetCEntryActive"];
        ElapsedPair latency_0;

        is_invalid = false;
        
        latency_0.start();

        _lock();
        result_cache_entry_t* entry = _getCEntry(vector_id);

        if (entry != nullptr)
        {
            bool is_valid = _handleInvalidCEntry(entry);

            if (is_valid == false)
            {
                is_invalid = true;

                entry = nullptr;

                stats.cache_miss++;
                stats.recordHitHistory();
            }
            else
            {
                stats.cache_hit++;
                stats.recordHitHistory();
            }
        }
        else
            stats.cache_miss++;

        latency_0.end();
        latency_measure.push_back(latency_0);

        _unlock();
        return entry;
    }

#pragma region simGetCEntry

    result_cache_entry_t*
    ResultCache2::simGetCEntry(
        float_qvec_t query_vector_data,
        bool& similar_entry, bool& is_invalid,
        distance_function_t distance_function,
        bool exhaustive_search
    ) noexcept
    {
        used = true;

        std::vector<ElapsedPair>& latency_measure   = stats.stat_level_0["simGetCEntry"];
        std::vector<ElapsedPair>& latency_sub_1     = stats.stat_level_1["simGetCEntry-1"];
        std::vector<ElapsedPair>& latency_sub_2     = stats.stat_level_1["simGetCEntry-2"];
        std::vector<ElapsedPair>& latency_sub_3     = stats.stat_level_1["simGetCEntry-3"];

        ElapsedPair latency_0, latency_int_1, latency_int_2, latency_int_3;

        latency_0.start();

        _lock();

        try_read_cnt++;

        // Default, we regard the entry as not similar.
        similar_entry = false;
        is_invalid = false;

        // 
        // Phase 1: First, get the exact match from the cache.
        //  This phase is the same as the exGetCEntryPassive function.
        result_cache_entry_t* entry = _getCEntry(query_vector_data.vector_id);

        if (entry != nullptr)
        {
            bool is_valid = _handleInvalidCEntry(entry);
            if (is_valid == false)
            {
                is_invalid = true;
                entry = nullptr;
            }
            else
            {
                bool linked_hit = (query_vector_data.vector_id != entry->query_vector->getVecId());
                if (linked_hit)
                {
                    stats.cache_sim_hit++;
                    stats.recordHitHistory();
                }
                else 
                {
                    stats.cache_hit++;
                    stats.recordHitHistory();
                }

                eviction_strategy->recentlyAccessed(entry->query_vector->getVecId());
                
                if (!linked_hit)
                {
                    entry->threshold = entry->threshold * (parameter.alpha_loosen); // Slightly increase the threshold
                    if (entry->threshold > entry->min_distance)
                        entry->threshold = entry->min_distance;
                }

                result_cache_entry_t* copy_entry = _copy_cache_entry(entry);
                entry = copy_entry;

                latency_0.end();
                latency_measure.push_back(latency_0);

                is_invalid = false;

                _unlock();
                return entry;
            }
        }

        // Phase 2: If the exact match is not found, search the approximate filter.
        //  We set the search number to 1, since we only need to find the most similar entry.
        const int num_query     = 1;
        size_t search_num       = 1;

        float* query_vector     = (float*)malloc((sizeof(float) * query_vector_data.vector_dim));
        float* distance         = (float*)malloc(sizeof(float) * (search_num * 2));
        faiss::idx_t* labels    = (faiss::idx_t*)malloc(sizeof(faiss::idx_t) * (search_num * 2));
        int* filter_ids         = new int[search_num * 2];

        bool convert_success = query_vector_data.conversion_function(
            query_vector_data.vector_data,              // The vector_data_t to be converted
            query_vector_data.vector_data_size,         // The total size of the vector_data_t
            query_vector_data.vector_dim,               // The dimension of the vector_data_t
            query_vector,                               // The destination float*
            query_vector_data.aux
        );

        if (!exhaustive_search)
            apprx_filter->searchSimVecs(
                query_vector, search_num, distance, labels
            );
        else
        {
            // Fill the labels with the brute-force search
            // Namely, the labels are the vector_ids
            std::vector<faiss::idx_t> ex_labels(search_num * 2, -1);
            std::vector<float> ex_distances(search_num * 2, INVALID_DISTANCE);
            _getSimCEntryEx(
                query_vector_data, distance_function, ex_labels, ex_distances
            );
        }

        // Convert the labels to the vector_id
        // If found at any point, break the loop.
        bool is_valid = false;
        for (int i = 0; i < search_num * 2; i++)
        {

            vector_id_t vector_id = 0;
            if (distance[i] == INVALID_DISTANCE)
                continue;

            vector_id = static_cast<vector_id_t>(labels[i]);
            result_cache_entry_t* found_entry = _getCEntry(vector_id);
            if (found_entry == nullptr)
                continue;

            is_valid = _handleInvalidCEntry(found_entry);
            if (is_valid == false)
            {
                entry = nullptr;
                continue;
            }

            float threshold = 0.0f;
            if (parameter.similar_match)
            {
                if (parameter.use_fixed_threshold)
                    threshold = parameter.fixed_threshold;

                else
                    threshold = std::min(found_entry->min_distance, found_entry->threshold);
            }

            float similarity_threshold = threshold;
            float query_distance = distance_function(
                query_vector_data.vector_data,                  // The query vector data
                found_entry->query_vector->getVecData(),        // The found entry vector data
                query_vector_data.vector_dim
            );

            if (query_distance < similarity_threshold)
            {
                entry = found_entry;                            // Return this as the found entry, do not check the rest
                stats.cache_sim_hit++;
                stats.recordHitHistory();

                similar_entry = true;

                found_entry->threshold = found_entry->threshold * (parameter.alpha_tighten); // Half the threhsold
                if (found_entry->threshold > found_entry->min_distance)
                    found_entry->threshold = found_entry->min_distance;

                break;
            }
        }

        if (entry == nullptr)
        {
            is_invalid = (is_valid == false) ? true : false;
            stats.cache_miss++;
            stats.recordHitHistory();
        }
        else
        {
            eviction_strategy->recentlyAccessed(entry->query_vector->getVecId());

            // Until here, we have now valid entry
            // We make the exact copy, and return it.
            result_cache_entry_t* copy_entry = _copy_cache_entry(entry);
            entry = copy_entry;
        }
    
        free(query_vector);
        free(distance);
        free(labels);
        
        delete[] filter_ids;

        // We unlock here
        // releaseCEntry(entry);

        latency_0.end();
        latency_measure.push_back(latency_0);

        stats.apprx_added.push_back(apprx_filter->getAddedCounts());
        stats.apprx_nrepr.push_back(apprx_filter->getReprVecNum());

        _unlock();
        return entry;
    }

#pragma endregion
#pragma region insertCEntry

    bool
    ResultCache2::insertCEntry2(
        vector_id_t vector_id, result_cache_entry_t* entry, float_qvec_t query_vector_data) noexcept
    {
        used = true;

        std::vector<ElapsedPair>& latency_measure   = stats.stat_level_0["insertCEntry"];
        std::vector<ElapsedPair>& latency_sub_1     = stats.stat_level_1["insertCEntry-1"];
        std::vector<ElapsedPair>& latency_sub_2     = stats.stat_level_1["insertCEntry-2"];
        std::vector<ElapsedPair>& latency_sub_3     = stats.stat_level_1["insertCEntry-3"];

        ElapsedPair latency_0, latency_int_1, latency_int_2, latency_int_3;
        latency_0.start();

        _lock();

        result_cache_entry_t* allocated_entry = entry;
        int inserted = lookup_table->map.try_emplace_or_visit(vector_id, entry, [&](const auto& pair) { });

        // 
        // Other thread may have inserted the entry.
        // In this case, we give up and return false.
        if (inserted == 0)
        {
            latency_0.end();
            latency_measure.push_back(latency_0);

            _unlock();
            return false;
        }

        // Evict the entries if necessary
        // Since the insertCEntryLin holds the lock, we assume that the overall size of the vectors
        // in the pool is not changed.
        // We only check the current number of elements in the pool, and the full capacity of the pool.
        if (_needEvict(allocated_entry->vector_list_size))
        {
            latency_int_1.start();

            std::vector<vector_id_t> evicted_list;

            _evictVecs2(allocated_entry->vector_list_size, evicted_list);
            if (evicted_list.size() > 0)
            {
                apprx_filter->deleteVecs(evicted_list);
                if (apprx_filter->needSwitch())
                    apprx_filter->clear();
            }

            latency_int_1.end();
            latency_int_1.setAux1(evicted_list.size());

            latency_sub_1.push_back(latency_int_1);
        }

        allocated_entry->checkpoint = rf_write_log->getTail();

        // When setting the checkpoint, make sure to mark the reference count
        if (allocated_entry->checkpoint != nullptr)
            allocated_entry->checkpoint->vector.increaseRefCount();

        allocated_entry->prev = nullptr;
        allocated_entry->next = nullptr;

        allocated_entry->version = 0;

        latency_int_2.start();

        // In case when not linked, we need to insert the entry to the approximate filter.
        apprx_filter->addVec(query_vector_data);

        latency_int_2.end();
        latency_sub_2.push_back(latency_int_2);

        latency_int_3.start();

        float dist_topk = allocated_entry->vector_slot_ref_list[parameter.vector_intopk - 1]->getDistance();
        float dist_max = allocated_entry->max_distance;

                // Allocate the vectors
        size_t vector_data_size = vector_pool->getVecDataSize();
        for (size_t i = 0; i < allocated_entry->vector_list_size; i++)
        {
            vector_id_t result_vector_id 
                = allocated_entry->vector_slot_ref_list[i]->getVecId();
            
            vector_data_t* result_vector_data 
                = allocated_entry->vector_slot_ref_list[i]->getVecData();

            float dist = allocated_entry->vector_slot_ref_list[i]->getDistance();

            // Here, if the vector is already allocated, allocateVec will not replace but increase the reference count.
            Vector2* allocated_vector = vector_pool->allocateVec(
                result_vector_id, result_vector_data, vector_data_size
                );
            allocated_entry->vector_slot_ref_list[i] = allocated_vector;

            // Fill the vector metadata, mark it with bigger one (if previously allocated)
            float old_dist = allocated_vector->getDistance();

            // Initial value is float_max.
            if (old_dist > dist)
                allocated_vector->setDistance(dist);
        }

        // 
        // Delta Delay Factor:
        // The assumption is that the index parameter.vector_intopk - 1 is always valid
        assert(allocated_entry->vector_list_size >= parameter.vector_intopk);
        assert(dist_max >= dist_topk);

        if (dist_max == 0.0f)
        {
            // Assuming the exact search
            dist_max = std::numeric_limits<float>::max();
            dist_topk = std::numeric_limits<float>::max();
        }

        float risk_factor = dist_topk / dist_max;

        allocated_entry->risk_factor = risk_factor;

        // fprintf(stdout, "dist_topk: %f, dist_max: %f\n", 
        //     dist_topk, dist_max);
        // fprintf(stdout, "risk_factor: %f\n", allocated_entry->risk_factor);
        // fflush(stdout);

        eviction_strategy->addEvictCandidate(vector_id);
        repr_entry_cnt = eviction_strategy->getCurrSize();

        rf_write_log->addToRr(reinterpret_cast<std::uint8_t*>(allocated_entry));
        rf_write_log->increaseCEntryRf(risk_factor, 0, repr_entry_cnt);

        latency_int_3.end();
        latency_sub_3.push_back(latency_int_3);

        latency_0.end();
        latency_measure.push_back(latency_0);

        stats.apprx_added.push_back(apprx_filter->getAddedCounts());
        stats.apprx_nrepr.push_back(apprx_filter->getReprVecNum());

        _unlock();

        return true;
    }

    void
    ResultCache2::consumeAgedWLEntry(
        distance_function_t distance_function,
        result_conversion_function_t result_conversion_function
    ) noexcept
    {
        used = true;

        _lock();

        std::vector<ElapsedPair>& latency_measure 
            = stats.stat_level_0["consumeAgedWLEntry"];

        ElapsedPair latency_0;
        latency_0.start();

        // If tried cache reads for some threshold, 
        //  trigger the slow path.
        if (try_read_cnt > (0.25 * repr_entry_cnt))
        {
            if (rf_write_log->needRunSlowPath())
                _incrBatchUpdateWLog2(distance_function, result_conversion_function);
            try_read_cnt = 0;
        }

        latency_0.end();
        latency_measure.push_back(latency_0);

        _unlock();
    }

#pragma endregion
#pragma region linkCEntry

    bool
    ResultCache2::linkCEntry(
        result_cache_entry_t* allocated_entry,
        vector_id_t found_id) noexcept
    {
        used = true;

        _lock();

        // At this point, this may be the one that is evicted, so jump out.
        result_cache_entry_t* root_entry = _getCEntry(found_id);
        if (root_entry == nullptr)
        {
            _unlock();
            return false;
        }

        int inserted = lookup_table->map.try_emplace_or_visit(
            allocated_entry->query_vector->getVecId(), 
            allocated_entry, 
            [&](const auto& pair) { }
        );

        // If somebody has inserted whether it is linked or not, we give up.
        if (inserted == 0)
        {
            _unlock();
            return false;
        }

        // Some other thread may have linked the entry.
        // In this case, the root_entry may not be the same as the real root.
        // Anyhow, we link the entry to the root_entry.

        allocated_entry->prev = root_entry;
        allocated_entry->next = root_entry->next;

        root_entry->next = allocated_entry;

        _unlock();
        return true;
    }

#pragma endregion
#pragma region insertWLEntry

    void
    ResultCache2::insertWLEntry3(
        float_qvec_t write_vector,
        distance_function_t distance_function,
        result_conversion_function_t result_conversion_function
    ) noexcept
    {
        used = true;

        std::vector<ElapsedPair>& latency_measure 
            = stats.stat_level_0["insertWLEntry3"];

        std::vector<ElapsedPair>& latency_sub_1     = stats.stat_level_1["insertWLEntry3-1"];
        std::vector<ElapsedPair>& latency_sub_2     = stats.stat_level_1["insertWLEntry3-2"];
        std::vector<ElapsedPair>& latency_sub_3     = stats.stat_level_1["insertWLEntry3-3"];
        std::vector<ElapsedPair>& latency_sub_4     = stats.stat_level_1["insertWLEntry3-4"];

        ElapsedPair latency_0;
        ElapsedPair latency_int_1, latency_int_2, latency_int_3, latency_int_4;
        latency_0.start();

        _lock();

        Vector2 vector_template(vector_pool->getVecDataSize());
        vector_template.setVecId(write_vector.vector_id);
        vector_template.setAuxData1(write_vector.aux_data_1);
        vector_template.setAuxData2(write_vector.aux_data_2);
        
        std::memcpy(
            vector_template.getVecData(), write_vector.vector_data, vector_pool->getVecDataSize()
        );

        latency_int_1.start();

        repr_entry_cnt = eviction_strategy->getCurrSize();
        rf_write_log->insertLEntry(vector_template, vector_pool->getVecDataSize());
        if (repr_entry_cnt > 0)
            rf_write_log->increaseCEntryRf(0, repr_entry_cnt, repr_entry_cnt);

        latency_int_1.end();
        latency_sub_1.push_back(latency_int_1);

        // We run the first fast path
        latency_int_2.start();
        _updateWLEntryFastPath(write_vector, distance_function, result_conversion_function);
        latency_int_2.end();
        latency_sub_2.push_back(latency_int_2);
        
        // If need to run the SlowPath
        latency_int_3.start();
        if (rf_write_log->needRunSlowPath())
            _incrBatchUpdateWLog2(distance_function, result_conversion_function);
        latency_int_3.end();
        latency_sub_3.push_back(latency_int_3);

        latency_0.end();
        latency_measure.push_back(latency_0);

        _unlock();
    }

#pragma endregion

    void
    ResultCache2::markVecDeleted(vector_id_t vector_id) noexcept
    {
        used = true;

        std::vector<ElapsedPair>& latency_measure 
            = stats.stat_level_0["markVecDeleted"];
        ElapsedPair latency_0;
        latency_0.start();

        _lock();
        
        // In the vector pool, mark the vector as deleted.
        vector_pool->markVecInvalid(vector_id);

        latency_0.end();
        latency_measure.push_back(latency_0);

        _unlock();
    }

    void
    ResultCache2::getPooledVecs(
        std::vector<Vector2*>& pooled_list) noexcept
    {
        _lock();
        vector_pool->getPooledVecs(pooled_list);
        _unlock();
    }

    
    std::string
    ResultCache2::toString() noexcept
    {
        std::string status_string;
        size_t total_entry_count = lookup_table->map.size();
        size_t virtual_entry_count = 0;
        size_t valid_entry_count = 0;

        status_string += "------------------------------- \n";
        status_string += "-- ResultCache2 Table Entry Status:\n";

        lookup_table->map.visit_all(
            [&](const auto& pair)
            {
                // Skip the virtual entries
                if (pair.second->vector_slot_ref_list == nullptr)
                    return;

                if (pair.second->next != nullptr)
                {
                    // List all the linked vector IDs
                    size_t linked_count = 0;
                    result_cache_entry_t* next_entry = pair.second->next;
                    while (next_entry != nullptr)
                    {
                        // status_string += std::to_string(next_entry->query_vector->getVecId()) + " ";
                        linked_count++;
                        next_entry = next_entry->next;
                    }

                    virtual_entry_count += linked_count;
                }

                if (pair.second->version == -1)
                    ;
                else
                    valid_entry_count++;
            }
        );

        size_t physical_entry_count = total_entry_count - virtual_entry_count;

        status_string += "    Total Entry Count: " + std::to_string(total_entry_count) + "\n";
        status_string += "    Physical Entry Count: " + std::to_string(physical_entry_count) + "\n";
        status_string += "    Virtual Entry Count: " + std::to_string(virtual_entry_count) + "\n";
        status_string += "        Valid Entry Count: " + std::to_string(valid_entry_count) + "\n";

        // Count not null checkpoints
        size_t checkpoint_count = 0;
        lookup_table->map.visit_all(
            [&](const auto& pair)
            {
                if (pair.second->checkpoint != nullptr)
                    checkpoint_count++;
            }
        );
        status_string += "    Checkpoint Count: " + std::to_string(checkpoint_count) + "\n";

        status_string += "-- ResultCache2 Vector Pool Status:\n";
        status_string += vector_pool->toString();

        status_string += "-- ResultCache2 Write Log Status:\n\t";
        // status_string += write_log->toString();
        status_string += rf_write_log->toString();

        status_string += "\n-- ResultCache2 Statistics:\n";
        status_string += "    Cache Hit: " + std::to_string(stats.cache_hit) + "\n";
        status_string += "    Cache Miss: " + std::to_string(stats.cache_miss) + "\n";
        status_string += "    Cache Invalid Detect: " + std::to_string(stats.cache_invalid_detect) + "\n";
        status_string += "    Cache Evict: " + std::to_string(stats.cache_evict) + "\n";
        status_string += "    Cache Sim-Hit: " + std::to_string(stats.cache_sim_hit) + "\n";

        // Eviction
        status_string += "\n-- ResultCache2 Eviction Strategy Status:\n";
        status_string += "    Eviction Strategy Size: " + std::to_string(eviction_strategy->getCurrSize()) + "\n\n-- ";

        // Approx Filter
        status_string += apprx_filter->toString();

        // 
        status_string += "\n-- Vaidation Checks: These numbers should match \n";
        status_string += "    Physical Entry Count: " + std::to_string(physical_entry_count) + "\n";
        status_string += "    Eviction Strategy Size: " + std::to_string(eviction_strategy->getCurrSize()) + "\n";
        status_string += "    Representative Vectors: " + std::to_string(apprx_filter->getReprVecNum()) + "\n";
        status_string += "    Write Log Mapped Size: " + std::to_string(rf_write_log->getMappedSize()) + "\n";

        return status_string;
    }

    std::string
    ResultCache2::toStringCsv() noexcept
    {
        // Do the same, but format the string as CSV
        std::string status_string;
        size_t total_entry_count = lookup_table->map.size();
        size_t virtual_entry_count = 0;
        size_t valid_entry_count = 0;

        lookup_table->map.visit_all(
            [&](const auto& pair)
            {
                // Skip the virtual entries
                if (pair.second->vector_slot_ref_list == nullptr)
                    return;

                if (pair.second->next != nullptr)
                {
                    // List all the linked vector IDs
                    size_t linked_count = 0;
                    result_cache_entry_t* next_entry = pair.second->next;
                    while (next_entry != nullptr)
                    {
                        // status_string += std::to_string(next_entry->query_vector->getVecId()) + " ";
                        linked_count++;
                        next_entry = next_entry->next;
                    }

                    virtual_entry_count += linked_count;
                }

                if (pair.second->version == -1)
                    ;
                else
                    valid_entry_count++;
            }
        );

        size_t physical_entry_count = total_entry_count - virtual_entry_count;

        status_string += "Total Entry Count\t" + std::to_string(total_entry_count) + "\n";
        status_string += "Physical Entry Count\t" + std::to_string(physical_entry_count) + "\n";
        status_string += "Virtual Entry Count\t" + std::to_string(virtual_entry_count) + "\n";
        status_string += "Valid Entry Count\t" + std::to_string(valid_entry_count) + "\n";

        // Count not null checkpoints
        size_t checkpoint_count = 0;
        lookup_table->map.visit_all(
            [&](const auto& pair)
            {
                if (pair.second->checkpoint != nullptr)
                    checkpoint_count++;
            }
        );

        status_string += "Checkpoint Count\t" + std::to_string(checkpoint_count) + "\n";
        status_string += "Vector Pool Status\t" + vector_pool->toString() + "\n";

        status_string += "Write Log Status\t" + rf_write_log->toString() + "\n";

        status_string += "Cache Hit\t" + std::to_string(stats.cache_hit) + "\n";
        status_string += "Cache Sim-Hit\t" + std::to_string(stats.cache_sim_hit) + "\n";
        status_string += "Cache Miss\t" + std::to_string(stats.cache_miss) + "\n";
        status_string += "Cache Evict\t" + std::to_string(stats.cache_evict) + "\n";
        
        float exact_hit_ratio = 0.0f;
        exact_hit_ratio = (stats.cache_hit) / 
            static_cast<float>(stats.cache_hit + stats.cache_miss + stats.cache_sim_hit);

        status_string += "Exact Hit Ratio\t" + std::to_string(exact_hit_ratio) + "\n";
        float total_hit_ratio = 0.0f;
        total_hit_ratio = (stats.cache_hit + stats.cache_sim_hit) /
            static_cast<float>(stats.cache_hit + stats.cache_miss + stats.cache_sim_hit);

        status_string += "Total Hit Ratio\t" + std::to_string(total_hit_ratio) + "\n";

        status_string += "Eviction Strategy Size\t" + std::to_string(eviction_strategy->getCurrSize()) + "\n";

        return status_string;
    }

    void
    ResultCache2::exportPrintAll() noexcept
    {
        if (used == false)
            return;

        stats.exportAll();

        std::string directory_path = stats.directory_path;

        // Print the current status to the file
        std::string status_string = toString();
        std::string filename = directory_path + "/result-cache2-summary.txt";

        std::ofstream ofs(filename, std::ios::out);
        if (ofs)
        {
            ofs << status_string;
        }

        ofs.close();

        // Export in csv file:
        std::string csv_filename = directory_path + "/result-cache2-summary.csv";
        std::ofstream csv_ofs(csv_filename, std::ios::out);
        if (csv_ofs)
        {
            csv_ofs << toStringCsv();
        }

        csv_ofs.close();
    }

    void
    ResultCache2::resetCache() noexcept
    {
        used = true;

        _lock();

        lookup_table->map.clear();
        vector_pool->reset();
        apprx_filter->clear();

        _unlock();
    }

    // Special local tests
    void
    ResultCache2::stressTestInvalidateRandom(float percent) noexcept
    {
        if (percent < 0.0f || percent > 100.0f)
            return;

        _lock();

        // Get the number of the elements in the vector pool
        size_t pool_size = vector_pool->getPoolCurrentSize();
        size_t target_invalidate = static_cast<size_t>(pool_size * (percent));

        std::vector<Vector2*> pooled_list;
        vector_pool->getPooledVecs(pooled_list);

        std::random_shuffle(pooled_list.begin(), pooled_list.end());

        for (auto& vec : pooled_list)
        {
            if (target_invalidate == 0)
                break;

            vector_pool->markVecInvalid(vec->getVecId());
            target_invalidate--;
        }

        // Here we reset the stats.
        stats.cache_hit = 0;
        stats.cache_miss = 0;
        stats.cache_invalid_detect = 0;
        stats.cache_evict = 0;
        stats.cache_sim_hit = 0;

        _unlock();
    }

#pragma endregion
}