
#include <vector>
#include <string>
#include <limits>
#include <algorithm>

#include <cstdio>
#include <cassert>

#define assertm(exp, msg) assert(((void)(msg), (exp)))

#include "core/ResultCache2.hh"

#define TESTS

// 
// For reference: https://github.com/boostorg/unordered/blob/develop/test/cfoa/interprocess_concurrency_tests.cpp

// ResultCache::
namespace topkache
{

#pragma region INTERNALS

    static inline void
    test_validity_reflist(result_cache_entry_t* entry)
    {
#ifdef TESTS
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
#endif
    }

    void
    ResultCache2::_test_existance(vector_id_t vector_id)
    {
#ifdef TESTS
        assert(_getCEntry(vector_id) != nullptr);
#endif
    }

    result_cache_entry_t*
    ResultCache2::_copy_cache_entry(result_cache_entry_t* entry) noexcept
    {
        result_cache_entry_t* new_entry = new result_cache_entry_t();

        new_entry->query_vector = new Vector2(vector_pool->getVecDataSize());
        new_entry->query_vector->setVecVersion(entry->query_vector->getVecVersion());
        new_entry->query_vector->setVecId(entry->query_vector->getVecId());

        std::memcpy(new_entry->query_vector->getVecData(), 
            entry->query_vector->getVecData(), vector_pool->getVecDataSize());

        new_entry->vector_list_size = entry->vector_list_size;
        new_entry->entry_status     = entry->entry_status;

        new_entry->vector_slot_ref_list = (Vector2**)aligned_alloc(8, sizeof(Vector2*) * entry->vector_list_size);

        for (size_t i = 0; i < entry->vector_list_size; i++)
        {
            new_entry->vector_slot_ref_list[i] = new Vector2(vector_pool->getVecDataSize());
            new_entry->vector_slot_ref_list[i]->setVecVersion(entry->vector_slot_ref_list[i]->getVecVersion());
            new_entry->vector_slot_ref_list[i]->setVecId(entry->vector_slot_ref_list[i]->getVecId());

            std::memcpy(new_entry->vector_slot_ref_list[i]->getVecData(), 
                entry->vector_slot_ref_list[i]->getVecData(), vector_pool->getVecDataSize());
        }

        new_entry->min_distance = entry->min_distance;
        new_entry->max_distance = entry->max_distance;
        new_entry->threshold    = entry->threshold;

        return new_entry;
    }

    /* Locks to block all the operations 
     */
    void
    ResultCache2::_lock() noexcept
    {
        while (_tryLock() == false)
            ;
    }

    bool
    ResultCache2::_tryLock() noexcept
    {
        bool expected = false;
        const bool desired = true;
        
        return cache_lock.compare_exchange_strong(expected, desired, std::memory_order_acquire);
    }

    void
    ResultCache2::_unlock() noexcept
    {
        cache_lock.store(false, std::memory_order_release);
    }

    static inline size_t
    writeLogDeltaBound(result_cache_parameter_t* parameter) noexcept
    {
        size_t delta_bound = 0;
        size_t estimated_entry_num = parameter->vector_pool_size / parameter->vector_list_size;

        delta_bound = parameter->vector_extras * estimated_entry_num;

        return delta_bound;
    }

    ResultCache2::ResultCache2(result_cache_parameter_t& parameter_info) noexcept
        : lookup_table(new result_cache_table2_t()),
        vector_pool(new VectorPool2(parameter_info.vector_pool_size, parameter_info.vector_data_size)),
        eviction_strategy(new EvictionStrategyFifo()), 
        apprx_filter(new ApproxFilterDualHNSW2(parameter_info)),
        rf_write_log(
            new RfWriteLog(parameter_info.vector_intopk, 64, parameter_info.risk_threshold)
        ),
        cache_lock(false)
    {
        parameter = parameter_info;

        repr_entry_cnt = 0;
        evict_entry_cnt = 0;

        try_read_cnt = 0;

        used = false;

        printf("ResultCache2 parameters:\n");
        printf("\tVector dimension: %d (parameter.vector_dim)\n", parameter.vector_dim);
        printf("\tVector pool size: %d (parameter.vector_pool_size)\n", parameter.vector_pool_size);
        printf("\tVector list size: %d (parameter.vector_list_size)\n", parameter.vector_list_size);
        printf("\tVector data size: %d (parameter.vector_data_size)\n", parameter.vector_data_size);
        printf("\tVector intopk: %d (parameter.vector_intopk)\n", parameter.vector_intopk);
        printf("\tVector extras: %d (parameter.vector_extras)\n", parameter.vector_extras);
        printf("\tSimilar match: %d (parameter.similar_match)\n", parameter.similar_match);
        printf("\tUse fixed threshold: %d (parameter.use_fixed_threshold)\n", parameter.use_fixed_threshold);
        printf("\tFixed threshold: %f (parameter.fixed_threshold)\n", parameter.fixed_threshold);
        printf("\tStart threshold: %f (parameter.start_threshold)\n", parameter.start_threshold);
        printf("\tRisk threshold: %f (parameter.risk_threshold)\n", parameter.risk_threshold);
        printf("\tAlpha tighten: %f (parameter.alpha_tighten)\n", parameter.alpha_tighten);
        printf("\tAlpha loosen: %f (parameter.alpha_loosen)\n", parameter.alpha_loosen);

        assert(parameter.vector_list_size == (parameter.vector_intopk + parameter.vector_extras));
    }

    bool
    ResultCache2::_tryLockCEntry(result_cache_entry_t* entry) noexcept
    {
        result_cache_entry_status_t expected_state  = RESULT_CACHE_ENTRY_STATUS_VALID;
        result_cache_entry_status_t desired_state   = RESULT_CACHE_ENTRY_STATUS_INMOD;

        bool success = __atomic_compare_exchange_n(
            &(entry->entry_status),
            &expected_state, desired_state, false, __ATOMIC_ACQUIRE, __ATOMIC_ACQUIRE);

        return success;
    }

    /* Check if eviction is required
     *  This function expects that the global lock is already acquired.
     */
    bool
    ResultCache2::_needEvict(size_t vector_list_size) noexcept
    {
        size_t pool_size            = vector_pool->getPoolCapacity();
        size_t current_pool_size    = vector_pool->getPoolCurrentSize();

        return ((current_pool_size + vector_list_size) > pool_size);
    }

    void
    ResultCache2::_evictVecs2(size_t to_evicts, std::vector<vector_id_t>& evicted_list) noexcept
    {
        if (to_evicts == 0)
            return;

        // Latency measurement
        std::vector<ElapsedPair>& latency_sub_1   = stats.stat_level_0["_evictVecs-1"];
        std::vector<ElapsedPair>& latency_sub_2   = stats.stat_level_0["_evictVecs-2"];
        std::vector<ElapsedPair>& latency_sub_3   = stats.stat_level_0["_evictVecs-3"];

        ElapsedPair latency_int_1, latency_int_2, latency_int_3;

        size_t eviction_size = 0;               // The number of vectors evicted, it is accumulated.
        size_t previous_eviction_size = 0;      // The number of vectors evicted in the previous iteration.

        while (eviction_size < to_evicts)
        {

            vector_id_t evict_candidate_id = 0;
            bool valid_evict_candid = eviction_strategy->nextEvictCandidate(&evict_candidate_id);   // Here, the candidate is removed from the list
            
            // If there is no candidate to evict, it is an exception.
            // This is not what we expect.
            assert(valid_evict_candid);

            result_cache_entry_t* evict_candidate_entry = _getCEntry(evict_candidate_id);

            // If the entry does not exist, this means that although the pool is full,
            // there is nothing stored in the FIFO queue.
            // This is not what we expect.
            assert(evict_candidate_entry != nullptr);

            // Evict the vectors, listed in the evict_candidate_entry
            for (size_t i = 0; i < evict_candidate_entry->vector_list_size; i++)
            {
                // VectorPool::deallocateVec returns true if the vector is successfully deleted.
                // One case the deallocateVec does not remove the vector, is when the reference count is not zero.
                // Multiple entries may be pointing to the same vector.
                // Evicting these vectors will break the mapping of the existing entries, but not yet to be evicted.
                // Future hits will break the fetches.
                // Thus, we only evict the vectors when the reference count is zero.
                // Otherwise, the deallocVector will only decrease the reference count.

                Vector2* vector = evict_candidate_entry->vector_slot_ref_list[i];
                if (vector == nullptr) 
                    continue;

                bool deleted = vector_pool->deallocateVec(vector->getVecId());

                if (deleted)
                    eviction_size++;
            }

            // 
            // If the entry has the children (semantically mapped entries), we need to erase them too.
            // The last element is nullptr, thus we repeat until it meets nullptr.
            // The children does not have the Vector2 list. Thus, we do not do deallocateVec.
            result_cache_entry_t* child_entry = evict_candidate_entry->next;
            while (child_entry != nullptr)
            {
                vector_id_t child_vector_id = child_entry->query_vector->getVecId();
                result_cache_entry_t* next = child_entry->next;

                lookup_table->map.erase(child_vector_id);
                delete child_entry;

                child_entry = next;
            }

            int unseen = 0;

            // The entry may have the checkpoint, 
            if (evict_candidate_entry->checkpoint != nullptr)
            {
                Vector2* write_log_vector = &(evict_candidate_entry->checkpoint->vector);

                unseen = rf_write_log->getDistance(evict_candidate_entry->checkpoint);

                write_log_vector->lock();
                write_log_vector->decreaseRefCount();
                write_log_vector->unlock();
            }

            rf_write_log->removeFromRr(reinterpret_cast<rev_centry_t*>(evict_candidate_entry));

            lookup_table->map.erase(evict_candidate_id);
            repr_entry_cnt = eviction_strategy->getCurrSize();

            rf_write_log->decreaseCEntryRf(
                evict_candidate_entry->risk_factor, unseen, repr_entry_cnt
            );

            evict_entry_cnt++;
            delete evict_candidate_entry;

            evicted_list.push_back(evict_candidate_id);
        }

        stats.cache_evict += eviction_size;

        // We run the safe trim only once.
        rf_write_log->safeTrims(rf_write_log->getHead());
    }
    
    /* Get the cache entry with the given vector_id
     *  This function waits until modification of a certain entry is completed.
     */
    result_cache_entry_t*
    ResultCache2::_getCEntry(
        vector_id_t vector_id) noexcept
    {
        result_cache_entry_t* entry = nullptr;

        lookup_table->map.visit(vector_id,
            [&](const auto& pair)
            {
                entry = pair.second;
                while (entry->prev != nullptr) 
                    entry = entry->prev;

                assert(entry->prev == nullptr);
            }
        );

        return entry;
    }

    void
    ResultCache2::_getSimCEntryEx(
        float_qvec_t query_vector_data,
        distance_function_t distance_function,
        std::vector<faiss::idx_t>& labels,
        std::vector<float>& distances
        ) noexcept
    {
        result_cache_entry_t* entry = nullptr;

        struct entry_distance_pair_t {
            result_cache_entry_t* entry;
            float distance;
        };

        std::vector<entry_distance_pair_t> entry_distance_pairs;

        lookup_table->map.visit_all(
            [&](const auto& pair)
            {
                result_cache_entry_t* existing_entry = pair.second;
                while (_tryLockCEntry(existing_entry))    // Wait until the entry is locked.
                    ;
                
                if (existing_entry->vector_slot_ref_list == nullptr)
                {
                    _unlockCEntry(existing_entry);
                    return;
                }

                float query_distance = distance_function(
                    existing_entry->query_vector->getVecData(),
                    query_vector_data.vector_data,
                    parameter.vector_dim
                );

                entry_distance_pairs.push_back({
                    existing_entry,
                    query_distance
                });

                _unlockCEntry(existing_entry);
            });

        // Sort the entries by distance, in ascending order
        std::sort(entry_distance_pairs.begin(), entry_distance_pairs.end(),
            [](const entry_distance_pair_t& a, const entry_distance_pair_t& b) {
                return a.distance < b.distance;
            });
        
        // If the number of entries is less than the search number, we fill the labels with the vector_ids.
        for (size_t i = 0; i < labels.size(); i++)
        {
            if (i < entry_distance_pairs.size())
            {
                result_cache_entry_t* existing_entry = entry_distance_pairs[i].entry;
                labels[i] = existing_entry->query_vector->getVecId();
            }
            else
            {
                labels[i] = -1; // Invalid label`
            }
        }
    }

    /* Try to map the semantic entry
     *  This function assumes that the global lock is already locked.
     */
    bool
    ResultCache2::_linkSimCEntryLin(
        result_cache_entry_t* allocated_entry,
        distance_function_t distance_function
        ) noexcept
    {
        bool mapped = false;
        float min_distance = allocated_entry->min_distance;

        result_cache_entry_t* map_parent_entry = nullptr;

        // Here, we filter the most similar entry by visiting all.
        lookup_table->map.visit_all(
            [&](const auto& pair)
            {
                result_cache_entry_t* existing_entry = pair.second;
                while(_tryLockCEntry(existing_entry))    // Wait until the entry is locked.
                    ;

                // Case when already mapped, dummy entry
                // Skip it.
                if (existing_entry->vector_slot_ref_list == nullptr)
                {
                    _unlockCEntry(existing_entry);
                    return; 
                }

                if (existing_entry->query_vector->getVecId() == allocated_entry->query_vector->getVecId())
                {
                    _unlockCEntry(existing_entry);
                    return;
                }

                float threshold = 0.0f;
                if (parameter.similar_match)
                {
                    if (parameter.use_fixed_threshold)
                        threshold = parameter.fixed_threshold;
    
                    else
                        threshold = std::min(existing_entry->min_distance, existing_entry->threshold);
                }

                float similarity_threshold = threshold;
                float query_distance = distance_function(
                    allocated_entry->query_vector->getVecData(),
                    existing_entry->query_vector->getVecData(),
                    parameter.vector_dim
                );
    
                if (query_distance < similarity_threshold)
                {
                    map_parent_entry = existing_entry;
                    min_distance = query_distance;
                }

                _unlockCEntry(existing_entry);
            }
        );

        if (map_parent_entry != nullptr)
        {   
            // Release the allocated entry. This is useless.
            {
                assert(allocated_entry->vector_slot_ref_list != nullptr);

                free(allocated_entry->vector_slot_ref_list);
                allocated_entry->vector_slot_ref_list = nullptr;
            }
            
            // Map the allocated entry to the parent entry.
            while(_tryLockCEntry(map_parent_entry))    // Wait until the entry is locked.
                ;

            allocated_entry->next = map_parent_entry->next;
            allocated_entry->prev = map_parent_entry;
            map_parent_entry->next = allocated_entry;

            _unlockCEntry(map_parent_entry);

            mapped = true;

            // Set max distance (float) to child so that no further mapping is done.
            allocated_entry->min_distance = std::numeric_limits<float>::max();
        }

        return mapped;
    }

    bool
    ResultCache2::_linkSimCEntryApprx(
        result_cache_entry_t* allocated_entry,
        float_qvec_t query_vector_data,
        distance_function_t distance_function
        ) noexcept
    {
        bool mapped = false;
        float min_distance = allocated_entry->min_distance;

        result_cache_entry_t* map_parent_entry = nullptr;

        const size_t search_num = 1;

        float* query_vector = (float*)malloc((sizeof(float) * parameter.vector_dim));
        float* distance = (float*)malloc(sizeof(float) * (search_num * 2));
        faiss::idx_t* labels = (faiss::idx_t*)malloc(sizeof(faiss::idx_t) * (search_num * 2));
        int* filter_ids = new int[search_num * 2];

        bool convert_success = query_vector_data.conversion_function(
            allocated_entry->query_vector->getVecData(), 
            parameter.vector_data_size, 
            parameter.vector_dim, 
            query_vector,
            query_vector_data.aux
        );
        
        /* Set the search number to 1, since we only need to find the most similar entry.
         */
        // apprx_filter->searchSimVecs(
        //     query_vector, search_num, distance, labels, filter_ids
        // );

        apprx_filter->searchSimVecs(
            query_vector, search_num, distance, labels
        );

        // Get the mapped vector_id
        for (size_t i = 0; i < search_num; i++)
        {
            vector_id_t vector_id = 0;

            if (distance[i] == INVALID_DISTANCE)
                continue;

            vector_id = static_cast<vector_id_t>(labels[i]);

            result_cache_entry_t* found_entry = _getCEntry(vector_id);
            if (found_entry == nullptr)
                continue;

            float threshold = 0.0f;
            if (parameter.similar_match)
            {
                if (parameter.use_fixed_threshold)
                    threshold = parameter.fixed_threshold;

                else
                    threshold = std::min(found_entry->min_distance, found_entry->threshold);
            }

            float similarity_threshold = threshold;
            float query_distance = std::numeric_limits<float>::max();
            
            query_distance = distance_function(
                query_vector_data.vector_data,                  // The query vector data
                found_entry->query_vector->getVecData(),     // The found entry vector data
                query_vector_data.vector_dim
            );

            if (query_distance < similarity_threshold)
            {
                map_parent_entry = found_entry;
                min_distance = query_distance;
            }
        }

        if (map_parent_entry != nullptr)
        {   
            // Release the allocated entry. This is useless.
            assert(allocated_entry->vector_slot_ref_list != nullptr);

            free(allocated_entry->vector_slot_ref_list);
            allocated_entry->vector_slot_ref_list = nullptr;
            
            // Map the allocated entry to the parent entry.
            while(_tryLockCEntry(map_parent_entry))    // Wait until the entry is locked.
                ;

            allocated_entry->next = map_parent_entry->next;
            allocated_entry->prev = map_parent_entry;
            map_parent_entry->next = allocated_entry;

            _unlockCEntry(map_parent_entry);

            mapped = true;

            // Set max distance (float) to child so that no further mapping is done.
            allocated_entry->min_distance = std::numeric_limits<float>::max();
        }

        free(query_vector);
        free(distance);
        free(labels);
        delete[] filter_ids;

        return mapped;
    }

    /* Unlock the entry
     * This function changes the status of the entry from INMOD to VALID.
     * @param entry: The entry to be unlocked.
     * @return: If the entry is successfully unlocked, true is returned. Otherwise, false is returned.
     */
    bool
    ResultCache2::_unlockCEntry(result_cache_entry_t* entry) noexcept
    {
        if (entry == nullptr)
            return false;

        result_cache_entry_status_t expected_state  = RESULT_CACHE_ENTRY_STATUS_INMOD;
        result_cache_entry_status_t desired_state   = RESULT_CACHE_ENTRY_STATUS_VALID;

        bool success = __atomic_compare_exchange_n(
            &(entry->entry_status),
            &expected_state, desired_state, false, __ATOMIC_RELEASE, __ATOMIC_RELEASE);

        return success;
    }

    ResultCache2::~ResultCache2()
    {        
        std::time_t t = std::time(nullptr);
        std::tm tm;

        localtime_r(&t, &tm);

        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");

        std::string filename = "/tmp/result-cache-stat" + oss.str() + ".txt";
        std::string export_status = toString().c_str();

        // Export to a file
        std::ofstream file(filename, std::ios::out);

        if (file.is_open())
        {
            file << export_status;
            file.close();
        }

        printf("Final state: \n%s\n", this->toString().c_str());
    }

    bool
    ResultCache2::_handleInvalidCEntry(
        result_cache_entry_t* entry) noexcept
    {
        assert(entry != nullptr);

        // If the version number is less than 0, we regard it as an invalid entry.
        // Invalid entry becomes when the entry is previously considered as "insufficient", but yet remains 
        // in the cache. 
        // This is the case when the entry is only waiting to be evicted (waiting its order for eviction).
        // We regard it as a cache miss, thus returns nullptr.
        if (entry->version < 0)
        {
            stats.cache_miss++;
            stats.cache_invalid_detect++;
            return false;
        }

        // 
        // If the entry is potentially valid, we need to check the validity of the vectors.
        // First, it checks whether there are deleted (invalid) vectors in the list.
        // If there are, we rearrange vector_slot_ref_list by moving the invalid vectors to the end of the list.
        // This is because when new vectors are inserted and taken as the candidate for result set updates, 
        // the substitution is taken from the end of the list. 

        std::vector<Vector2*> valid_list;
        std::vector<Vector2*> invalid_list;

        // 
        // We need to check the validity of the vectors.
        // If any of the vectors are invalid, we move the invalid vectors, to the end of the list.
        for (int i = 0; i < entry->vector_list_size; i++)
        {
            if (entry->vector_slot_ref_list[i] == nullptr)
                invalid_list.push_back(nullptr); 

            else if (entry->vector_slot_ref_list[i]->isValid())
                valid_list.push_back(entry->vector_slot_ref_list[i]);

            else
                invalid_list.push_back(entry->vector_slot_ref_list[i]); 
        }

        // But, before we do the substitution, we first check the number of the valid vectors.
        // The number should be at least the number of the parameter.vector_intopk, 
        // to provide the delete-tolerance at some level.

        bool is_valid = true;
        if (valid_list.size() < parameter.vector_intopk)
        {
            // If the number of the valid vectors is less than the parameter.vector_intopk, 
            // we regard it as a cache miss.
            // The entry is not valid, thus we return nullptr.
            entry->version = -1;
            stats.cache_miss++;
            stats.cache_invalid_detect++;

            // Free from the approximate filter
            std::vector<vector_id_t> invalid_vecs{entry->query_vector->getVecId()};
            apprx_filter->deleteVecs(invalid_vecs);

            is_valid = false;
        }

        // If that is not the case, then we rearrange the vector_slot_ref_list.
        if (invalid_list.size() > 0)
        {
            for (int i = 0; i < entry->vector_list_size; i++)
            {
                if (i < valid_list.size())
                    entry->vector_slot_ref_list[i] = valid_list[i];

                else
                    entry->vector_slot_ref_list[i] = invalid_list[i - valid_list.size()];
            }
        }

        return is_valid;
    }

    void
    ResultCache2::_updateWLEntryFastPath(
        float_qvec_t write_vector,
        distance_function_t distance_function,
        result_conversion_function_t result_conversion_function
    ) noexcept
    {
        size_t search_num = 1;

        float* query_vector     = (float*)malloc((sizeof(float) * parameter.vector_dim));
        float* distance         = (float*)malloc(sizeof(float) * (search_num * 2));
        faiss::idx_t* labels    = (faiss::idx_t*)malloc(sizeof(faiss::idx_t) * (search_num * 2));
        int* filter_ids         = new int[search_num * 2];

        // After insert, we search for the nearest vectors using the approximate filter.
        
        bool convert_success = write_vector.conversion_function(
            write_vector.vector_data,              // The vector_data_t to be converted
            write_vector.vector_data_size,         // The total size of the vector_data_t
            write_vector.vector_dim,               // The dimension of the vector_data_t
            query_vector,                          // The destination float*
            write_vector.aux
        );

        // 
        // First, we run the fast path
        for (int i = 0; i < search_num; i++)
        {
            apprx_filter->searchSimVecs(
                query_vector, search_num, distance, labels
            );

            if (distance[i] == INVALID_DISTANCE)
                continue;

            vector_id_t vector_id = static_cast<vector_id_t>(labels[i]);

            result_cache_entry_t* found_entry = _getCEntry(vector_id);
            if (found_entry == nullptr)
                continue;

            float query_distance = INVALID_DISTANCE;
            query_distance = distance_function(
                write_vector.vector_data,                  // The query vector data
                found_entry->query_vector->getVecData(),   // The found entry vector data
                write_vector.vector_dim
            );

            if (query_distance < found_entry->max_distance)
            {
                // Substitute here.
                for (int j = found_entry->vector_list_size - 1; j >= 0; j--)
                {
                    if (!(found_entry->vector_slot_ref_list[j]->isValid()))
                        continue;

                    vector_id_t delete_vector_id 
                        = found_entry->vector_slot_ref_list[j]->getVecId();
                    float distance = found_entry->vector_slot_ref_list[j]->getDistance();
                    
                    vector_id_t write_vector_id = write_vector.vector_id;
                    vector_data_t* write_vector_data = write_vector.vector_data;

                    if (distance > query_distance)
                    {
                        Vector2* sub_vec = vector_pool->substituteVec(
                            delete_vector_id, write_vector_id, write_vector_data, vector_pool->getVecDataSize()
                        );

                        // substituteVec returns the new vector, from which it uses the raw vector 
                        assert(sub_vec != nullptr);

                        // Here, if the result structure form is different from the raw vector form,
                        // we need to convert the vector data to the different form.
                        // For this, we add the handler.
                        std::uint64_t aux_data_1 = found_entry->query_vector->getAuxData1();
                        std::uint64_t aux_data_2 = found_entry->query_vector->getAuxData2();

                        // Here, if the result structure form is different from the raw vector form, 
                        // we need to convert the vector data to the different form.
                        // For this, we add the handler.
                        if (result_conversion_function != nullptr)
                            result_conversion_function(
                                write_vector_id,
                                sub_vec->getVecData(), 
                                vector_pool->getVecDataSize(), 
                                aux_data_1,                 // Essential
                                aux_data_2
                            );

                        found_entry->vector_slot_ref_list[j] = sub_vec;

                        if (j == found_entry->vector_list_size - 1)
                        {
                            found_entry->max_distance = query_distance;
                            found_entry->vector_slot_ref_list[j]->setDistance(query_distance);
                        }

                        break;
                    }
                }
            }
        }

        free(query_vector);
        free(distance);
        free(labels);
        delete[] filter_ids;
    }

    void
    ResultCache2::_incrBatchUpdateWLog2(
        distance_function_t distance_function,
        result_conversion_function_t result_conversion_function
    ) noexcept
    {
        std::vector<ElapsedPair>& latency_sub_1 = stats.stat_level_2["_incrBatchUpdateWLog2-1"];
        std::vector<ElapsedPair>& latency_sub_2 = stats.stat_level_2["_incrBatchUpdateWLog2-2"];
        std::vector<ElapsedPair>& latency_sub_3 = stats.stat_level_2["_incrBatchUpdateWLog2-3"];
        std::vector<ElapsedPair>& latency_sub_4 = stats.stat_level_2["_incrBatchUpdateWLog2-4"];

        ElapsedPair latency_int_1, latency_int_2, latency_int_3, latency_int_4;

        result_cache_entry_t* saved_entry = nullptr;
        
        saved_entry = reinterpret_cast<result_cache_entry_t*>(rf_write_log->getNextRr());
        if (saved_entry == nullptr)
            return;

        // For debug
        vector_id_t vector_id = saved_entry->query_vector->getVecId();

        // Find one
        bool found = false;
        lookup_table->map.visit(vector_id, [&](const auto& pair) { found = true; });

        assert(found == true);
        std::vector<close_candidates_t> found_entries;

        latency_int_1.start();
        write_log_entry_t* old_ckpt = saved_entry->checkpoint;
        write_log_entry_t* new_ckpt = rf_write_log->sweepLEntryBatch(
            saved_entry->query_vector->getVecData(), 
            parameter.vector_dim, 
            saved_entry->max_distance, 
            found_entries, 
            distance_function, 
            old_ckpt
        );

        // Sort the found entries by the distance, in descending order (largest distance first)
        std::sort(found_entries.begin(), found_entries.end(),
            [](const close_candidates_t& a, const close_candidates_t& b) {
                return a.distance > b.distance;
            });

        latency_int_1.end();
        latency_sub_1.push_back(latency_int_1);

        latency_int_2.start();

        // If nothing is found from the write log, skip updating the result set.
        if (found_entries.size() > 0)
        {
            Vector2** new_slot_ref_list = nullptr;
            new_slot_ref_list = new (Vector2*[parameter.vector_intopk]);
            std::memset(new_slot_ref_list, 0, sizeof(Vector2*) * parameter.vector_intopk);

            int new_slot_ref_index = 0;
            size_t vector_data_size = vector_pool->getVecDataSize();
            for (int k = 0; k < parameter.vector_intopk; k++)
            {
                float current_distance = saved_entry->vector_slot_ref_list[k]->getDistance();
                if (found_entries.size() > 0)
                {                
                    close_candidates_t& current_candidate = found_entries.back();
                    if (current_distance > current_candidate.distance)
                    {
                        // If the currrent distance is larger, push the new one.
                        vector_id_t delete_vector_id = saved_entry->vector_slot_ref_list[k]->getVecId();
                        
                        vector_id_t found_vector_id = current_candidate.entry->vector.getVecId();
                        vector_data_t* found_vector_data = current_candidate.entry->vector.getVecData();

                        std::uint64_t aux_data_1 = current_candidate.entry->vector.getAuxData1();
                        std::uint64_t aux_data_2 = current_candidate.entry->vector.getAuxData2();

                        Vector2* sub_vec = vector_pool->substituteVec(
                            delete_vector_id, found_vector_id, found_vector_data, vector_data_size
                        );
                        sub_vec->setDistance(current_candidate.distance);

                        assert(sub_vec != nullptr);

                        // Here, if the result structure form is different from the raw vector form, 
                        // we need to convert the vector data to the different form.
                        // For this, we add the handler.
                        if (result_conversion_function != nullptr)
                            result_conversion_function(
                                found_vector_id,
                                sub_vec->getVecData(), 
                                vector_data_size, 
                                aux_data_1,                 // Essential
                                aux_data_2
                            );

                        new_slot_ref_list[new_slot_ref_index] = sub_vec;
                        new_slot_ref_index++;

                        rf_write_log->getStatsPtr()->refresh_count++;

                        found_entries.pop_back();
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

            // Make the new reference list
            std::memcpy(
                saved_entry->vector_slot_ref_list, 
                new_slot_ref_list, 
                sizeof(Vector2*) * parameter.vector_intopk
            );
            delete[] new_slot_ref_list;
        }

        latency_int_2.end();
        latency_sub_2.push_back(latency_int_2);
        
        saved_entry->checkpoint = new_ckpt;

        std::uint64_t distance = 0;
        if (old_ckpt == nullptr)
            distance = new_ckpt->epoch;
        else
            distance = (new_ckpt->epoch - old_ckpt->epoch); 

        rf_write_log->decreaseCEntryRf(0, distance, repr_entry_cnt);
    }

#pragma endregion
}