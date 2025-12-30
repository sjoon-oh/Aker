// 
// Author: Sukjoon Oh

#include "ApproxFilter.hh"

#include <cmath>
#include <cassert>

#include <algorithm>

namespace topkache 
{
    // 
    // --------------------------------------------------------------------------------------------
    // For HNSW version of the approximate filter

        // For approximate filter
    ApproxFilterHNSW::ApproxFilterHNSW(
        result_cache_parameter_t parameter_info,
        faiss::MetricType metric
    ) noexcept
        : filter_lock(false)
    {
        parameter = parameter_info;
        hnsw_index.reset(
            // new faiss::IndexHNSWFlat(parameter.vector_dim, 8, metric)
            new faiss::IndexHNSWFlat(parameter.vector_dim, 4, metric)
        ); // M = second parameter

        // In FAISS, the default efSearch is 16. 
        hnsw_index->hnsw.efSearch = 8;
        // hnsw_index->hnsw.efSearch = 16;
        hnsw_index->hnsw.efConstruction = 16;
        // hnsw_index->hnsw.efConstruction = 32;

        eff_repr_vec_num = 0;
    }

    void
    ApproxFilterHNSW::__lock() noexcept
    {
        while (__tryLock() == false)
            ;
    }

    bool
    ApproxFilterHNSW::__tryLock() noexcept
    {
        bool expected = false;
        bool desired = true;
        return filter_lock.compare_exchange_strong(expected, desired, std::memory_order_acquire);
    }

    void
    ApproxFilterHNSW::__unlock() noexcept
    {
        filter_lock.store(false, std::memory_order_release);
    }

    void
    ApproxFilterHNSW::addVec(float_qvec_t query) noexcept
    {
        __lock();

        // Search first
        int visited = reverse_vector_id_map.visit(query.vector_id, 
            [&](const auto& pair)
            {

            }
        );

        if (visited)
        {
            __unlock();
            return;
        }
        
        float* float_query_data = (float*)malloc(sizeof(float) * query.vector_dim);
        bool converted = query.conversion_function(
            query.vector_data, query.vector_data_size, query.vector_dim, float_query_data, query.aux
        );

        if (!converted)
        {
            free(float_query_data);
            __unlock();
            return;
        }

        bool inserted_vector_id = 0;
        bool inserted_sequence_id = 0;

        eff_repr_vec_num++;

        // Current sequence ID 
        faiss::idx_t seq_id = representative_vecs.size() / query.vector_dim;
        vector_id_map.try_emplace_or_visit(
            seq_id, query.vector_id,
            [&](const auto& pair)
            {
                // If the vector ID already exists, the sequence ID is not allocated.
                // We convey the existing sequence ID to the allocated sequence ID.
                seq_id = pair.second;
                inserted_vector_id = 1;
            }
        );

        // Add the vector to the reverse_vector_id_map
        reverse_vector_id_map.try_emplace_or_visit(
            query.vector_id, seq_id,
            [&](const auto& pair)
            {
                inserted_sequence_id = 1;
            }
        );

        assert(inserted_vector_id == inserted_sequence_id);

        if (inserted_vector_id == 1) // In case when the vector ID already exists
        {
            free(float_query_data);
            __unlock();
            return; // Do not add the vector again
        }

        // Resize the representative vectors
        representative_vecs.resize(representative_vecs.size() + query.vector_dim);
        
        // Add the vector to the HNSW index
        // Copy the vector data to the representative vectors in bulk
        std::uint32_t index = seq_id * query.vector_dim;
        std::memcpy(
            representative_vecs.data() + index,
            float_query_data,
            sizeof(float) * query.vector_dim
        );

        free(float_query_data);
        hnsw_index->add(1, representative_vecs.data() + (seq_id * query.vector_dim));
        
        __unlock();
    }


    int
    ApproxFilterHNSW::deleteVecs(std::vector<vector_id_t>& vector_id_list) noexcept
    {
        __lock();

        int deleted = 0;

        for (auto vector_id : vector_id_list)
        {
            faiss::idx_t sequence_id = -1;

            reverse_vector_id_map.visit(vector_id,
                [&](const auto& pair)
                {
                    sequence_id = pair.second;
                }
            );

            if (sequence_id == -1)
                continue;

            reverse_vector_id_map.erase(vector_id);

            int visited = vector_id_map.visit(sequence_id,
                [&](const auto& pair)
                {
                    vector_id = pair.second;
                }
            );

            if (visited == 0)
                continue;

            eff_repr_vec_num--;

            vector_id_map.erase(sequence_id);
            std::uint32_t index = sequence_id * parameter.vector_dim;

            std::memset(
                representative_vecs.data() + index,
                0,
                sizeof(float) * parameter.vector_dim
            );

            deleted++;
        }

        if (eff_repr_vec_num == 0)
        {
            representative_vecs.clear();    // Clear the representative vectors
            
            vector_id_map.clear();              // Clear the vector ID map
            reverse_vector_id_map.clear();      // Clear the reverse vector ID map
    
            hnsw_index->reset();
        }

        __unlock();

        return deleted;
    }

    void
    ApproxFilterHNSW::searchSimVecs(
        const float* x,
        faiss::idx_t k,
        float* distances,
        faiss::idx_t* labels) noexcept
    {
        __lock();
        hnsw_index->search(1, x, k, distances, labels);
        __unlock();
    }

    bool
    ApproxFilterHNSW::seqToVecId(faiss::idx_t seq_id, vector_id_t& vector_id) noexcept
    {
        __lock();

        bool found = false;
        vector_id_map.visit(seq_id,
            [&](const auto& pair)
            {
                vector_id = pair.second;
                found = true;
            }
        );

        __unlock();

        return found;
    }

    /*
     * Warning! Call of compressIndex() is not thread-safe.
     */
    void
    ApproxFilterHNSW::compressIndex() noexcept
    {
        // Remove deleted vectors, to compress the index.
        __lock();

        std::vector<float> compressed_representative_vecs;

        vector_id_map.visit_all(
            [&](const auto& pair)
            {
                faiss::idx_t seq_id = pair.first;
                vector_id_t vector_id = pair.second;

                std::uint32_t index = seq_id * parameter.vector_dim;
                // for (int i = 0; i < parameter.vector_dim; i++)
                //     compressed_representative_vecs.push_back(representative_vecs[index + i]);

                // Move in bulk
                compressed_representative_vecs.insert(
                    compressed_representative_vecs.end(),
                    representative_vecs.begin() + index,
                    representative_vecs.begin() + index + parameter.vector_dim
                );
            }
        );

        printf("ApproxFilterHNSW::compressIndex() representative_vecs size: %zu\n", 
                representative_vecs.size());


        // Clear the representative vectors, and move the compressed vectors.
        representative_vecs.clear();
        representative_vecs = std::move(compressed_representative_vecs);

        eff_repr_vec_num = representative_vecs.size() / parameter.vector_dim;

        printf("ApproxFilterHNSW::compressIndex() compressed representative_vecs size: %zu\n", 
                representative_vecs.size());

        __unlock();
    }

    void
    ApproxFilterHNSW::clear() noexcept
    {
        __lock();

        representative_vecs.clear();        // Clear the representative vectors
        
        vector_id_map.clear();              // Clear the vector ID map
        reverse_vector_id_map.clear();      // Clear the reverse vector ID map

        hnsw_index->reset();

        eff_repr_vec_num = 0;

        __unlock();
    }

    std::string
    ApproxFilterHNSW::toString() noexcept
    {
        std::string status_string;
        size_t element_count = 0;
        size_t reverse_element_count = 0;

        status_string += "ApproxFilterHNSW Element Status:\n";
        vector_id_map.visit_all(
            [&](const auto& pair)
            {
                element_count++;
                // status_string += "    Sequence ID: " + std::to_string(pair.first) + " -> Vector ID: " + std::to_string(pair.second) + "\n";
            }
        );

        reverse_vector_id_map.visit_all(
            [&](const auto& pair)
            {
                reverse_element_count++;
                // status_string += "    Vector ID: " + std::to_string(pair.first) + " -> Sequence ID: " + std::to_string(pair.second) + "\n";
            }
        );

        status_string += "    Total SEQ_ID Count: " + std::to_string(element_count) + "\n";
        status_string += "    Total Vector_ID Count: " + std::to_string(reverse_element_count) + "\n";

        status_string += "ApproxFilterHNSW Representative Array Status:\n";
        status_string += "    Size: " + std::to_string(representative_vecs.size()) + "\n";

        return status_string;
    }

    size_t
    ApproxFilterHNSW::getNumReprVecs() noexcept
    {
        return eff_repr_vec_num;
    }

    size_t
    ApproxFilterHNSW::getReprVecsArrSize() noexcept
    {
        return representative_vecs.size() / parameter.vector_dim;
    }

    // -------------------------------------------------------------------------------------------- 
    // For dual HNSW version of the approximate filter

    ApproxFilterHNSW2::ApproxFilterHNSW2(
        result_cache_parameter_t parameter_info,
        faiss::MetricType metric
    ) noexcept
        : filter_lock(false), add_count(0)
    {
        parameter = parameter_info;
        
        faiss::IndexHNSWFlat* hnsw_index = new faiss::IndexHNSWFlat(parameter.vector_dim, 4, metric);

        hnsw_index->hnsw.efSearch = 8; // In FAISS, the default efSearch is 16.
        hnsw_index->hnsw.efConstruction = 16;

        hnsw_index_wrapper.reset(new faiss::IndexIDMap(hnsw_index));
        reg_map.clear();
    }

    void
    ApproxFilterHNSW2::__lock() noexcept
    {
        while (__tryLock() == false)
            ;
    }

    void
    ApproxFilterHNSW2::__unlock() noexcept
    {
        filter_lock.store(false, std::memory_order_release);
    }

    bool
    ApproxFilterHNSW2::__tryLock() noexcept
    {
        bool expected = false;
        bool desired = true;
        return filter_lock.compare_exchange_strong(expected, desired, std::memory_order_acquire);
    }

    void
    ApproxFilterHNSW2::addVec(float_qvec_t query) noexcept
    {
        __lock();

        // Search first, if the vector ID already exists
        bool inserted = reg_map.try_emplace_or_visit(
            query.vector_id, 1,
            [&](const auto& pair)
            {
                // If the vector ID exists, the return value is true.
            }
        );

        if (!inserted)
        {
            __unlock();

            assert(0);      // This should not happen, as the vector ID should be unique.
            return;         // Do not add the vector again
        }

        float* float_query_data = (float*)malloc(sizeof(float) * query.vector_dim);
        query.conversion_function(
            query.vector_data, query.vector_data_size, query.vector_dim, float_query_data, query.aux
        );

        faiss::idx_t faiss_id = query.vector_id;
        add_count++;

        // Add the vector to the HNSW index
        // The HNSW index is wrapped with IndexIDMap2, so the vector ID
        // is used as the ID in the HNSW index.
        hnsw_index_wrapper->add_with_ids(1, float_query_data, &faiss_id);
        free(float_query_data);

        __unlock();
    }

    int
    ApproxFilterHNSW2::deleteVecs(std::vector<vector_id_t>& vector_id_list) noexcept
    {
        __lock();

        int deleted = 0;

        std::vector<faiss::idx_t> ids_to_remove;
        for (auto vector_id : vector_id_list)
        {
            // Check if the vector ID exists in the reg_map
            int visited = reg_map.visit(vector_id,
                [&](const auto& pair)
                {
                    // If the vector ID exists, the return value is true.
                }
            );

            if (visited == 0)
                continue;   // The vector ID does not exist, so skip

            reg_map.erase(vector_id);

            ids_to_remove.push_back(vector_id);
            deleted++;
        }
        
        if (!ids_to_remove.empty())
        {
            // IndexHNSWFlat does not support the "remove_ids" method directly,
            // so we skip the removal.
            // 
            // faiss::IDSelectorBatch selector(ids_to_remove.size(), ids_to_remove.data());
            // hnsw_index_wrapper->remove_ids(selector);
        }

        if (reg_map.size() == 0)
        {
            // If the reg_map is empty, clear the HNSW index
            hnsw_index_wrapper->reset();
        }

        __unlock();
        return deleted;
    }

    void
    ApproxFilterHNSW2::searchSimVecs(
        const float* x,
        faiss::idx_t k, 
        float* distances,
        faiss::idx_t* labels) noexcept
    {
        __lock();
        hnsw_index_wrapper->search(1, x, k, distances, labels);
        __unlock();
    }

    bool
    ApproxFilterHNSW2::isRegistered(vector_id_t vector_id) noexcept
    {
        __lock();
        bool registered = false;

        reg_map.visit(vector_id,
            [&](const auto& pair)
            {
                registered = true;  // If the vector ID exists, it is registered
            }
        );

        __unlock();
        return registered;
    }

    size_t
    ApproxFilterHNSW2::getReprVecNum() noexcept
    {
        __lock();
        size_t num_repr_vecs = reg_map.size();
        __unlock();
        return num_repr_vecs;
    }

    size_t
    ApproxFilterHNSW2::getAddedCounts() noexcept
    {
        __lock();
        size_t count = add_count;
        __unlock();
        return count;
    }

    void
    ApproxFilterHNSW2::clear() noexcept
    {
        __lock();

        reg_map.clear();                // Clear the reg_map
        hnsw_index_wrapper->reset();    // Clear the HNSW index

        add_count = 0;                  // Reset the add count

        __unlock();
    }

    std::string
    ApproxFilterHNSW2::toString() noexcept
    {
        std::string status_string;
        size_t element_count = 0;

        status_string += "ApproxFilterHNSW2 Element Status:\n";
        reg_map.visit_all(
            [&](const auto& pair)
            {
                element_count++;
                // status_string += "    Vector ID: " + std::to_string(pair.first) + "\n";
            }
        );

        status_string += "    Total Vector_ID Count: " + std::to_string(element_count) + "\n";

        return status_string;
    }


    // --------------------------------------------------------------------------------------------
    // Dual Fitler
    ApproxFilterDualHNSW::ApproxFilterDualHNSW(
        result_cache_parameter_t parameter_info,
        faiss::MetricType metric
    ) noexcept {
        filter_lock.store(false);

        filter_list = (ApproxFilterHNSW**)malloc(sizeof(ApproxFilterHNSW*) * num_filters);
        for (size_t i = 0; i < num_filters; i++)
        {
            filter_list[i] = new ApproxFilterHNSW(parameter_info, metric);
        }

        parameter = parameter_info;
    }

    ApproxFilterDualHNSW::~ApproxFilterDualHNSW() noexcept
    {
        for (size_t i = 0; i < num_filters; i++)
        {
            delete filter_list[i];
        }
        free(filter_list);
    }

    void
    ApproxFilterDualHNSW::__lock() noexcept
    {
        while (__tryLock() == false)
            ;
    }

    bool
    ApproxFilterDualHNSW::__tryLock() noexcept
    {
        bool expected = false;
        bool desired = true;
        return filter_lock.compare_exchange_strong(expected, desired, std::memory_order_acquire);
    }

    void
    ApproxFilterDualHNSW::__unlock() noexcept
    {
        filter_lock.store(false, std::memory_order_release);
    }

    void
    ApproxFilterDualHNSW::__switchFilter() noexcept
    {
        // Switch the filter
        ApproxFilterHNSW* temp = filter_list[0];
        filter_list[0] = filter_list[1];
        filter_list[1] = temp;
    }

    void
    ApproxFilterDualHNSW::addVec(float_qvec_t query) noexcept
    {
        __lock();

        // 
        // We always add the vectors to the first filter.
        //  The second filter is used for searching.
        filter_list[0]->addVec(query);

        __unlock();
    }

    int
    ApproxFilterDualHNSW::deleteVecs(std::vector<vector_id_t>& vector_id_list) noexcept
    {
        __lock();

        // Delete the vectors from both filters.
        for (size_t i = 0; i < num_filters; i++)
        {
            filter_list[i]->deleteVecs(vector_id_list);
        }

        __unlock();
    }

    void
    ApproxFilterDualHNSW::searchSimVecs(
        const float* x,
        faiss::idx_t k,
        float* distances,
        faiss::idx_t* labels,
        int* filter_id
    ) noexcept
    {
        __lock();

        const size_t search_num = k * 2;
    
        float*       distance_local = static_cast<float*>(malloc(sizeof(float) * search_num));
        float*       d0             = distance_local;
        float*       d1             = distance_local + k;
    
        faiss::idx_t* labels_local  = static_cast<faiss::idx_t*>(malloc(sizeof(faiss::idx_t) * search_num));
        faiss::idx_t* l0            = labels_local;
        faiss::idx_t* l1            = labels_local + k;
    
        filter_list[0]->searchSimVecs(x, k, d0, l0);
        filter_list[1]->searchSimVecs(x, k, d1, l1);
    
        // Merge the results
        //  We need to sort the results, and assign the filter ID.

        int index_0 = 0, index_1 = 0;
        for (size_t i = 0; i < search_num; i++)
        {
            if (d0[index_0] < d1[index_1])
            {
                distances[i] = d0[index_0];
                labels[i] = l0[index_0];
                filter_id[i] = 0;
                index_0++;
            }
            else
            {
                distances[i] = d1[index_1];
                labels[i] = l1[index_1];
                filter_id[i] = 1;
                index_1++;
            }   
        }
    
        free(distance_local);   // free the *original* malloc pointers
        free(labels_local);
    
        __unlock();
    }

    bool
    ApproxFilterDualHNSW::seqToVecId(faiss::idx_t seq_id, int filter_id, vector_id_t& vector_id) noexcept
    {
        __lock();

        bool found = filter_list[filter_id]->seqToVecId(seq_id, vector_id);

        __unlock();

        return found;
    }

    void
    ApproxFilterDualHNSW::clear() noexcept
    {
        __lock();

        filter_list[1]->clear();
        __switchFilter();           // After clearing the second filter, we switch the filter.

        __unlock();
    }

    std::string
    ApproxFilterDualHNSW::toString() noexcept
    {
        std::string status_string;
        size_t element_count = 0;
        size_t reverse_element_count = 0;

        status_string += "ApproxFilterDualHNSW Element Status:\n";
        for (size_t i = 0; i < num_filters; i++)
        {
            status_string += "Filter ID: " + std::to_string(i) + "\n";
            status_string += filter_list[i]->toString();
        }

        return status_string;
    }

    size_t
    ApproxFilterDualHNSW::getNumReprVecs() noexcept
    {
        size_t effective_repr_vec_num = filter_list[0]->getNumReprVecs() + filter_list[1]->getNumReprVecs();
        return effective_repr_vec_num;
    }

    size_t
    ApproxFilterDualHNSW::getReprVecsArrSize() noexcept
    {
        size_t effective_repr_vecs_arr_size = filter_list[0]->getReprVecsArrSize() + filter_list[1]->getReprVecsArrSize();
        return effective_repr_vecs_arr_size;
    }

    bool
    ApproxFilterDualHNSW::needSwitch() noexcept
    {
        size_t per_filter_entry_count = parameter.vector_pool_size / (parameter.vector_list_size * num_filters);

        size_t valid_entry_count = filter_list[0]->getNumReprVecs();
        size_t total_entry_count = filter_list[0]->getReprVecsArrSize();

        if (total_entry_count < per_filter_entry_count)
            return false;

        return ((valid_entry_count * 2) < total_entry_count);
    }


    // Second typed Dual Filter
    ApproxFilterDualHNSW2::ApproxFilterDualHNSW2(
        result_cache_parameter_t parameter_info
    ) noexcept {
        filter_lock.store(false);

        // Get the parameter
        faiss::MetricType metric_type;
        switch (parameter_info.distance_type)
        {
            case distance_type_t::DISTANCE_TYPE_L2:
                metric_type = faiss::METRIC_L2;
                break;
            case distance_type_t::DISTANCE_TYPE_IP:
                metric_type = faiss::METRIC_INNER_PRODUCT;
                break;
            default:
                assert(false); // Unsupported distance type
        }

        filter_list = (ApproxFilterHNSW2**)malloc(sizeof(ApproxFilterHNSW2*) * num_filters);
        for (size_t i = 0; i < num_filters; i++)
        {
            filter_list[i] = new ApproxFilterHNSW2(parameter_info, metric_type);
        }

        parameter = parameter_info;
    }

    ApproxFilterDualHNSW2::~ApproxFilterDualHNSW2() noexcept
    {
        for (size_t i = 0; i < num_filters; i++)
        {
            delete filter_list[i];
        }
        free(filter_list);
    }

    void
    ApproxFilterDualHNSW2::__lock() noexcept
    {
        while (__tryLock() == false)
            ;
    }

    bool
    ApproxFilterDualHNSW2::__tryLock() noexcept
    {
        bool expected = false;
        bool desired = true;
        return filter_lock.compare_exchange_strong(expected, desired, std::memory_order_acquire);
    }

    void
    ApproxFilterDualHNSW2::__unlock() noexcept
    {
        filter_lock.store(false, std::memory_order_release);
    }

    void
    ApproxFilterDualHNSW2::__switchFilter() noexcept
    {
        // Switch the filter
        ApproxFilterHNSW2* temp = filter_list[0];
        filter_list[0] = filter_list[1];
        filter_list[1] = temp;  
    }

    void
    ApproxFilterDualHNSW2::addVec(float_qvec_t query) noexcept
    {
        __lock();
        // We always add the vectors to the first filter.
        //  The second filter is used for searching
        filter_list[0]->addVec(query);
        __unlock();
    }

    int
    ApproxFilterDualHNSW2::deleteVecs(std::vector<vector_id_t>& vector_id_list) noexcept
    {
        __lock();

        int deleted = 0;

        // Delete the vectors from both filters.
        for (size_t i = 0; i < num_filters; i++)
            deleted = filter_list[i]->deleteVecs(vector_id_list);

        __unlock();
        return deleted;
    }

    void
    ApproxFilterDualHNSW2::searchSimVecs(
        const float* x,
        faiss::idx_t k,
        float* distances,
        faiss::idx_t* labels
    ) noexcept
    {
        __lock();

        const size_t search_num = k * 2;

        std::vector<float> found_distances(search_num);
        std::vector<faiss::idx_t> found_labels(search_num);

        float*       distance_local = found_distances.data();
        float*       d0             = distance_local;
        float*       d1             = distance_local + k;

        faiss::idx_t* labels_local  = found_labels.data();
        faiss::idx_t* l0            = labels_local;
        faiss::idx_t* l1            = labels_local + k;

        filter_list[0]->searchSimVecs(x, k, d0, l0);
        filter_list[1]->searchSimVecs(x, k, d1, l1);

        bool negate_distances = false;
        switch (parameter.distance_type)
        {
            case distance_type_t::DISTANCE_TYPE_L2:
                // No need to negate the distances for L2
                break;
            case distance_type_t::DISTANCE_TYPE_IP:
                // If inner product, we need to negate the distances
                negate_distances = true;
                break;
            default:
                assert(false); // Unsupported distance type
        }

        if (negate_distances)
        {
            for (size_t i = 0; i < search_num; i++)
                distance_local[i] = -distance_local[i];
        }

        // Sort the results by the distances, and its corresponding labels in order
        std::vector<std::pair<float, faiss::idx_t>> results;
        results.reserve(search_num);

        for (int i = 0; i < search_num; i++) 
        {
            // Check if the label is valid
            float found_distance = INVALID_DISTANCE;
            
            // If index range is under k, it came from the first filter
            // Check if the label is valid
            bool is_valid_label = (i < k) ? 
                filter_list[0]->isRegistered(labels_local[i]) : 
                filter_list[1]->isRegistered(labels_local[i]);

            if (is_valid_label)
                found_distance = distance_local[i];

            results.emplace_back(distance_local[i], labels_local[i]);
        }

        std::sort(results.begin(), results.end(),
        [](auto &a, auto &b) {
            return a.first < b.first;
        });

        // Fill the output arrays with the sorted results
        for (size_t i = 0; i < search_num; i++)
        {
            distances[i]    = results[i].first;
            labels[i]       = results[i].second;
        }

        __unlock();
    }

    void
    ApproxFilterDualHNSW2::clear() noexcept
    {
        __lock();

        filter_list[1]->clear();
        __switchFilter();           // After clearing the second filter, we switch the filter.

        __unlock();
    } 

    size_t
    ApproxFilterDualHNSW2::getReprVecNum() noexcept
    {
        size_t effective_repr_vec_num = filter_list[0]->getReprVecNum() + filter_list[1]->getReprVecNum();
        return effective_repr_vec_num;
    }

    size_t
    ApproxFilterDualHNSW2::getAddedCounts() noexcept
    {
        return (filter_list[0]->getAddedCounts() + filter_list[1]->getAddedCounts());
    }

    std::string
    ApproxFilterDualHNSW2::toString() noexcept
    {
        std::string status_string;
        size_t element_count = 0;
        size_t reverse_element_count = 0;

        status_string += "ApproxFilterDualHNSW Element Status:\n";
        for (size_t i = 0; i < num_filters; i++)
        {
            status_string += "Filter ID: " + std::to_string(i) + "\n";
            status_string += filter_list[i]->toString();
        }

        return status_string;
    }

    bool
    ApproxFilterDualHNSW2::needSwitch() noexcept
    {
        size_t per_filter_entry_count = parameter.vector_pool_size / (parameter.vector_list_size * num_filters);
        size_t repr_entries = filter_list[0]->getReprVecNum();
        size_t curr_added = filter_list[0]->getAddedCounts();

        if (repr_entries < per_filter_entry_count)
            return false;

        return ((repr_entries * 2) < curr_added);
    }

}