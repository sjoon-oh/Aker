#include <algorithm>
#include <array>
#include <cassert>
#include <limits>
#include <vector>

#include "core/ak_anns_cache_similarity_engine.hh"
#include "core/ak_anns_cache_entry_store.hh"
#include "ak_logger.hh"
#include "utils/ak_malloc_ptr.hh"

namespace aker
{
    ANNSCacheSimilarityEngine::ANNSCacheSimilarityEngine(
        ANNSCacheContext* context,
        ANNSCacheEntryStore* entry_store) noexcept
        : context_(context),
          entry_store_(entry_store)
    {
        assert(context_ != nullptr);
        assert(entry_store_ != nullptr);
    }

    bool
    ANNSCacheSimilarityEngine::validateCacheEntryLocked(anns_cache_entry_t* entry) noexcept
    {
        /* Validates an entry by pushing invalid vectors to the tail.
         * If the number of valid vectors falls below intopk, the entry becomes invalid.
         */
        assert(entry != nullptr);

        if (entry->version < 0)
        {
            context_->stats.cache_miss++;
            context_->stats.cache_invalid_detect++;
            return false;
        }

        std::vector<VectorSlot*> valid_list;
        std::vector<VectorSlot*> invalid_list;

        for (int i = 0; i < entry->vector_list_size; i++)
        {
            if (entry->vector_slot_ref_list[i] == nullptr)
                invalid_list.push_back(nullptr);
            else if (entry->vector_slot_ref_list[i]->isValid())
                valid_list.push_back(entry->vector_slot_ref_list[i]);
            else
                invalid_list.push_back(entry->vector_slot_ref_list[i]);
        }

        bool is_valid = true;
        if (valid_list.size() < context_->parameter.capacity.vector_in_topk)
        {
            entry->version = -1;
            context_->stats.cache_miss++;
            context_->stats.cache_invalid_detect++;

            std::vector<vector_id_t> invalid_vecs{entry->query_vector->getVectorId()};
            context_->apprx_filter->deleteVectors(invalid_vecs);

            is_valid = false;
        }

        if (!invalid_list.empty())
        {
            for (int i = 0; i < entry->vector_list_size; i++)
            {
                if (i < static_cast<int>(valid_list.size()))
                    entry->vector_slot_ref_list[i] = valid_list[i];
                else
                    entry->vector_slot_ref_list[i] = invalid_list[i - static_cast<int>(valid_list.size())];
            }
        }

        return is_valid;
    }

    anns_cache_entry_t*
    ANNSCacheSimilarityEngine::getCacheEntryLocked(
        vector_view_t query_vector_data,
        bool& similar_entry,
        bool& is_invalid,
        distance_function_t distance_function,
        const float* query_vector_float) noexcept
    {
        /* Performs exact-hit first and then sim-hit using the approximate filter.
         * The returned entry is always a deep copy (caller-owned) on hit.
         */
        context_->try_read_count++;

        similar_entry = false;
        is_invalid = false;

        anns_cache_entry_t* entry = entry_store_->getCacheEntry(query_vector_data.vector_id);

        if (entry != nullptr)
        {
            bool is_valid = validateCacheEntryLocked(entry);
            if (!is_valid)
            {
                is_invalid = true;
                entry = nullptr;

                AKER_LOG_DEBUG << "[ANNSCacheSimilarity] exact hit invalidated: query_id=" << query_vector_data.vector_id;
            }
            else
            {
                bool linked_hit = (query_vector_data.vector_id != entry->query_vector->getVectorId());

                if (linked_hit)
                    context_->stats.cache_sim_hit++;
                else
                    context_->stats.cache_hit++;

                context_->stats.recordHitHistory();
                context_->eviction_strategy->recentlyAccessed(entry->query_vector->getVectorId());

                if (!linked_hit)
                {
                    entry->thresh = entry->thresh * context_->parameter.tuning.alpha_loosen;
                    if (entry->thresh > entry->min_distance)
                        entry->thresh = entry->min_distance;
                }

                AKER_LOG_DEBUG << "[ANNSCacheSimilarity] exact hit: query_id=" << query_vector_data.vector_id
                              << " root_id=" << entry->query_vector->getVectorId()
                              << " linked_hit=" << static_cast<int>(linked_hit);

                anns_cache_entry_t* copy_entry = entry_store_->copyCacheEntry(entry);
                return copy_entry;
            }
        }

        /* Approximate filter lookup.
         */
        static constexpr faiss::idx_t k_search_per_filter = 1;
        static constexpr size_t k_candidate_count = 2;

        /* Convert the query vector outside the critical section when possible.
         * The caller may provide a pre-converted float array.
         */
        const float* faiss_query_vector = query_vector_float;
        MallocPtr<float> owned_query_vector;
        if (faiss_query_vector == nullptr)
        {
            owned_query_vector = makeMallocPtr<float>(query_vector_data.vector_dim);

            bool convert_success = query_vector_data.conversion_function(
                query_vector_data.vector_data,
                query_vector_data.vector_data_size,
                query_vector_data.vector_dim,
                owned_query_vector.get(),
                query_vector_data.aux);
            (void)convert_success;

            faiss_query_vector = owned_query_vector.get();
        }

        std::array<float, k_candidate_count> distances{};
        std::array<faiss::idx_t, k_candidate_count> labels{};

        context_->apprx_filter->searchSimilarVectors(
            faiss_query_vector,
            k_search_per_filter,
            distances.data(),
            labels.data());

        bool last_candidate_invalid = false;
        for (size_t i = 0; i < k_candidate_count; i++)
        {
            if (distances[i] == k_invalid_distance)
                continue;

            if (labels[i] < 0)
                continue;

            vector_id_t vector_id = static_cast<vector_id_t>(labels[i]);
            anns_cache_entry_t* found_entry = entry_store_->getCacheEntry(vector_id);
            if (found_entry == nullptr)
                continue;

            bool is_valid = validateCacheEntryLocked(found_entry);
            if (!is_valid)
            {
                last_candidate_invalid = true;
                continue;
            }

            float thresh = 0.0f;
            if (context_->parameter.tuning.similar_match)
            {
                if (context_->parameter.tuning.use_fixed_thresh)
                    thresh = context_->parameter.tuning.fixed_thresh;
                else
                    thresh = std::min(found_entry->min_distance, found_entry->thresh);
            }

            float query_distance = distance_function(
                query_vector_data.vector_data,
                found_entry->query_vector->getVectorData(),
                query_vector_data.vector_dim);

            if (query_distance < thresh)
            {
                AKER_LOG_DEBUG << "[ANNSCacheSimilarity] approx hit: query_id=" << query_vector_data.vector_id
                              << " repr_id=" << found_entry->query_vector->getVectorId()
                              << " query_distance=" << query_distance
                              << " thresh=" << thresh;

                context_->stats.cache_sim_hit++;
                context_->stats.recordHitHistory();

                similar_entry = true;

                found_entry->thresh = found_entry->thresh * context_->parameter.tuning.alpha_tighten;
                if (found_entry->thresh > found_entry->min_distance)
                    found_entry->thresh = found_entry->min_distance;

                context_->eviction_strategy->recentlyAccessed(found_entry->query_vector->getVectorId());

                context_->stats.approx_added_counts.push_back(context_->apprx_filter->getAddedCounts());
                context_->stats.approx_representative_counts.push_back(context_->apprx_filter->getRepresentativeVectorNumber());

                return entry_store_->copyCacheEntry(found_entry);
            }
        }

        /* Miss path.
         */
        is_invalid = last_candidate_invalid;
        context_->stats.cache_miss++;
        context_->stats.recordHitHistory();

        AKER_LOG_DEBUG << "[ANNSCacheSimilarity] cache miss: query_id=" << query_vector_data.vector_id
                      << " last_candidate_invalid=" << static_cast<int>(last_candidate_invalid);

        context_->stats.approx_added_counts.push_back(context_->apprx_filter->getAddedCounts());
        context_->stats.approx_representative_counts.push_back(context_->apprx_filter->getRepresentativeVectorNumber());

        return nullptr;
    }
}
