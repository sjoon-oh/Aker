#include <cassert>
#include <cstdlib>
#include <cstring>

#include "core/ak_ann_cache2_entry_store.hh"
#include "core/ak_cache_vector_copy.hh"

namespace aker
{
    ANNCache2EntryStore::ANNCache2EntryStore(ANNCache2Context* context) noexcept
        : context_(context)
    {
        assert(context_ != nullptr);
    }

    result_cache_entry_t*
    ANNCache2EntryStore::getCEntry(vector_id_t vector_id) noexcept
    {
        /* Resolves the entry ID to the root entry.
         * The root entry is defined as the head of the linked chain (prev == nullptr).
         */
        result_cache_entry_t* entry = nullptr;

        context_->lookup_table->map.visit(
            vector_id,
            [&](const auto& pair)
            {
                entry = pair.second;
                while (entry->prev != nullptr)
                    entry = entry->prev;
            });

        return entry;
    }

    result_cache_entry_t*
    ANNCache2EntryStore::copyCacheEntry(result_cache_entry_t* entry) noexcept
    {
        /* Creates a deep copy of a cache entry for external consumption.
         * The returned copy owns its query vector and slot VectorSlot objects.
         */
        if (entry == nullptr)
            return nullptr;

        result_cache_entry_t* new_entry = new result_cache_entry_t();

        const size_t vec_data_size = context_->vector_pool->getPayloadSize();
        new_entry->query_vector = cloneVectorBasic(entry->query_vector, vec_data_size);

        new_entry->entry_kind = RESULT_CACHE_ENTRY_KIND_RETURNED_COPY;
        new_entry->entry_status = entry->entry_status;
        new_entry->version = entry->version;

        new_entry->vector_list_size = entry->vector_list_size;

        if (entry->vector_slot_ref_list != nullptr && entry->vector_list_size > 0)
        {
            new_entry->vector_slot_ref_list = static_cast<VectorSlot**>(
                aligned_alloc(k_cache_entry_slot_alignment, sizeof(VectorSlot*) * entry->vector_list_size));

            for (size_t i = 0; i < entry->vector_list_size; i++)
                new_entry->vector_slot_ref_list[i] = cloneVectorBasic(entry->vector_slot_ref_list[i], vec_data_size);
        }
        else
        {
            new_entry->vector_slot_ref_list = nullptr;
        }

        new_entry->min_distance = entry->min_distance;
        new_entry->max_distance = entry->max_distance;
        new_entry->thresh = entry->thresh;

        new_entry->prev = nullptr;
        new_entry->next = nullptr;
        new_entry->risk_factor = entry->risk_factor;
        new_entry->checkpoint = nullptr;

        return new_entry;
    }

    bool
    ANNCache2EntryStore::tryLockCEntry(result_cache_entry_t* entry) noexcept
    {
        /* CAS-based entry lock using entry_status.
         */
        result_cache_entry_status_t expected_state = RESULT_CACHE_ENTRY_STATUS_VALID;
        result_cache_entry_status_t desired_state = RESULT_CACHE_ENTRY_STATUS_INMOD;

        return __atomic_compare_exchange_n(
            &(entry->entry_status),
            &expected_state,
            desired_state,
            false,
            __ATOMIC_ACQUIRE,
            __ATOMIC_ACQUIRE);
    }

    bool
    ANNCache2EntryStore::unlockCEntry(result_cache_entry_t* entry) noexcept
    {
        /* CAS-based entry unlock.
         */
        if (entry == nullptr)
            return false;

        result_cache_entry_status_t expected_state = RESULT_CACHE_ENTRY_STATUS_INMOD;
        result_cache_entry_status_t desired_state = RESULT_CACHE_ENTRY_STATUS_VALID;

        return __atomic_compare_exchange_n(
            &(entry->entry_status),
            &expected_state,
            desired_state,
            false,
            __ATOMIC_RELEASE,
            __ATOMIC_RELEASE);
    }

    bool
    ANNCache2EntryStore::linkCEntryLocked(result_cache_entry_t* allocated_entry, vector_id_t found_id) noexcept
    {
        /* Links an allocated dummy entry to an existing root entry.
         * This preserves the original semantics.
         */
        result_cache_entry_t* root_entry = getCEntry(found_id);
        if (root_entry == nullptr)
            return false;

        int inserted = context_->lookup_table->map.try_emplace_or_visit(
            allocated_entry->query_vector->getVectorId(),
            allocated_entry,
            [&](const auto& pair)
            {
                (void)pair;
            });

        if (inserted == 0)
            return false;

        allocated_entry->prev = root_entry;
        allocated_entry->next = root_entry->next;

        root_entry->next = allocated_entry;

        return true;
    }
}
