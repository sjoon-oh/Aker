// 
// Author: Sukjoon Oh
//
// Refactor note:
// - Replace a single global lock with sharded (striped) spin locks for better concurrency.
// - Remove unused/unsafe APIs and provide a clearer pool API.
// - Keep the original pool semantics: reference-counted VectorSlot instances keyed by vector_id.
//

#ifndef AKER_VECTOR_SLOT_POOL_H
#define AKER_VECTOR_SLOT_POOL_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "ak_vector_slot.hh"
#include "utils/ak_spin_mutex.hh"

namespace aker
{
    class VectorSlotPool
    {
    private:
        /* A shard owns a subset of IDs protected by a dedicated spin lock.
         */
        struct Shard
        {
            SpinMutex                                   shard_lock_;
            std::unordered_map<vector_id_t, VectorSlot*>    vector_map_;
        };

        /* Pool configuration and counters.
         */
        const size_t                                    pool_capacity_;
        std::atomic<size_t>                             pool_size_;
        const size_t                                    payload_size_;

        /* Sharding configuration.
         */
        const size_t                                    shard_count_;
        const size_t                                    shard_mask_;
        std::vector<Shard>                              shards_;

        /* Internal helpers.
         */
        size_t                                          getShardIndex(vector_id_t vector_id) const noexcept;
        Shard&                                          getShardByIndex(size_t shard_index) noexcept;
        const Shard&                                    getShardByIndex(size_t shard_index) const noexcept;

        VectorSlot*                                        acquireOrCreateVectorLocked(
                                                            Shard& shard,
                                                            vector_id_t vector_id,
                                                            const vector_data_t* vector_data) noexcept;

        bool                                            releaseVectorReferenceLocked(
                                                            Shard& shard,
                                                            vector_id_t vector_id,
                                                            VectorSlot** deleted_vector) noexcept;

    public:
        static constexpr size_t                         kDefaultShardCount = 1024;

        VectorSlotPool(size_t pool_capacity, size_t payload_size,
                   size_t shard_count = kDefaultShardCount) noexcept;

        virtual ~VectorSlotPool() noexcept;

        /* Main pool API.
         * acquireOrCreateVector() increases refcount on hit; creates a new VectorSlot on miss.
         * releaseVectorReference() decreases refcount and deletes when it reaches zero.
         */
        VectorSlot*                                        acquireOrCreateVector(
                                                            vector_id_t vector_id,
                                                            const vector_data_t* vector_data) noexcept;

        bool                                            releaseVectorReference(vector_id_t vector_id) noexcept;

        VectorSlot*                                        replaceVectorReference(
                                                            vector_id_t delete_vector_id,
                                                            vector_id_t alloc_vector_id,
                                                            const vector_data_t* alloc_vector_data) noexcept;

        bool                                            invalidateVector(vector_id_t vector_id) noexcept;

        /* Accessors.
         */
        size_t                                          getCapacity() const noexcept;
        size_t                                          getSize() const noexcept;
        size_t                                          getPayloadSize() const noexcept;
        size_t                                          getShardCount() const noexcept;

        /* Test/debug helpers.
         * collectPooledVectors() and clear() are expected to be called by a single thread.
         */
        void                                            collectPooledVectors(
                                                            std::vector<VectorSlot*>& pooled_list) noexcept;

        void                                            clear() noexcept;

        /**
         * @brief Returns a human-readable status string.
         */
        std::string                                     getStatusText() noexcept;
    };
}

#endif
