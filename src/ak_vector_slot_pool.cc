#include <algorithm>
#include <cassert>
#include <cstring>
#include <functional>
#include <limits>
#include <mutex>

#include "ak_vector_slot_pool.hh"

#include "ak_logger.hh"

namespace aker
{
    /* Small helpers to normalize shard count.
     * We prefer a power-of-two shard count so that shard selection is a fast bitmask.
     */
    static inline bool
    isPowerOfTwo(size_t x) noexcept
    {
        return (x != 0) && ((x & (x - 1)) == 0);
    }

    static inline size_t
    nextPowerOfTwo(size_t x) noexcept
    {
        if (x == 0)
            return 1;

        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;

        if (sizeof(size_t) >= 8)
            x |= x >> 32;

        return x + 1;
    }

    static inline size_t
    normalizeShardCount(size_t shard_count) noexcept
    {
        if (shard_count == 0)
            shard_count = VectorSlotPool::kDefaultShardCount;

        return isPowerOfTwo(shard_count) ? shard_count : nextPowerOfTwo(shard_count);
    }

    VectorSlotPool::VectorSlotPool(size_t pool_capacity, size_t payload_size, size_t shard_count) noexcept
        : pool_capacity_(pool_capacity),
          pool_size_(0),
          payload_size_(payload_size),
          shard_count_(normalizeShardCount(shard_count)),
          shard_mask_(shard_count_ - 1),
          shards_(shard_count_)
    {
        /* Pre-reserve per-shard maps to reduce rehashing on hot paths.
         */
        const size_t per_shard_reserve = (pool_capacity_ / shard_count_) + 16;
        for (auto& shard : shards_)
            shard.vector_map_.reserve(per_shard_reserve);
    }

    VectorSlotPool::~VectorSlotPool() noexcept
    {
        /* Keep destructor semantics predictable: release all managed vectors.
         */
        clear();
    }

    size_t
    VectorSlotPool::getShardIndex(vector_id_t vector_id) const noexcept
    {
        /* Hash-based shard selection.
         * shard_count_ is normalized to a power-of-two so bitmask is safe.
         */
        const size_t hash_value = std::hash<vector_id_t>{}(vector_id);
        return (hash_value & shard_mask_);
    }

    VectorSlotPool::Shard&
    VectorSlotPool::getShardByIndex(size_t shard_index) noexcept
    {
        return shards_[shard_index];
    }

    const VectorSlotPool::Shard&
    VectorSlotPool::getShardByIndex(size_t shard_index) const noexcept
    {
        return shards_[shard_index];
    }

    VectorSlot*
    VectorSlotPool::acquireOrCreateVectorLocked(
        Shard& shard,
        vector_id_t vector_id,
        const vector_data_t* vector_data) noexcept
    {
        /* Fast path: hit in shard map -> increase refcount.
         */
        auto it = shard.vector_map_.find(vector_id);
        if (it != shard.vector_map_.end())
        {
            VectorSlot* target_vector = it->second;
            target_vector->lock();
            target_vector->increaseRefCount();
            target_vector->unlock();
            return target_vector;
        }

        /* Miss path: allocate and publish a new VectorSlot.
         */
        assert(vector_data != nullptr);

        VectorSlot* new_vector = new VectorSlot(payload_size_);
        new_vector->setVectorId(vector_id);
        std::memcpy(new_vector->getVectorData(), vector_data, payload_size_);

        shard.vector_map_.emplace(vector_id, new_vector);
        pool_size_.fetch_add(1, std::memory_order_relaxed);

        AKER_LOG_DEBUG << "[VectorSlotPool] created vector slot: vector_id=" << vector_id
                      << " pool_size=" << pool_size_.load(std::memory_order_relaxed);

        return new_vector;
    }

    bool
    VectorSlotPool::releaseVectorReferenceLocked(
        Shard& shard,
        vector_id_t vector_id,
        VectorSlot** deleted_vector) noexcept
    {
        /* Decrease refcount; erase from the shard map when it reaches zero.
         * The caller is responsible for deleting *deleted_vector outside the shard lock.
         */
        *deleted_vector = nullptr;

        auto it = shard.vector_map_.find(vector_id);
        if (it == shard.vector_map_.end())
            return false;

        VectorSlot* target_vector = it->second;
        target_vector->lock();
        target_vector->decreaseRefCount();

        if (target_vector->getRefCount() == 0)
        {
            shard.vector_map_.erase(it);
            pool_size_.fetch_sub(1, std::memory_order_relaxed);

            target_vector->unlock();
            *deleted_vector = target_vector;

            AKER_LOG_DEBUG << "[VectorSlotPool] deleted vector slot: vector_id=" << vector_id
                          << " pool_size=" << pool_size_.load(std::memory_order_relaxed);
            return true;
        }

        target_vector->unlock();
        return false;
    }

    VectorSlot*
    VectorSlotPool::acquireOrCreateVector(vector_id_t vector_id, const vector_data_t* vector_data) noexcept
    {
        /* Shard-level critical section.
         */
        const size_t shard_index = getShardIndex(vector_id);
        Shard& shard = getShardByIndex(shard_index);

        std::lock_guard<SpinMutex> lock_guard(shard.shard_lock_);
        return acquireOrCreateVectorLocked(shard, vector_id, vector_data);
    }

    bool
    VectorSlotPool::releaseVectorReference(vector_id_t vector_id) noexcept
    {
        /* Shard-level critical section with deferred deletion.
         */
        const size_t shard_index = getShardIndex(vector_id);
        Shard& shard = getShardByIndex(shard_index);

        VectorSlot* deleted_vector = nullptr;
        bool deleted = false;

        {
            std::lock_guard<SpinMutex> lock_guard(shard.shard_lock_);
            deleted = releaseVectorReferenceLocked(shard, vector_id, &deleted_vector);
        }

        if (deleted && deleted_vector != nullptr)
        {
            deleted_vector->freeVectorData();
            delete deleted_vector;
        }

        return deleted;
    }

    VectorSlot*
    VectorSlotPool::replaceVectorReference(
        vector_id_t delete_vector_id,
        vector_id_t alloc_vector_id,
        const vector_data_t* alloc_vector_data) noexcept
    {
        /* Replace operation touches two IDs; lock both shards in a total order.
         * This preserves the original "single critical section" semantics for involved IDs
         * while allowing unrelated shards to progress concurrently.
         */
        const size_t delete_shard_index = getShardIndex(delete_vector_id);
        const size_t alloc_shard_index  = getShardIndex(alloc_vector_id);

        VectorSlot* deleted_vector = nullptr;
        VectorSlot* allocated_vector = nullptr;
        bool deleted = false;

        if (delete_shard_index == alloc_shard_index)
        {
            Shard& shard = getShardByIndex(delete_shard_index);
            std::lock_guard<SpinMutex> lock_guard(shard.shard_lock_);

            deleted = releaseVectorReferenceLocked(shard, delete_vector_id, &deleted_vector);
            allocated_vector = acquireOrCreateVectorLocked(shard, alloc_vector_id, alloc_vector_data);
        }
        else
        {
            const size_t first_index  = std::min(delete_shard_index, alloc_shard_index);
            const size_t second_index = std::max(delete_shard_index, alloc_shard_index);

            Shard& first_shard  = getShardByIndex(first_index);
            Shard& second_shard = getShardByIndex(second_index);

            std::lock_guard<SpinMutex> first_guard(first_shard.shard_lock_);
            std::lock_guard<SpinMutex> second_guard(second_shard.shard_lock_);

            Shard& delete_shard = (delete_shard_index == first_index) ? first_shard : second_shard;
            Shard& alloc_shard  = (alloc_shard_index  == first_index) ? first_shard : second_shard;

            deleted = releaseVectorReferenceLocked(delete_shard, delete_vector_id, &deleted_vector);
            allocated_vector = acquireOrCreateVectorLocked(alloc_shard, alloc_vector_id, alloc_vector_data);
        }

        if (deleted && deleted_vector != nullptr)
        {
            deleted_vector->freeVectorData();
            delete deleted_vector;
        }

        return allocated_vector;
    }

    bool
    VectorSlotPool::invalidateVector(vector_id_t vector_id) noexcept
    {
        /* Mark a vector invalid under shard lock.
         */
        const size_t shard_index = getShardIndex(vector_id);
        Shard& shard = getShardByIndex(shard_index);

        std::lock_guard<SpinMutex> lock_guard(shard.shard_lock_);

        auto it = shard.vector_map_.find(vector_id);
        if (it == shard.vector_map_.end())
            return false;

        VectorSlot* target_vector = it->second;
        target_vector->lock();
        target_vector->makeInvalid();
        target_vector->unlock();

        return true;
    }

    size_t
    VectorSlotPool::getCapacity() const noexcept
    {
        return pool_capacity_;
    }

    size_t
    VectorSlotPool::getSize() const noexcept
    {
        return pool_size_.load(std::memory_order_relaxed);
    }

    size_t
    VectorSlotPool::getPayloadSize() const noexcept
    {
        return payload_size_;
    }

    size_t
    VectorSlotPool::getShardCount() const noexcept
    {
        return shard_count_;
    }

    void
    VectorSlotPool::collectPooledVectors(std::vector<VectorSlot*>& pooled_list) noexcept
    {
        /* Collect a snapshot of all pooled vectors.
         * This helper is intended for tests, and it locks all shards to avoid a partial view.
         */
        pooled_list.clear();

        std::vector<std::unique_lock<SpinMutex>> locks;
        locks.reserve(shard_count_);

        for (size_t i = 0; i < shard_count_; ++i)
            locks.emplace_back(getShardByIndex(i).shard_lock_);

        for (size_t i = 0; i < shard_count_; ++i)
        {
            Shard& shard = getShardByIndex(i);
            for (const auto& kv : shard.vector_map_)
                pooled_list.push_back(kv.second);
        }
    }

    void
    VectorSlotPool::clear() noexcept
    {
        /* Clear the pool by locking all shards and deleting all vectors.
         * This is primarily used for tests and destructor cleanup.
         */
        std::vector<std::unique_lock<SpinMutex>> locks;
        locks.reserve(shard_count_);

        for (size_t i = 0; i < shard_count_; ++i)
            locks.emplace_back(getShardByIndex(i).shard_lock_);

        for (size_t i = 0; i < shard_count_; ++i)
        {
            Shard& shard = getShardByIndex(i);
            for (auto& kv : shard.vector_map_)
            {
                VectorSlot* vector = kv.second;
                vector->freeVectorData();
                delete vector;
            }
            shard.vector_map_.clear();
        }

        pool_size_.store(0, std::memory_order_relaxed);
    }

    std::string
    VectorSlotPool::getStatusText() noexcept
    {
        /* Human-readable status snapshot.
         * This is a debugging helper, so it locks shards sequentially.
         */
        const size_t pool_size = getSize();
        const size_t pool_capacity = getCapacity();

        size_t observed_elements = 0;
        size_t valid_count = 0;

        for (size_t i = 0; i < shard_count_; ++i)
        {
            Shard& shard = getShardByIndex(i);
            std::lock_guard<SpinMutex> lock_guard(shard.shard_lock_);

            for (const auto& kv : shard.vector_map_)
            {
                ++observed_elements;
                if (kv.second->isValid())
                    ++valid_count;
            }
        }

        std::string status_string = "VectorSlotPool Status: " + std::to_string(pool_size)
            + " inserted out of total " + std::to_string(pool_capacity);

        status_string += ", observed elements: " + std::to_string(observed_elements) + "\n";
        status_string += "    Valid vectors: " + std::to_string(valid_count) + "/" + std::to_string(observed_elements) + "\n";

        return status_string;
    }
}
