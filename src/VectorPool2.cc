
#include <iostream>
#include <cassert>

#include "VectorPool2.hh"

namespace topkache
{
    void
    VectorPool2::__lock() noexcept
    {
        while (pool_lock.test_and_set(std::memory_order_acquire))
            ;
    }

    void
    VectorPool2::__unlock() noexcept
    {
        pool_lock.clear(std::memory_order_release);
    }

    bool
    VectorPool2::__deallocVec(vector_id_t vector_id) noexcept
    {
        bool deleted = false;
        pool_map.visit(
            vector_id, [&](const auto& pair) { 

                Vector2* target_vector = pair.second;
                target_vector->lock();
                target_vector->decreaseRefCount();

                if (target_vector->getRefCount() == 0)
                {
                    target_vector->freeVecData();
                    
                    delete target_vector; 
                    deleted = true;

                    return;
                }

                target_vector->unlock();
            }
        );

        if (deleted)
        {
            pool_map.erase(vector_id);
            vector_pool_current_size.fetch_sub(1);

            // printf("deallocated: %d\n", vector_id);
        }

        return deleted;
    }

    Vector2*
    VectorPool2::__allocVec(vector_id_t vector_id, vector_data_t* vector_data, size_t vector_data_size) noexcept
    {
        Vector2* vector = nullptr;
        pool_map.try_emplace_or_visit(
            vector_id, nullptr, 
            [&](const auto& pair) {
                vector = pair.second;

                vector->lock();
                vector->increaseRefCount();
                vector->unlock();
            }
        );


        // If the vector is found, do not make another vector but return the existing one.
        if (vector)
        {
            return vector;
        }

        vector = new Vector2(vector_data_size);
        vector_pool_current_size.fetch_add(1);

        vector->setVecId(vector_id);
        // vector->setVecData(vector_data);
        std::memcpy(vector->getVecData(), vector_data, vector_data_size);

        pool_map.visit(
            vector_id, [&](auto& pair) { 
                pair.second = vector;
            }
        );

        return vector;
    }

    bool
    VectorPool2::__registerVec(vector_id_t vector_id, Vector2* vector) noexcept
    {
        bool inserted = pool_map.try_emplace_or_visit(
            vector_id, vector, [&](const auto& pair) { }
        );

        return inserted;
    }

    VectorPool2::VectorPool2(
        size_t vector_pool_size, size_t free_size
    ) noexcept
        : vector_pool_capacity(vector_pool_size), vector_pool_current_size(0), vector_data_size(free_size)
    {

    }

    VectorPool2::~VectorPool2()
    {

    }

    Vector2* 
    VectorPool2::allocateVec(vector_id_t vector_id, vector_data_t* vector_data, size_t vector_data_size) noexcept
    {
        // Allocates new, or increases the reference count of the existing vector.
        __lock();

        // We disable size capacity, since it may break the insert (substitution)
        // if (vector_pool_current_size.load() > vector_pool_capacity)
        // {
        //     __unlock();
        //     return nullptr;
        // }

        Vector2* vector = __allocVec(vector_id, vector_data, vector_data_size);

        __unlock();
        return vector;
    }

    Vector2*
    VectorPool2::getVec(vector_id_t vector_id) noexcept
    {
        __lock();

        Vector2* target_vector = nullptr;
        pool_map.visit(
            vector_id, [&](const auto& pair) { 
                target_vector = pair.second;
                target_vector->lock();
            }
        );

        __unlock();
        return target_vector;
    }

    void
    VectorPool2::releaseVec(Vector2* vector) noexcept
    {
        __lock();
        vector->unlock();
        __unlock();
    }

    bool 
    VectorPool2::deallocateVec(vector_id_t vector_id) noexcept
    {
        __lock();

        // Deallocator does not conflict with other threads communicating with this pool,
        // but cannot manage the thread that holds the pointer of the vector.
        // If the thread is waiting for read lock, but this deallocateVec deletes the vector,
        // the thread will access the deleted vector.

        bool deleted = __deallocVec(vector_id);

        __unlock();
        return deleted;   
    }

    Vector2*
    VectorPool2::substituteVec(
        vector_id_t delete_vector_id, vector_id_t alloc_vector_id, vector_data_t* alloc_vector_data, size_t vector_data_size) noexcept
    {
        __lock();

        // The __deallocVec may decrease (physically delete) the vector, or just decrease the reference count.
        // In the latter case, the vector is not deleted, thus the total size of the pool does not change.
        // In this function, we do not care the current size of the vector pool.
        // Instead, we try to register a new Vector2 object to the pool.
        
        bool success    = __deallocVec(delete_vector_id);
        Vector2* vector = __allocVec(alloc_vector_id, alloc_vector_data, vector_data_size);
        if (vector == nullptr)
            assert(false);

        __unlock();

        return vector;
    }

    bool
    VectorPool2::markVecInvalid(vector_id_t vector_id) noexcept
    {
        __lock();

        // The __deallocVec may decrease (physically delete) the vector, or just decrease the reference count.
        // In this case, we only mark the vector as invalid, but do not delete the vector.
        // We need to track whether the external request does not want the vector to exist.
        // If we physically deallocate the vector, the cache entry that refers to the vector will break.
        // Instead, we mark the vector as invalid, and wait until the all the cache entries that refer to the vector are deleted.

        Vector2* target_vector = nullptr;
        pool_map.visit(
            vector_id, [&](const auto& pair) { 
                target_vector = pair.second;

                target_vector->lock();

                // Mark the vector as invalid
                target_vector->makeInvalid();
                target_vector->unlock();
            }
        );

        __unlock();
    
        return (target_vector != nullptr);
    }

    bool
    VectorPool2::increaseVecRefCount(vector_id_t vector_id) noexcept
    {
        __lock();

        bool success = false;
        pool_map.visit(
            vector_id, [&](const auto& pair) { 
                Vector2* target_vector = pair.second;
                target_vector->lock();
                target_vector->increaseRefCount();
                target_vector->unlock();
                success = true;
            }
        );

        __unlock();
        return success;
    }

    size_t
    VectorPool2::getPoolCapacity() noexcept
    {
        return vector_pool_capacity;
    }

    size_t
    VectorPool2::getPoolCurrentSize() noexcept
    {
        return vector_pool_current_size.load();
    }

    size_t
    VectorPool2::getVecDataSize() noexcept
    {
        return vector_data_size;
    }

    void
    VectorPool2::getPooledVecs(
        std::vector<Vector2*>& pooled_list
    ) noexcept
    {
        __lock();

        pooled_list.clear();
        pool_map.visit_all(
            [&](const auto& pair) {
                pooled_list.push_back(pair.second);
            }
        );

        __unlock();
    }

    std::string 
    VectorPool2::toString() noexcept
    {
        size_t pool_size = getPoolCurrentSize();
        size_t pool_capacity = getPoolCapacity();

        std::string status_string = "VectorPool2 Status: " + std::to_string(pool_size) \
            + " inserted out of total " + std::to_string(pool_capacity);
        
        size_t vector_element_count = 0;
        size_t valid_vector_count = 0;

        pool_map.visit_all(
            [&](const auto& pair) {
                // status_string += "    Vector ID: " + std::to_string(pair.first) + " ";
                // status_string += pair.second->toString();
                // status_string += "\n";

                vector_element_count++;
                if (pair.second->isValid() == true)
                    valid_vector_count++;
            }
        );

        status_string += ", observed elements: " + std::to_string(vector_element_count) + "\n";
        status_string += "    Valid vectors: " + std::to_string(valid_vector_count) + "/" + std::to_string(vector_element_count) + "\n";

        return status_string;
    }
}