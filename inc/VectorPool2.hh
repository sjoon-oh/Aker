// 
// Author: Sukjoon Oh

#include <cstdlib>
#include <cstring>
#include <cassert>

#include <cstdio>
#include <atomic>
#include <vector>

#include <boost/unordered/concurrent_flat_map.hpp>

#ifndef TOPKACHE_VECTORPOOL2_H
#define TOPKACHE_VECTORPOOL2_H

#include "Vector2.hh"
// #include "VectorPool.hh"


namespace topkache
{
    class VectorPool2
    {
    private:
        const size_t                        vector_pool_capacity;           // Number of elements in the pool
        std::atomic<size_t>                 vector_pool_current_size;

        const size_t                        vector_data_size;

        boost::unordered::concurrent_flat_map<
            vector_id_t, Vector2*>          pool_map;

        std::atomic_flag                    pool_lock = ATOMIC_FLAG_INIT;

        void                                __lock() noexcept;
        void                                __unlock() noexcept;

        bool                                __deallocVec(vector_id_t vector_id) noexcept;
        Vector2*                            __allocVec(vector_id_t vector_id, vector_data_t* 
                                                vector_data, size_t vector_data_size) noexcept;

        bool                                __registerVec(vector_id_t vector_id, Vector2* vector) noexcept;

    public:
        VectorPool2(size_t vector_pool_size, size_t free_size) noexcept;

        virtual ~VectorPool2() noexcept;

        Vector2*                            allocateVec(vector_id_t vector_id, vector_data_t* vector_data, 
                                                size_t vector_data_size) noexcept;
        bool                                deallocateVec(vector_id_t vector_id) noexcept;
        Vector2*                            substituteVec(vector_id_t delete_vector_id, vector_id_t alloc_vector_id, 
                                                vector_data_t* alloc_vector_data, size_t vector_data_size) noexcept;
        bool                                increaseVecRefCount(vector_id_t vector_id) noexcept;

        bool                                markVecInvalid(vector_id_t vector_id) noexcept;

        Vector2*                            getVec(vector_id_t vector_id) noexcept;
        void                                releaseVec(Vector2* vector) noexcept;

        size_t                              getPoolCapacity() noexcept;
        size_t                              getPoolCurrentSize() noexcept;
        size_t                              getVecDataSize() noexcept;

        // These functions are used for testing
        void                                getPooledVecs(
                                                std::vector<Vector2*>& pooled_list) noexcept;

        void                                reset() noexcept;

        std::string                         toString() noexcept;
    };
}

#endif