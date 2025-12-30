// 
// Author: Sukjoon Oh

#ifndef TOPKACHE_VECTOR2_H
#define TOPKACHE_VECTOR2_H

#include <cstdlib>
#include <cstring>
#include <cassert>

#include <cstdio>
#include <atomic>

#include <boost/unordered/concurrent_flat_map.hpp>


namespace topkache
{
    typedef std::uint64_t                   vector_id_t;
    typedef std::uint32_t                   vector_version_t;
    typedef std::uint8_t                    vector_state_t;
    typedef std::uint8_t                    vector_data_t;

    typedef std::uint64_t                   aux_data_t;

    enum
    {
        VECTOR_STATE_VALID                  = 0x00,
        VECTOR_STATE_INVALID                = 0x01
    };

    typedef std::uint32_t                   vector_type_t;

    enum
    {
        VECTOR_TYPE_FREE                    = 0x00,    // Free vector
        VECTOR_TYPE_FLOAT                   = 0x01,
        VECTOR_TYPE_INT8                    = 0x02,
        VECTOR_TYPE_UINT8                   = 0x03
    };

    typedef struct
    {
        vector_version_t                    vector_version;
        vector_id_t                         vector_id;
        std::uint32_t                       entry_reference_count;

    } vector_header_t;

    class Vector2
    {
    private:
        vector_header_t                     vector_header;
        vector_data_t*                      vector_data;
        aux_data_t                          aux_data_1;
        aux_data_t                          aux_data_2;

        std::atomic<bool>                   vector_lock;
        std::uint8_t                        vector_state;

        // Add
        float                               distance;

        void                                __lock() noexcept;
        bool                                __tryLock() noexcept;
        void                                __unlock() noexcept;
        
    public:

        Vector2() noexcept;
        Vector2(size_t vector_data_size) noexcept;

        virtual ~Vector2() noexcept = default;

        void                                lock() noexcept;
        bool                                tryLock() noexcept;
        void                                unlock() noexcept;

        void                                makeValid() noexcept;
        void                                makeInvalid() noexcept;

        bool                                isValid() noexcept;

        vector_id_t                         getVecId() const noexcept;
        void                                setVecId(vector_id_t vector_id) noexcept;

        vector_version_t                    getVecVersion() const noexcept;
        void                                setVecVersion(vector_version_t vector_version) noexcept;
        void                                increaseVectorVersion() noexcept;

        std::uint32_t                       getRefCount() const noexcept;
        void                                resetRefCount() noexcept;
        void                                increaseRefCount() noexcept;
        void                                decreaseRefCount() noexcept;

        vector_data_t*                      getVecData() noexcept;
        void                                setVecData(vector_data_t* vector_data) noexcept;

        aux_data_t                          getAuxData1() noexcept;
        void                                setAuxData1(aux_data_t aux_data) noexcept;

        aux_data_t                          getAuxData2() noexcept;
        void                                setAuxData2(aux_data_t aux_data) noexcept;

        float                               getDistance() const noexcept;
        void                                setDistance(float distance) noexcept;

        void                                freeVecData() noexcept;

        std::string                         toString() noexcept;

    };

}

#endif