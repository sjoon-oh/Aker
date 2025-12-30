#include <limits>
#include <cassert>

#include "Vector2.hh"

namespace topkache
{
    Vector2::Vector2() noexcept
    {
        vector_header.vector_version        = 0;
        vector_header.vector_id             = 0;
        vector_header.entry_reference_count = 1;

        vector_data                         = nullptr;
        aux_data_1                          = 0;
        aux_data_2                          = 0;
        distance                            = std::numeric_limits<float>::max();

        makeValid();

        vector_lock.store(false);
    }

    Vector2::Vector2(size_t vector_data_size) noexcept
    {
        vector_header.vector_version        = 0;
        vector_header.vector_id             = 0;
        vector_header.entry_reference_count = 1;

        vector_data                         = (vector_data_t*)malloc(vector_data_size);
        aux_data_1                          = 0;
        aux_data_2                          = 0;
        distance                            = std::numeric_limits<float>::max();

        makeValid();

        vector_lock.store(false);
    }

    void
    Vector2::__lock() noexcept
    {
        while (__tryLock() == false)
            ;
    }

    bool
    Vector2::__tryLock() noexcept
    {
        bool expected = false;
        bool desired = true;
        return vector_lock.compare_exchange_strong(expected, desired, std::memory_order_acquire);
    }

    void
    Vector2::__unlock() noexcept
    {
        vector_lock.store(false, std::memory_order_release);
    }

    void
    Vector2::lock() noexcept
    {
        __lock();
    }

    bool
    Vector2::tryLock() noexcept
    {
        return __tryLock();
    }

    void
    Vector2::unlock() noexcept
    {
        __unlock();
    }

    void
    Vector2::makeValid() noexcept
    {
        vector_state = VECTOR_STATE_VALID;
    }

    void
    Vector2::makeInvalid() noexcept
    {
        vector_state = VECTOR_STATE_INVALID;
    }

    bool
    Vector2::isValid() noexcept
    {
        return (vector_state == VECTOR_STATE_VALID);
    }

    vector_id_t 
    Vector2::getVecId() const noexcept
    {
        return vector_header.vector_id;
    }

    void
    Vector2::setVecId(vector_id_t vector_id) noexcept
    {
        vector_header.vector_id = vector_id;
    }

    vector_version_t
    Vector2::getVecVersion() const noexcept
    {
        return vector_header.vector_version;
    }

    void
    Vector2::setVecVersion(vector_version_t vector_version) noexcept
    {
        vector_header.vector_version = vector_version;
    }

    void
    Vector2::resetRefCount() noexcept
    {
        vector_header.entry_reference_count = 1;
    }
    
    std::uint32_t
    Vector2::getRefCount() const noexcept
    {
        return vector_header.entry_reference_count;
    }

    void
    Vector2::increaseVectorVersion() noexcept
    {
        vector_header.vector_version++;
    }

    void
    Vector2::increaseRefCount() noexcept
    {
        vector_header.entry_reference_count++;
    }

    void
    Vector2::decreaseRefCount() noexcept
    {
        vector_header.entry_reference_count--;
        assert(vector_header.entry_reference_count < 1000000);
    }

    vector_data_t*
    Vector2::getVecData() noexcept
    {
        return vector_data;
    }

    void
    Vector2::setVecData(vector_data_t* vector_data) noexcept
    {
        this->vector_data = vector_data;
    }

    aux_data_t
    Vector2::getAuxData1() noexcept
    {
        return aux_data_1;
    }

    void
    Vector2::setAuxData1(aux_data_t aux_data) noexcept
    {
        this->aux_data_1 = aux_data;
    }

    aux_data_t
    Vector2::getAuxData2() noexcept
    {
        return aux_data_2;
    }

    void
    Vector2::setAuxData2(aux_data_t aux_data) noexcept
    {
        this->aux_data_2 = aux_data;
    }

    float
    Vector2::getDistance() const noexcept
    {
        return distance;
    }

    void
    Vector2::setDistance(float distance) noexcept
    {
        this->distance = distance;
    }

    void
    Vector2::freeVecData() noexcept
    {
        free(vector_data);
    }

    std::string
    Vector2::toString() noexcept
    {
        char buffer[128];
        sprintf(buffer, "VectorSlot[ID: %lu, Version: %u, Ref: %u, Data: %p]", 
            vector_header.vector_id, vector_header.vector_version, vector_header.entry_reference_count, 
            vector_data);

        return std::string(buffer);
    }
}
