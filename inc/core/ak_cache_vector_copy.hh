// 
// Author: Sukjoon Oh
//
// Helper routines for copying VectorSlot payloads.
// These helpers intentionally copy only the basic fields (id/version + raw data)
// to preserve the existing ANNSCache behaviour.

#ifndef AKER_CACHE_VECTOR_COPY_HH
#define AKER_CACHE_VECTOR_COPY_HH

#include <cstddef>
#include <cstring>

#include "ak_vector_slot.hh"

namespace aker
{
    inline void
    copyVectorBasic(VectorSlot* dst_vector, const VectorSlot* src_vector, size_t vector_data_size) noexcept
    {
        /* Copies only {id, version, raw bytes}.
         * Distance / aux metadata are intentionally not copied.
         */
        dst_vector->setVectorVersion(src_vector->getVectorVersion());
        dst_vector->setVectorId(src_vector->getVectorId());
        std::memcpy(dst_vector->getVectorData(), src_vector->getVectorData(), vector_data_size);
    }

    inline VectorSlot*
    cloneVectorBasic(const VectorSlot* src_vector, size_t vector_data_size) noexcept
    {
        /* Allocates and clones a VectorSlot using copyVectorBasic().
         */
        VectorSlot* dst_vector = new VectorSlot(vector_data_size);
        copyVectorBasic(dst_vector, src_vector, vector_data_size);
        return dst_vector;
    }
}

#endif
