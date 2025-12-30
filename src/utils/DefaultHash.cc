#include <cstdint>
#include <cstddef>

#include "utils/DefaultHash.hh"
#include "xxHash/xxh3.h"

namespace topkache
{
    std::uint64_t
    default_hash(const void* data, size_t size) noexcept
    {
        std::uint64_t hash = XXH3_64bits(data, size);
        return hash;
    }
}