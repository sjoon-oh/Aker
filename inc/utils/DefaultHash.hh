// 
// Author: Sukjoon Oh

#ifndef TOPKACHE_HASH_H
#define TOPKACHE_HASH_H

#include <cstdint>
#include <cstddef>

namespace topkache
{
    std::uint64_t default_hash(const void* data, size_t size) noexcept;
}

#endif