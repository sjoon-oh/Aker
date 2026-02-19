#pragma once

#include "utils/ak_spin_mutex.hh"

namespace aker
{
    /**
     * @brief A no-op mutex used when upper layers provide full synchronization.
     *
     * This is useful for stripping redundant internal locks while keeping the
     * lock/unlock call sites intact and readable.
     */
    class NullMutex
    {
    public:
        NullMutex() noexcept = default;
        NullMutex(const NullMutex&) = delete;
        NullMutex& operator=(const NullMutex&) = delete;

        void lock() noexcept {}
        bool tryLock() noexcept { return true; }
        void unlock() noexcept {}
    };

    /**
     * @brief Controls whether internal module locks are enabled.
     *
     * When a global cache lock is used, internal locks can be disabled to reduce
     * overhead and simplify concurrency reasoning.
     */
    #ifndef AKER_ENABLE_INTERNAL_LOCKS
    #define AKER_ENABLE_INTERNAL_LOCKS 0
    #endif

    #if AKER_ENABLE_INTERNAL_LOCKS
    using InternalMutex = SpinMutex;
    #else
    using InternalMutex = NullMutex;
    #endif
}
