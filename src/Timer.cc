#include "Timer.hh"

namespace topkache {

    ElapsedPair::ElapsedPair() noexcept
        : elapsed_ms(0.0), elapsed_ns(0.0)
    {
        aux_1 = 0;
        aux_2 = 0;
    }

    void
    ElapsedPair::start() noexcept
    {
        clock_gettime(CLOCK_MONOTONIC, &start_time);
    }

    void 
    ElapsedPair::end() noexcept
    {
        clock_gettime(CLOCK_MONOTONIC, &end_time);
    }

    void
    ElapsedPair::elapsedMs() noexcept
    {
        elapsed_ms = (end_time.tv_nsec + end_time.tv_sec * 1000000000UL 
            - start_time.tv_nsec - start_time.tv_sec * 1000000000UL) / 1000000.0;
    }

    void
    ElapsedPair::elapsedNs() noexcept
    {
        elapsed_ns = (end_time.tv_nsec + end_time.tv_sec * 1000000000UL 
            - start_time.tv_nsec - start_time.tv_sec * 1000000000UL);
    }

    void
    ElapsedPair::setAux1(std::uint64_t aux) noexcept
    {
        aux_1 = aux;
    }

    void
    ElapsedPair::setAux2(std::uint64_t aux) noexcept
    {
        aux_2 = aux;
    }

    std::uint64_t
    ElapsedPair::getAux1() const noexcept
    {
        return aux_1;
    }

    std::uint64_t
    ElapsedPair::getAux2() const noexcept
    {
        return aux_2;
    }

    double
    ElapsedPair::getElapsedMs() const noexcept
    {
        return elapsed_ms;
    }

    double
    ElapsedPair::getElapsedNs() const noexcept
    {
        return elapsed_ns;
    }
}