#ifndef AKER_ANNS_CACHE_CONFIG_HH
#define AKER_ANNS_CACHE_CONFIG_HH

#include <cstddef>
#include <cstdint>

namespace aker
{
    /**
     * @brief Distance metric selection.
     */
    enum class distance_type_t : std::uint8_t
    {
        DISTANCE_TYPE_L2 = 0,
        DISTANCE_TYPE_IP
    };

    /**
     * @brief System-wide vector format configuration.
     */
    struct VectorFormatConfig
    {
        std::uint32_t vector_dim{0};
        size_t vector_data_size{0};
    };

    /**
     * @brief Cache capacity configuration.
     */
    struct CacheCapacityConfig
    {
        size_t slot_pool_size{0};
        size_t slot_list_size{0};
        size_t vector_in_topk{0};
        size_t vector_extras{0};
    };

    /**
     * @brief Algorithmic tuning configuration.
     */
    struct AlgorithmTuningConfig
    {
        bool similar_match{false};

        bool use_fixed_thresh{false};
        float fixed_thresh{0.0f};
        float start_thresh{0.0f};

        float risk_thresh{0.0f};
        float alpha_tighten{0.0f};
        float alpha_loosen{0.0f};
    };

    /**
     * @brief Project-wide configuration used to construct the cache.
     */
    struct ANNSCacheConfig
    {
        VectorFormatConfig vector_format;
        CacheCapacityConfig capacity;
        AlgorithmTuningConfig tuning;

        distance_type_t distance_type{distance_type_t::DISTANCE_TYPE_L2};
    };

    /**
     * @brief Backward-compatible typedef used by the cache constructors.
     */
    using anns_cache_config_t = ANNSCacheConfig;
}

#endif // AKER_ANNS_CACHE_CONFIG_HH
