
#ifndef TOPKACHE_COMMONS_H
#define TOPKACHE_COMMONS_H

#include <cstdint>
#include <cstddef>

namespace topkache
{
    enum class distance_type_t : std::uint8_t
    {
        DISTANCE_TYPE_L2 = 0,           // L2 distance
        DISTANCE_TYPE_IP                // Inner product
    };

    typedef struct ParameterInfo
    {
        std::uint32_t                               vector_dim;
        size_t                                      vector_pool_size;
        size_t                                      vector_list_size;
        size_t                                      vector_data_size;
        size_t                                      vector_intopk;
        size_t                                      vector_extras;
        
        // Extras
        bool                                        similar_match;
        bool                                        use_fixed_threshold;
        float                                       fixed_threshold;
        float                                       start_threshold;

        float                                       risk_threshold;
        float                                       alpha_tighten;
        float                                       alpha_loosen;

        distance_type_t                             distance_type;

    } result_cache_parameter_t;
}

#endif