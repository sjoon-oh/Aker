#include "ResultCache2CWrapper.h"
#include "core/ResultCache2.hh"
#include "utils/ParamParser.hh"
#include "utils/DistanceFuncs.hh"

#include "xxHash/xxh3.h"

#include <cassert>

// Pointer dance
static inline topkache::ResultCache2* as_cache(char* p) { 
    return reinterpret_cast<topkache::ResultCache2*>(p); 
}
static inline char* as_char(topkache::ResultCache2* c) {
    return reinterpret_cast<char*>(c); 
}
static inline topkache::Vector2* as_vec(char* p) { 
    return reinterpret_cast<topkache::Vector2*>(p); 
}
static inline char* as_char(topkache::Vector2* v) { 
    return reinterpret_cast<char*>(v); 
}
static inline topkache::float_qvec_t* as_fvec(char* p) { 
    return reinterpret_cast<topkache::float_qvec_t*>(p); 
}
static inline char* as_char(topkache::float_qvec_t* v) { 
    return reinterpret_cast<char*>(v); 
}
static inline topkache::result_cache_entry_t* as_entry(char* e) { 
    return reinterpret_cast<topkache::result_cache_entry_t*>(e); 
}
static inline char* as_char(topkache::result_cache_entry_t* e) { 
    return reinterpret_cast<char*>(e); 
}

void
import_topkache_parameter(char* path, result_cache_parameter_c_t* parameter)
{
    std::string file_path(path);

    topkache::ParameterParser parser(file_path);
    topkache::result_cache_parameter_t parameter_info = parser.getParameter();

    parameter->vector_dim               = parameter_info.vector_dim;
    parameter->vector_pool_size         = parameter_info.vector_pool_size;
    parameter->vector_list_size         = parameter_info.vector_list_size;
    parameter->vector_data_size         = parameter_info.vector_data_size;
    parameter->vector_intopk            = parameter_info.vector_intopk;
    parameter->vector_extras            = parameter_info.vector_extras;
    parameter->similar_match            = parameter_info.similar_match;
    parameter->use_fixed_threshold      = parameter_info.use_fixed_threshold;
    parameter->fixed_threshold          = parameter_info.fixed_threshold;
    parameter->start_threshold          = parameter_info.start_threshold;
    parameter->risk_threshold           = parameter_info.risk_threshold;
    parameter->alpha_tighten            = parameter_info.alpha_tighten;
    parameter->alpha_loosen             = parameter_info.alpha_loosen;

}

bool
conversion_function_c_wrapper(
    void* src,
    size_t src_size,
    size_t dim,
    void* dst,
    uint8_t* aux) {
    
    // Dummy

    return true;
}

result_cache_2_c_wrapper_t* 
create_result_cache_2_c_wrapper(
    result_cache_parameter_c_t parameter
    )
{
    
    topkache::result_cache_parameter_t parameter_info;

    parameter_info.vector_dim           = parameter.vector_dim;
    parameter_info.vector_pool_size     = parameter.vector_pool_size;
    parameter_info.vector_list_size     = parameter.vector_list_size;
    parameter_info.vector_data_size     = parameter.vector_data_size;
    parameter_info.vector_intopk        = parameter.vector_intopk;
    parameter_info.vector_extras        = parameter.vector_extras;

    parameter_info.similar_match        = parameter.similar_match;
    parameter_info.use_fixed_threshold  = parameter.use_fixed_threshold;
    parameter_info.fixed_threshold      = parameter.fixed_threshold;
    parameter_info.start_threshold      = parameter.start_threshold;

    parameter_info.risk_threshold       = parameter.risk_threshold;
    parameter_info.alpha_tighten        = parameter.alpha_tighten;
    parameter_info.alpha_loosen         = parameter.alpha_loosen;
    parameter_info.distance_type        = static_cast<topkache::distance_type_t>(parameter.distance_type);

    result_cache_2_c_wrapper_t* wrapper = new result_cache_2_c_wrapper_t;
    wrapper->result_cache = as_char(new topkache::ResultCache2(parameter_info));

    wrapper->parameter.vector_dim           = parameter_info.vector_dim;
    wrapper->parameter.vector_pool_size     = parameter_info.vector_pool_size;
    wrapper->parameter.vector_list_size     = parameter_info.vector_list_size;
    wrapper->parameter.vector_data_size     = parameter_info.vector_data_size;
    wrapper->parameter.vector_intopk        = parameter_info.vector_intopk;
    wrapper->parameter.vector_extras        = parameter_info.vector_extras;
    
    wrapper->parameter.similar_match        = parameter_info.similar_match;
    wrapper->parameter.use_fixed_threshold  = parameter_info.use_fixed_threshold;
    wrapper->parameter.fixed_threshold      = parameter_info.fixed_threshold;
    wrapper->parameter.start_threshold      = parameter_info.start_threshold;

    wrapper->parameter.risk_threshold       = parameter_info.risk_threshold;
    wrapper->parameter.alpha_tighten        = parameter_info.alpha_tighten;
    wrapper->parameter.alpha_loosen         = parameter_info.alpha_loosen;
    wrapper->parameter.distance_type        = static_cast<uint8_t>(parameter_info.distance_type);

    return wrapper;
}

void
destroy_result_cache_2_c_wrapper(
    result_cache_2_c_wrapper_t* wrapper
    )
{
    if (wrapper != nullptr)
    {
        delete as_cache(wrapper->result_cache);
        delete wrapper;
    }
}

char*
create_vector_2_c_wrapper(
    uint64_t vector_id,
    size_t vector_size,
    char* vector_data,
    uint64_t aux_data_1,
    uint64_t aux_data_2,
    float distance
    )
{
    topkache::Vector2* vector = new topkache::Vector2(vector_size);

    vector->setVecId(vector_id);
    vector->setAuxData1(aux_data_1);
    vector->setAuxData2(aux_data_2);
    
    if (vector_data != nullptr)
    {
        std::memcpy(vector->getVecData(), vector_data, vector_size);
    }
    vector->setDistance(distance);

    return as_char(vector);
}

uint64_t
get_vid_entry_2_c_wrapper(
    char* searched_entry
    )
{
    topkache::result_cache_entry_t* entry = as_entry(searched_entry);
    return entry->query_vector->getVecId();
}

uint64_t
get_vid_vector_2_c_wrapper(
    char* vector_2_wrapper
    )
{
    topkache::Vector2* vector = as_vec(vector_2_wrapper);
    return vector->getVecId();
}

uint64_t
get_vid_float_vector_2_c_wrapper(
    char* float_vector_2_wrapper
)
{
    topkache::float_qvec_t* float_vector = as_fvec(float_vector_2_wrapper);
    return float_vector->vector_id;
}

char*
get_data_vector_2_c_wrapper(
    char* vector_2_wrapper
    )
{
    topkache::Vector2* vector = as_vec(vector_2_wrapper);
    return reinterpret_cast<char*>(vector->getVecData());
}

void
set_distance_vector_2_c_wrapper(
    char* vector_2_wrapper,
    float distance
    )
{
    topkache::Vector2* vector = as_vec(vector_2_wrapper);
    vector->setDistance(distance);
}

void
destroy_vector_2_c_wrapper(
    char* vector_2_wrapper
    )
{
    if (vector_2_wrapper != nullptr)
    {
        delete as_vec(vector_2_wrapper);
    }
}

char*
create_float_vector_2_c_wrapper(
    char* vector_2_wrapper, size_t dim, size_t vector_data_size,
    bool (*conversion_function)(
        void*, size_t, size_t, void*, uint8_t*
    )
    )
{
    topkache::Vector2* vector = as_vec(vector_2_wrapper);
    topkache::float_qvec_t* float_vector = new topkache::float_qvec_t;

    float_vector->vector_id         = vector->getVecId();
    float_vector->vector_data       = vector->getVecData();
    float_vector->vector_dim        = dim;
    float_vector->vector_data_size  = vector_data_size;
    float_vector->conversion_function = conversion_function;

    float_vector->aux_data_1        = vector->getAuxData1();
    float_vector->aux_data_2        = vector->getAuxData2();

    return reinterpret_cast<char*>(float_vector);
}

void
destroy_float_vector_2_c_wrapper(
    char* float_vector_2_wrapper
    )
{
    if (float_vector_2_wrapper != nullptr)
    {
        delete as_fvec(float_vector_2_wrapper);
    }
}

char*
make_cache_entry_c_wrapper(
    result_cache_2_c_wrapper_t* wrapper,
    char* query_vector,
    size_t vector_list_size,
    char** vector_slot_ref_list
    )
{
    topkache::ResultCache2* result_cache = as_cache(wrapper->result_cache);

    topkache::result_cache_entry_t* entry =
        result_cache->makeCEntry(
            as_vec(query_vector),
            vector_list_size,
            reinterpret_cast<topkache::Vector2**>(vector_slot_ref_list)
        );
    
    return as_char(entry);
}

char*
sim_search_c_wrapper(
    result_cache_2_c_wrapper_t* wrapper,
    char* float_query_vector,
    bool* similar_entry,
    bool* is_invalid,
    float (*distance_function)(uint8_t*, uint8_t*, size_t)
    )
{
    topkache::result_cache_entry_t* entry = nullptr;
    
    if (float_query_vector != nullptr)
    {
        topkache::float_qvec_t* query_vector_data = as_fvec(float_query_vector);
        topkache::ResultCache2* result_cache = as_cache(wrapper->result_cache);

        topkache::result_cache_entry_t* found_entry =
            result_cache->simGetCEntry(
                *query_vector_data,
                *similar_entry,
                *is_invalid,
                distance_function
            );

        return reinterpret_cast<char*>(found_entry);
    }

    return NULL;
}

char*
get_result_sets_c_wrapper(char* searched_entry)
{
    if (searched_entry != nullptr)
    {
        topkache::result_cache_entry_t* entry = as_entry(searched_entry);
        return reinterpret_cast<char*>(entry->vector_slot_ref_list);
    }

    return NULL;
}

void
debug_print_cache_entry_c_wrapper(char* entry, char* status_string)
{
    if (entry != nullptr)
    {
        topkache::result_cache_entry_t* cache_entry = as_entry(entry);
        
        std::string status = "";
        status += "{";
        status += "Entry ID: " + std::to_string(cache_entry->query_vector->getVecId()) + ", ";
        status += "Entry Size: " + std::to_string(cache_entry->vector_list_size) + ", ";
        status += "Entry Threshold: " + std::to_string(cache_entry->threshold) + ", ";
        status += "Entry Min Distance: " + std::to_string(cache_entry->min_distance) + ", ";
        status += "Entry Max Distance: " + std::to_string(cache_entry->max_distance) + ", ";
        status += "Entry Risk Factor: " + std::to_string(cache_entry->risk_factor) + ", ";
        
        status += "List: [";
        for (int i = 0; i < cache_entry->vector_list_size; i++)
        {
            if (i > 0)
                status += ", ";
            status += "{";
            topkache::Vector2* element = cache_entry->vector_slot_ref_list[i];

            status += "VID: ";

            if (element == nullptr)
            {
                status += "NULL, ";
                continue;
            }

            if (!element->isValid())
            {
                status += "INVALID, ";
                continue;
            }

            status += std::to_string(element->getVecId()) + ", ";
            status += "}";
        }
        status += "]}";

        std::strcpy(status_string, status.c_str());
    }
}

char*
get_result_c_wrapper(char* searched_entry, int index)
{
    if (searched_entry != nullptr)
    {
        topkache::result_cache_entry_t* entry = as_entry(searched_entry);
        return reinterpret_cast<char*>(entry->vector_slot_ref_list[index]);
    }

    return NULL;
}

bool
insert_cache_entry_c_wrapper(
        result_cache_2_c_wrapper_t* wrapper,
        uint64_t vector_id,
        char* new_cache_entry,
        char* float_query_vector)
{
    topkache::float_qvec_t* query_vector_data = as_fvec(float_query_vector);
    topkache::ResultCache2* result_cache = as_cache(wrapper->result_cache);

    return result_cache->insertCEntry2(
        vector_id,
        as_entry(new_cache_entry),
        *query_vector_data
    );
}

bool
link_cache_entry_c_wrapper(
    result_cache_2_c_wrapper_t* wrapper,
    char* new_cache_entry,
    uint64_t found_id)
{
    topkache::ResultCache2* result_cache = as_cache(wrapper->result_cache);

    if (as_entry(new_cache_entry)->vector_slot_ref_list == NULL)
        as_entry(new_cache_entry)->vector_slot_ref_list = nullptr; 

    return result_cache->linkCEntry(
        as_entry(new_cache_entry),
        found_id
    );
}

void
insert_wl_entry_c_wrapper(
    result_cache_2_c_wrapper_t* wrapper,
    char* float_vector,
    float (*distance_function)(uint8_t*, uint8_t*, size_t),
    void (*result_conversion_function)(uint64_t, uint8_t*, size_t, uint64_t, uint64_t)
    )
{
    topkache::ResultCache2* result_cache = as_cache(wrapper->result_cache);
    result_cache->insertWLEntry3(
        *as_fvec(float_vector),
        distance_function,
        result_conversion_function
    );
}

void
consume_wl_entry_c_wrapper(
    result_cache_2_c_wrapper_t* wrapper,
    float (*distance_function)(uint8_t*, uint8_t*, size_t),
    void (*result_conversion_function)(uint64_t, uint8_t*, size_t, uint64_t, uint64_t)
    )
{
    topkache::ResultCache2* result_cache = as_cache(wrapper->result_cache);
    result_cache->consumeAgedWLEntry(distance_function, result_conversion_function);
}

void
mark_deleted_c_wrapper(
    result_cache_2_c_wrapper_t* wrapper,
    uint64_t vector_id)
{
    topkache::ResultCache2* result_cache = as_cache(wrapper->result_cache);
    result_cache->markVecDeleted(vector_id);
}

char*
print_cache_c_wrapper(
    result_cache_2_c_wrapper_t* wrapper
    )
{
    if (wrapper != nullptr)
    {
        topkache::ResultCache2* result_cache = as_cache(wrapper->result_cache);
        return (char*)result_cache->toString().c_str();
    }

    return NULL;
}

void
export_call_c_wrapper(result_cache_2_c_wrapper_t* wrapper) 
{
    if (wrapper != nullptr)
    {
        topkache::ResultCache2* result_cache = as_cache(wrapper->result_cache);
        result_cache->exportPrintAll();
    }
}

uint64_t
default_hash(char* data, std::size_t nbytes)
{
    if (data == nullptr || nbytes == 0)
        return 0;                                // choose any “empty” hash you like

    uint64_t hash = XXH3_64bits(data, nbytes);
    return hash;
}

float
l2_dist_c_wrapper(float* a, float* b, size_t dim)
{
    return l2_dist(a, b, dim);
}

float
ip_dist_c_wrapper(float* a, float* b, size_t dim)
{
    return inner_product_dist(a, b, dim);
}

void
set_distance_function_c_wrapper(
    result_cache_2_c_wrapper_t* wrapper,
    float (*distance_function)(uint8_t*, uint8_t*, size_t)
    )
{
    if (wrapper != nullptr)
    {
        topkache::ResultCache2* result_cache = as_cache(wrapper->result_cache);
        result_cache->setDistanceFunction(distance_function);
    }
}

void
stress_test_invalidate_random_c_wrapper(
    result_cache_2_c_wrapper_t* wrapper,
    float percent
    )
{
    if (wrapper != nullptr)
    {
        topkache::ResultCache2* result_cache = as_cache(wrapper->result_cache);
        result_cache->stressTestInvalidateRandom(percent);
    }
}
