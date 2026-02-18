#include "ak_ann_cache2_c_wrapper.h"
#include "core/ak_ann_cache2.hh"
#include "utils/ak_param_parser.hh"
#include "utils/ak_distance_funcs.hh"

#include "xxHash/xxh3.h"

#include <cassert>
#include <cstdlib>
#include <cstring>

#include <map>

// Additional helper hashtable
static std::map<uint64_t, char*> helper_map;

// Pointer dance
static inline aker::ANNCache2* as_cache(char* p) { 
    return reinterpret_cast<aker::ANNCache2*>(p); 
}
static inline char* as_char(aker::ANNCache2* c) {
    return reinterpret_cast<char*>(c); 
}
static inline aker::VectorSlot* as_vec(char* p) { 
    return reinterpret_cast<aker::VectorSlot*>(p); 
}
static inline char* as_char(aker::VectorSlot* v) { 
    return reinterpret_cast<char*>(v); 
}
static inline aker::vector_view_t* as_fvec(char* p) { 
    return reinterpret_cast<aker::vector_view_t*>(p); 
}
static inline char* as_char(aker::vector_view_t* v) { 
    return reinterpret_cast<char*>(v); 
}
static inline aker::result_cache_entry_t* as_entry(char* e) { 
    return reinterpret_cast<aker::result_cache_entry_t*>(e); 
}
static inline char* as_char(aker::result_cache_entry_t* e) { 
    return reinterpret_cast<char*>(e); 
}

void
import_aker_parameter(char* path, ann_cache_parameter_c_t* parameter)
{
    std::string file_path(path);

    aker::ParameterParser parser(file_path);
    aker::ann_cache_config_t parameter_info = parser.getParameter();

    parameter->vector_dim               = parameter_info.vector_format.vector_dim;
    parameter->vector_pool_size         = parameter_info.capacity.slot_pool_size;
    parameter->vector_list_size         = parameter_info.capacity.slot_list_size;
    parameter->vector_data_size         = parameter_info.vector_format.vector_data_size;
    parameter->vector_intopk            = parameter_info.capacity.vector_in_topk;
    parameter->vector_extras            = parameter_info.capacity.vector_extras;
    parameter->similar_match            = parameter_info.tuning.similar_match;
    parameter->use_fixed_thresh      = parameter_info.tuning.use_fixed_thresh;
    parameter->fixed_thresh          = parameter_info.tuning.fixed_thresh;
    parameter->start_thresh          = parameter_info.tuning.start_thresh;
    parameter->risk_thresh           = parameter_info.tuning.risk_thresh;
    parameter->alpha_tighten            = parameter_info.tuning.alpha_tighten;
    parameter->alpha_loosen             = parameter_info.tuning.alpha_loosen;

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

ann_cache_2_c_wrapper_t* 
create_ann_cache_2_c_wrapper(
    ann_cache_parameter_c_t parameter
    )
{
    
    aker::ann_cache_config_t parameter_info;

    parameter_info.vector_format.vector_dim           = parameter.vector_dim;
    parameter_info.capacity.slot_pool_size     = parameter.vector_pool_size;
    parameter_info.capacity.slot_list_size     = parameter.vector_list_size;
    parameter_info.vector_format.vector_data_size     = parameter.vector_data_size;
    parameter_info.capacity.vector_in_topk        = parameter.vector_intopk;
    parameter_info.capacity.vector_extras        = parameter.vector_extras;

    parameter_info.tuning.similar_match        = parameter.similar_match;
    parameter_info.tuning.use_fixed_thresh  = parameter.use_fixed_thresh;
    parameter_info.tuning.fixed_thresh      = parameter.fixed_thresh;
    parameter_info.tuning.start_thresh      = parameter.start_thresh;

    parameter_info.tuning.risk_thresh       = parameter.risk_thresh;
    parameter_info.tuning.alpha_tighten        = parameter.alpha_tighten;
    parameter_info.tuning.alpha_loosen         = parameter.alpha_loosen;
    parameter_info.distance_type        = static_cast<aker::distance_type_t>(parameter.distance_type);

    ann_cache_2_c_wrapper_t* wrapper = new ann_cache_2_c_wrapper_t;
    wrapper->result_cache = as_char(new aker::ANNCache2(parameter_info));

    wrapper->parameter.vector_dim           = parameter_info.vector_format.vector_dim;
    wrapper->parameter.vector_pool_size     = parameter_info.capacity.slot_pool_size;
    wrapper->parameter.vector_list_size     = parameter_info.capacity.slot_list_size;
    wrapper->parameter.vector_data_size     = parameter_info.vector_format.vector_data_size;
    wrapper->parameter.vector_intopk        = parameter_info.capacity.vector_in_topk;
    wrapper->parameter.vector_extras        = parameter_info.capacity.vector_extras;
    
    wrapper->parameter.similar_match        = parameter_info.tuning.similar_match;
    wrapper->parameter.use_fixed_thresh  = parameter_info.tuning.use_fixed_thresh;
    wrapper->parameter.fixed_thresh      = parameter_info.tuning.fixed_thresh;
    wrapper->parameter.start_thresh      = parameter_info.tuning.start_thresh;

    wrapper->parameter.risk_thresh       = parameter_info.tuning.risk_thresh;
    wrapper->parameter.alpha_tighten        = parameter_info.tuning.alpha_tighten;
    wrapper->parameter.alpha_loosen         = parameter_info.tuning.alpha_loosen;
    wrapper->parameter.distance_type        = static_cast<uint8_t>(parameter_info.distance_type);

    return wrapper;
}

void
destroy_ann_cache_2_c_wrapper(
    ann_cache_2_c_wrapper_t* wrapper
    )
{
    if (wrapper != nullptr)
    {
        delete as_cache(wrapper->result_cache);
        delete wrapper;
    }
}

char*
create_vector_slot_c_wrapper(
    uint64_t vector_id,
    size_t vector_size,
    char* vector_data,
    uint64_t aux_data_1,
    uint64_t aux_data_2,
    float distance
    )
{
    aker::VectorSlot* vector = new aker::VectorSlot(vector_size);

    vector->setVectorId(vector_id);
    vector->setAuxData1(aux_data_1);
    vector->setAuxData2(aux_data_2);
    
    if (vector_data != nullptr)
    {
        std::memcpy(vector->getVectorData(), vector_data, vector_size);
    }
    vector->setDistance(distance);

    return as_char(vector);
}

uint64_t
get_vector_id_entry_2_c_wrapper(
    char* searched_entry
    )
{
    aker::result_cache_entry_t* entry = as_entry(searched_entry);
    return entry->query_vector->getVectorId();
}

uint64_t
get_vector_id_vector_slot_c_wrapper(
    char* vector_2_wrapper
    )
{
    aker::VectorSlot* vector = as_vec(vector_2_wrapper);
    return vector->getVectorId();
}

uint64_t
get_vector_id_vector_view_c_wrapper(
    char* float_vector_2_wrapper
)
{
    aker::vector_view_t* float_vector = as_fvec(float_vector_2_wrapper);
    return float_vector->vector_id;
}

char*
get_vector_data_vector_slot_c_wrapper(
    char* vector_2_wrapper
    )
{
    aker::VectorSlot* vector = as_vec(vector_2_wrapper);
    return reinterpret_cast<char*>(vector->getVectorData());
}

void
set_distance_vector_slot_c_wrapper(
    char* vector_2_wrapper,
    float distance
    )
{
    aker::VectorSlot* vector = as_vec(vector_2_wrapper);
    vector->setDistance(distance);
}

void
destroy_vector_slot_c_wrapper(
    char* vector_2_wrapper
    )
{
    if (vector_2_wrapper != nullptr)
    {
        delete as_vec(vector_2_wrapper);
    }
}

char*
create_vector_view_c_wrapper(
    char* vector_2_wrapper, size_t dim, size_t vector_data_size,
    bool (*conversion_function)(
        void*, size_t, size_t, void*, uint8_t*
    )
    )
{
    aker::VectorSlot* vector = as_vec(vector_2_wrapper);
    aker::vector_view_t* float_vector = new aker::vector_view_t;

    float_vector->vector_id         = vector->getVectorId();
    float_vector->vector_data       = vector->getVectorData();
    float_vector->vector_dim        = dim;
    float_vector->vector_data_size  = vector_data_size;
    float_vector->conversion_function = conversion_function;

    float_vector->aux_data_1        = vector->getAuxData1();
    float_vector->aux_data_2        = vector->getAuxData2();

    return reinterpret_cast<char*>(float_vector);
}

void
destroy_vector_view_c_wrapper(
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
    ann_cache_2_c_wrapper_t* wrapper,
    char* query_vector,
    size_t vector_list_size,
    char** vector_slot_ref_list
    )
{
    aker::ANNCache2* result_cache = as_cache(wrapper->result_cache);

    aker::result_cache_entry_t* entry =
        result_cache->makeCEntry(
            as_vec(query_vector),
            vector_list_size,
            reinterpret_cast<aker::VectorSlot**>(vector_slot_ref_list)
        );
    
    return as_char(entry);
}

void
free_cache_entry_c_wrapper(char* cache_entry_ptr)
{
    /* Frees a cache entry returned to the caller.
     * This mirrors ANNCache2::freeCEntry() without requiring a cache instance.
     */
    if (cache_entry_ptr == nullptr)
        return;

    aker::result_cache_entry_t* entry = as_entry(cache_entry_ptr);

    if (entry->entry_kind == aker::RESULT_CACHE_ENTRY_KIND_RETURNED_COPY)
    {
        if (entry->vector_slot_ref_list != nullptr)
        {
            for (size_t i = 0; i < entry->vector_list_size; i++)
                delete entry->vector_slot_ref_list[i];
        }
    }

    delete entry->query_vector;
    entry->query_vector = nullptr;

    free(entry->vector_slot_ref_list);
    entry->vector_slot_ref_list = nullptr;

    delete entry;
}

char*
sim_search_c_wrapper(
    ann_cache_2_c_wrapper_t* wrapper,
    char* float_query_vector,
    bool* similar_entry,
    bool* is_invalid,
    float (*distance_function)(uint8_t*, uint8_t*, size_t)
    )
{
    aker::result_cache_entry_t* entry = nullptr;
    
    if (float_query_vector != nullptr)
    {
        aker::vector_view_t* query_vector_data = as_fvec(float_query_vector);
        aker::ANNCache2* result_cache = as_cache(wrapper->result_cache);

        aker::result_cache_entry_t* found_entry =
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
        aker::result_cache_entry_t* entry = as_entry(searched_entry);
        return reinterpret_cast<char*>(entry->vector_slot_ref_list);
    }

    return NULL;
}

void
debug_print_cache_entry_c_wrapper(char* entry, char* status_string)
{
    if (entry != nullptr)
    {
        aker::result_cache_entry_t* cache_entry = as_entry(entry);
        
        std::string status = "";
        status += "{";
        status += "Entry ID: " + std::to_string(cache_entry->query_vector->getVectorId()) + ", ";
        status += "Entry Size: " + std::to_string(cache_entry->vector_list_size) + ", ";
        status += "Entry Threshold: " + std::to_string(cache_entry->thresh) + ", ";
        status += "Entry Min Distance: " + std::to_string(cache_entry->min_distance) + ", ";
        status += "Entry Max Distance: " + std::to_string(cache_entry->max_distance) + ", ";
        status += "Entry Risk Factor: " + std::to_string(cache_entry->risk_factor) + ", ";
        
        status += "List: [";
        for (int i = 0; i < cache_entry->vector_list_size; i++)
        {
            if (i > 0)
                status += ", ";
            status += "{";
            aker::VectorSlot* element = cache_entry->vector_slot_ref_list[i];

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

            status += std::to_string(element->getVectorId()) + ", ";
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
        aker::result_cache_entry_t* entry = as_entry(searched_entry);
        return reinterpret_cast<char*>(entry->vector_slot_ref_list[index]);
    }

    return NULL;
}

bool
insert_cache_entry_c_wrapper(
        ann_cache_2_c_wrapper_t* wrapper,
        uint64_t vector_id,
        char* new_cache_entry,
        char* float_query_vector)
{
    aker::vector_view_t* query_vector_data = as_fvec(float_query_vector);
    aker::ANNCache2* result_cache = as_cache(wrapper->result_cache);

    return result_cache->insertCEntry2(
        vector_id,
        as_entry(new_cache_entry),
        *query_vector_data
    );
}

bool
link_cache_entry_c_wrapper(
    ann_cache_2_c_wrapper_t* wrapper,
    char* new_cache_entry,
    uint64_t found_id)
{
    aker::ANNCache2* result_cache = as_cache(wrapper->result_cache);

    if (as_entry(new_cache_entry)->vector_slot_ref_list == NULL)
        as_entry(new_cache_entry)->vector_slot_ref_list = nullptr; 

    return result_cache->linkCEntry(
        as_entry(new_cache_entry),
        found_id
    );
}

void
insert_wl_entry_c_wrapper(
    ann_cache_2_c_wrapper_t* wrapper,
    char* float_vector,
    float (*distance_function)(uint8_t*, uint8_t*, size_t),
    void (*result_conversion_function)(uint64_t, uint8_t*, size_t, uint64_t, uint64_t)
    )
{
    aker::ANNCache2* result_cache = as_cache(wrapper->result_cache);
    result_cache->insertWLEntry3(
        *as_fvec(float_vector),
        distance_function,
        result_conversion_function
    );
}

void
consume_wl_entry_c_wrapper(
    ann_cache_2_c_wrapper_t* wrapper,
    float (*distance_function)(uint8_t*, uint8_t*, size_t),
    void (*result_conversion_function)(uint64_t, uint8_t*, size_t, uint64_t, uint64_t)
    )
{
    aker::ANNCache2* result_cache = as_cache(wrapper->result_cache);
    result_cache->consumeAgedWLEntry(distance_function, result_conversion_function);
}

void
mark_deleted_c_wrapper(
    ann_cache_2_c_wrapper_t* wrapper,
    uint64_t vector_id)
{
    aker::ANNCache2* result_cache = as_cache(wrapper->result_cache);
    result_cache->markVectorDeleted(vector_id);
}

char*
print_cache_c_wrapper(
    ann_cache_2_c_wrapper_t* wrapper
    )
{
    if (wrapper != nullptr)
    {
        aker::ANNCache2* result_cache = as_cache(wrapper->result_cache);
        static thread_local std::string tl_cache_string;
        tl_cache_string = result_cache->getStatusText();
        return (char*)tl_cache_string.c_str();
    }

    return NULL;
}

void
export_call_c_wrapper(ann_cache_2_c_wrapper_t* wrapper) 
{
    if (wrapper != nullptr)
    {
        aker::ANNCache2* result_cache = as_cache(wrapper->result_cache);
        result_cache->exportTraceToFiles();
    }
}

void
stress_test_invalidate_random_c_wrapper(
    ann_cache_2_c_wrapper_t* wrapper,
    float percent
    )
{
    if (wrapper != nullptr)
    {
        aker::ANNCache2* result_cache = as_cache(wrapper->result_cache);
        result_cache->stressTestInvalidateRandom(percent);
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
init_helpermap_c_wrapper()
{
    helper_map.clear();
}

void
insert_helpermap_c_wrapper(uint64_t key, char* value, size_t value_size)
{
    char* value_copy = new char[value_size];
    std::memcpy(value_copy, value, value_size);
    helper_map[key] = value_copy;
}

char*
get_helpermap_c_wrapper(uint64_t key)
{
    if (helper_map.find(key) != helper_map.end())
    {
        return helper_map[key];
    }

    return nullptr;
}

void
clear_helpermap_c_wrapper()
{
    for (auto& pair : helper_map)
    {
        delete[] pair.second;
    }
    helper_map.clear();
}