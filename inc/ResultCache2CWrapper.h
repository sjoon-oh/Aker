#ifndef TOPKACHE_RESULTCACHE2_CWRAPPER_H
#define TOPKACHE_RESULTCACHE2_CWRAPPER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define TOPKACHE_START
#define TOPKACHE_END

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct ParameterInfoC
{
    uint32_t                            vector_dim;
    size_t                              vector_pool_size;
    size_t                              vector_list_size;
    size_t                              vector_data_size;
    size_t                              vector_intopk;
    size_t                              vector_extras;

    bool                                similar_match;
    bool                                use_fixed_threshold;
    float                               fixed_threshold;
    float                               start_threshold;

    float                               risk_threshold;
    float                               alpha_tighten;
    float                               alpha_loosen;

    uint8_t                             distance_type; // 0: L2, 1: Inner Product

} result_cache_parameter_c_t;


typedef struct ResultCache2CWrapper
{
    char*                               result_cache;
    result_cache_parameter_c_t          parameter;
} result_cache_2_c_wrapper_t;


void                                    import_topkache_parameter(
                                            char* path,
                                            result_cache_parameter_c_t* parameter
                                        );

bool                                    conversion_function_c_wrapper(
                                            void* src,
                                            size_t src_size,
                                            size_t dim,
                                            void* dst,
                                            uint8_t* aux
                                        );

result_cache_2_c_wrapper_t*             create_result_cache_2_c_wrapper(result_cache_parameter_c_t parameter);
void                                    destroy_result_cache_2_c_wrapper(
                                            result_cache_2_c_wrapper_t* wrapper);

char*                                   create_vector_2_c_wrapper(
                                            uint64_t vector_id,
                                            size_t vector_size,
                                            char* vector_data,
                                            uint64_t aux_data_1,
                                            uint64_t aux_data_2,
                                            float distance
                                        );
uint64_t                                get_vid_entry_2_c_wrapper(
                                            char* searched_entry
                                        );
uint64_t                                get_vid_vector_2_c_wrapper(
                                            char* vector_2_wrapper
                                        );

uint64_t                                get_vid_float_vector_2_c_wrapper(
                                            char* float_vector_2_wrapper
                                        );

char*                                   get_data_vector_2_c_wrapper(
                                            char* vector_2_wrapper
                                        );
void                                    set_distance_vector_2_c_wrapper(
                                            char* vector_2_wrapper,
                                            float distance
                                        );
void                                    destroy_vector_2_c_wrapper(
                                            char* vector_2_wrapper
                                        );

char*                                   create_float_vector_2_c_wrapper(
                                            char* vector_2_wrapper,
                                            size_t dim, size_t vector_data_size,
                                            bool (*conversion_function)(
                                                void*, size_t, size_t, void*, uint8_t*
                                            )
                                        );
void                                    destroy_float_vector_2_c_wrapper(
                                            char* float_vector_2_wrapper
                                        );

char*                                   make_cache_entry_c_wrapper(
                                            result_cache_2_c_wrapper_t* wrapper,
                                            char* query_vector,
                                            size_t vector_list_size,
                                            char** vector_slot_ref_list);
void                                    free_cache_entry_c_wrapper(
                                            char* query_vector
                                        );

char*                                   sim_search_c_wrapper(
                                            result_cache_2_c_wrapper_t* wrapper,
                                            char* float_query_vector,
                                            bool* similar_entry,
                                            bool* is_invalid,
                                            float (*distance_function)(uint8_t*, uint8_t*, size_t)
                                        );
char*                                   get_result_c_wrapper(
                                            char* searched_entry,
                                            int index
                                        );
char*                                   get_result_sets_c_wrapper(
                                            char* searched_entry
                                        );

void                                    debug_print_cache_entry_c_wrapper(
                                            char* entry, char* status_string
                                        );

bool                                    insert_cache_entry_c_wrapper(
                                            result_cache_2_c_wrapper_t* wrapper,
                                            uint64_t vector_id,
                                            char* new_cache_entry,
                                            char* float_query_vector
                                        );

bool                                    link_cache_entry_c_wrapper(
                                            result_cache_2_c_wrapper_t* wrapper,
                                            char* new_cache_entry,
                                            uint64_t found_id
                                        );

void                                    insert_wl_entry_c_wrapper(
                                            result_cache_2_c_wrapper_t* wrapper,
                                            char* float_vector,
                                            float (*distance_function)(uint8_t*, uint8_t*, size_t),
                                            void (*result_conversion_function)(uint64_t, uint8_t*, size_t, uint64_t, uint64_t)
                                        );

void                                    consume_wl_entry_c_wrapper(
                                            result_cache_2_c_wrapper_t* wrapper,
                                            float (*distance_function)(uint8_t*, uint8_t*, size_t),
                                            void (*result_conversion_function)(uint64_t, uint8_t*, size_t, uint64_t, uint64_t)
                                        );

void                                    mark_deleted_c_wrapper(
                                            result_cache_2_c_wrapper_t* wrapper,
                                            uint64_t vector_id
                                        );

char*                                   print_cache_c_wrapper(
                                            result_cache_2_c_wrapper_t* wrapper
                                        );
void                                    export_call_c_wrapper(result_cache_2_c_wrapper_t* wrapper);

void                                    stress_test_invalidate_random_c_wrapper(
                                            result_cache_2_c_wrapper_t* wrapper,
                                            float percent
                                        );

uint64_t                                default_hash(
                                            char* buffer, size_t size);

float                                   l2_dist_c_wrapper(
                                            float* a,
                                            float* b,
                                            size_t dim
                                        );

float                                   ip_dist_c_wrapper(
                                            float* a,
                                            float* b,
                                            size_t dim
                                        );


// Helper map functions
void                                    init_helpermap_c_wrapper();
void                                    insert_helpermap_c_wrapper(
                                            uint64_t key, 
                                            char* value,
                                            size_t value_size);

char*                                   get_helpermap_c_wrapper(uint64_t key);
void                                    clear_helpermap_c_wrapper();

#ifdef __cplusplus
}
#endif
#endif