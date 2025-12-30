
#include "ResultCache2CWrapper.h"
#include "test-common-c.h"

#include <stdio.h>
#include <assert.h>

float
distance_function_local(
    uint8_t* a,
    uint8_t* b,
    size_t dim)
{
    float distance = 0.0f;
    for (size_t i = 0; i < dim; i++)
    {
        float diff = ((float*)a)[i] - ((float*)b)[i];
        distance += diff * diff;
    }
    return distance;
}

bool
conversion_function_local(
    void* src,
    size_t src_size,
    size_t dim,
    void* dst,
    uint8_t* aux) {
    
    // Dummy conversion function

    float* src_array = (float*)src;
    float* dst_array = (float*)dst;

    for (size_t i = 0; i < dim; i++)
    {
        dst_array[i] = src_array[i];
    }

    return true;
}

int
main(int argc, char** argv)
{
    printf("Starting test...\n");

    result_cache_parameter_c_t parameter = {
        .vector_dim        = 100,
        .vector_pool_size  = 10000,
        .vector_list_size  = 96,
        .vector_data_size  = sizeof(float) * 100,
        .vector_intopk     = 10,
        .vector_extras     = 5
    };

    const size_t query_num = 10000;

    result_cache_2_c_wrapper_t* cache = create_result_cache_2_c_wrapper(parameter);

    {
        char** query_vectors = generate_sample_query_vectors(query_num, &parameter, 10000000);
        if (query_vectors == NULL)
        {
            fprintf(stderr, "Failed to generate sample query vectors.\n");
            return -1;
        }

        // Test inserts
        for (size_t i = 0; i < query_num; i++)
        {
            char* query_vector = query_vectors[i];
            char* vector_list[96] = { 0, };

            // Fill the vector list with random data
            for (size_t j = 0; j < 96; j++)
            {
                vector_list[j] = create_vector_2_c_wrapper(0, parameter.vector_data_size, NULL, 0, 0, 0);
            }

            // Create a cache entry
            char* float_query_vector = create_float_vector_2_c_wrapper(
                query_vector, 
                parameter.vector_dim, 
                parameter.vector_data_size, 
                conversion_function_local
            );

            // Simulate a search
            bool similar_entry = false;
            bool is_invalid = false;
            char* found_entry = sim_search_c_wrapper(
                cache, 
                float_query_vector,
                &similar_entry, &is_invalid, distance_function_local
            );

            if (found_entry != NULL)
            {
                printf("Found entry for query vector %zu\n", i);
                destroy_result_cache_2_c_wrapper(found_entry);
            }
            
        }

        cleanup_sample_query_vectors(query_vectors, query_num);
    }

    printf("Similarity search completed.\n");
    printf("Status: %s\n", print_cache_c_wrapper(cache));

    printf("Inserting cache entries...\n");

    {
        char** query_vectors = generate_sample_query_vectors(query_num, &parameter, 10000000);
        if (query_vectors == NULL)
        {
            fprintf(stderr, "Failed to generate sample query vectors.\n");
            return -1;
        }

        // Test inserts
        for (size_t i = 0; i < query_num; i++)
        {
            char* query_vector = query_vectors[i];
            char* vector_list[96] = { 0, };

            // Fill the vector list with random data
            for (size_t j = 0; j < 96; j++)
            {
                // Make random vector ID
                uint64_t vector_id = rand() % 10000000;
                vector_list[j] = create_vector_2_c_wrapper(vector_id, parameter.vector_data_size, NULL, 0, 0, 0);
            }

            // Create a cache entry
            char* entry = make_cache_entry_c_wrapper(
                cache,
                query_vector,
                parameter.vector_list_size,
                vector_list
            );

            assert(entry != NULL);

            char* float_query_vector = create_float_vector_2_c_wrapper(
                query_vector, 
                parameter.vector_dim, 
                parameter.vector_data_size, 
                conversion_function_local
            );
            assert(float_query_vector != NULL);

            // Simulate an insert
            bool inserted = insert_cache_entry_c_wrapper(
                cache,
                get_vid_vector_2_c_wrapper(query_vector),
                entry,
                float_query_vector
            );

            // printf("Inserted entry for query vector %zu: %s\n", i, inserted ? "Success" : "Failed");
        }

        cleanup_sample_query_vectors(query_vectors, query_num);
    }    
    // Delete the cache
    destroy_result_cache_2_c_wrapper(cache);
    
    printf("Test completed successfully.\n");

    return 0;
}