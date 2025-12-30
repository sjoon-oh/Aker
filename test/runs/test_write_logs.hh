#include "../test-common.hh"

// Atomic counter
#include <atomic>

std::atomic<uint32_t> prefills_done(0);

TEST_FUNCTION_START(test_write_logs)
{
    std::uint32_t dimension = 100;
    size_t operation_number = 10000;

    topkache::result_cache_parameter_t parameter_info = {
        .vector_dim = dimension,
        .vector_pool_size = (100 * 1000) - 20, // The number is about 500K entries.
        .vector_list_size = 96,
        .vector_data_size = sizeof(float) * 100,
        .vector_intopk = 40,
        .vector_extras = 56,
        .similar_match = 1,
        .fixed_threshold = 0.0,
        .start_threshold = 100,
        .risk_threshold = 0.1,
        .alpha_tighten = 0.5,
        .alpha_loosen = 1.3
    };

    topkache::ResultCache2 cache(parameter_info);

    // We record all the inserted vectors in a map
    std::atomic_flag inserted_vector_map_lock = ATOMIC_FLAG_INIT;
    std::map<topkache::vector_id_t, topkache::Vector2*> inserted_vectors_map;

    // First, we randomly fills out some data to the cache.
    std::vector<std::thread> prefill_threads;
    for (int tid = 0; tid < 10; tid++)
    {
        prefill_threads.push_back(std::thread(
            [&parameter_info, &logger, &cache, operation_number, tid,
                &inserted_vector_map_lock, &inserted_vectors_map, &prefills_done]()
            {
                const uint32_t vector_list_size = parameter_info.vector_list_size;
                
                std::vector<topkache::Vector2*> prefill_list;
                std::vector<topkache::Vector2*> inserted_vector_list;

                generateSampleQueryVectors(
                    prefill_list,
                    operation_number,
                    parameter_info,
                    10000000
                );

                generateSampleQueryVectors(
                    inserted_vector_list,
                    operation_number,
                    parameter_info,
                    10000000
                );

                for (int i = 0; i < operation_number; i++)
                {
                    topkache::Vector2* query_vector = prefill_list[i];
                    topkache::vector_id_t query_vector_id = query_vector->getVecId();

                    topkache::Vector2** vectors = prepareResultSampleVector(vector_list_size, 10000000);
                    topkache::result_cache_entry_t* new_entry = cache.makeCEntry(query_vector, vector_list_size, vectors);

                    // Set the distance range
                    // 

                    // Randomly insert first to the wlog
                    if ((rand() % 4) == 0)
                    {
                        topkache::float_qvec_t float_query_insert_wlog;
                        convertVec2ToFVec(
                            (inserted_vector_list[i]),
                            float_query_insert_wlog, parameter_info
                        );

                        cache.insertWLEntry3(
                            float_query_insert_wlog,
                            topkache::sampleL2Dist
                        );
                    }
                    
                    std::vector<float> test_distances;
                    for (int j = 0; j < vector_list_size; j++)
                    {
                        test_distances.push_back(generateNormalDistFloat(14.5, 1));
                    }

                    // Sort them 
                    std::sort(test_distances.begin(), test_distances.end());
                    for (int j = 0; j < vector_list_size; j++)
                    {
                        vectors[j]->setDistance(test_distances[j]);
                    }

                    new_entry->min_distance = test_distances[0];
                    new_entry->max_distance = test_distances[vector_list_size - 1];

                    new_entry->threshold = new_entry->min_distance;

                    topkache::float_qvec_t float_query;
                    convertVec2ToFVec(query_vector, float_query, parameter_info);

                    // Lock the map before inserting
                    while (inserted_vector_map_lock.test_and_set(std::memory_order_acquire))
                        ;

                    bool is_present = (inserted_vectors_map.find(query_vector_id) != inserted_vectors_map.end());
                    if (is_present)
                    {
                        delete query_vector;
                    }
                    else
                    {
                        inserted_vectors_map[query_vector_id] = query_vector;
                        bool insert_success = cache.insertCEntry2(
                            query_vector_id,
                            new_entry, 
                            float_query
                        );
                    }

                    free(float_query.vector_data);

                    inserted_vector_map_lock.clear(std::memory_order_release);

                    prefills_done++;
                }
            }        
        ));
    }

    for (auto& thread : prefill_threads)
        thread.join();

    logger.getLogger()->info("Prefill done.");

    std::atomic<bool> trigger(false);

    // We run in single thread,
    // because we want to use the what is already in the cache.

    struct SearchResult
    {
        bool is_found;
        bool is_similar;
        float distance;
    };

    int apply_calls = 0;
    for (auto& element: inserted_vectors_map)
    {
        topkache::Vector2* query_vector = element.second;

        topkache::float_qvec_t float_query;
        convertVec2ToFVec(
            query_vector,
            float_query, parameter_info);

        struct SearchResult result = {
            .is_found = false,
            .is_similar = false,
            .distance = 0.0
        }; 

        topkache::result_cache_entry_t* found_entry 
            = cache.simGetCEntry(
                float_query,
                result.is_similar,
                result.is_found,
                topkache::sampleL2Dist
            );

        cache.consumeAgedWLEntry(topkache::sampleL2Dist);

        // return true;

        free(float_query.vector_data);

        delete query_vector;
    }

    logger.getLogger()->info("All write logs are checked.");

    logger.getLogger()->info("Apply calls: {}", apply_calls);

    logger.getLogger()->info("Write logs inserted.");
    logger.getLogger()->info("ResultCache2 status: {}", cache.toString());

    logger.getLogger()->info("Test write logs done.");

}
TEST_FUNCTION_END()
