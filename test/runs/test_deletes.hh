#include "../test-common.hh"

TEST_FUNCTION_START(test_deletes_1)
{
    std::uint32_t dimension = 100;
    size_t operation_number = 10000;

    topkache::result_cache_parameter_t parameter_info = {
        .vector_dim = dimension,
        .vector_pool_size = (100 * 500) - 20, // The number is about 500K entries.
        .vector_list_size = 96,
        .vector_data_size = sizeof(float) * 100,
        .vector_intopk = 50,
        .vector_extras = 46
    };

    topkache::ResultCache2 cache(parameter_info);

    // We record all the inserted vectors in a map
    std::atomic_flag inserted_vector_map_lock = ATOMIC_FLAG_INIT;

    std::map<topkache::vector_id_t, topkache::Vector2*> inserted_qvecs_map;
    std::map<topkache::vector_id_t, topkache::Vector2*> inserted_slotvecs_map;

    // First, we randomly fills out some data to the cache.
    std::vector<std::thread> prefill_threads;
    for (int tid = 0; tid < 1; tid++)
    {
        prefill_threads.push_back(std::thread(
            [&parameter_info, &logger, &cache, operation_number, tid,
                &inserted_vector_map_lock, &inserted_qvecs_map,
                &inserted_slotvecs_map
            ]()
            {
                const uint32_t vector_list_size = parameter_info.vector_list_size;
                
                std::vector<topkache::Vector2*> prefill_list;

                generateSampleQueryVectors(
                    prefill_list,
                    operation_number,
                    parameter_info,
                    10000000
                );

                for (int i = 0; i < operation_number; i++)
                {

                    // Lock the map before inserting
                    while (inserted_vector_map_lock.test_and_set(std::memory_order_acquire))
                        ;            

                    topkache::Vector2* query_vector = prefill_list[i];
                    topkache::vector_id_t query_vector_id = query_vector->getVecId();

                    topkache::Vector2** vectors = prepareResultSampleVector(vector_list_size, 10000000);
                    topkache::result_cache_entry_t* new_entry = cache.makeCEntry(query_vector, vector_list_size, vectors);

                    for (int j = 0; j < vector_list_size; j++)
                    {
                        topkache::vector_id_t result_id = vectors[j]->getVecId();
                        inserted_slotvecs_map[result_id] = nullptr;
                    }

                    new_entry->min_distance = 12.0;
                    new_entry->max_distance = 15.0;

                    topkache::float_qvec_t float_query;
                    convertVec2ToFVec(query_vector, float_query, parameter_info);

                    bool is_present = (inserted_qvecs_map.find(query_vector_id) != inserted_qvecs_map.end());
                    if (is_present)
                    {
                        delete query_vector;
                    }
                    else
                    {
                        inserted_qvecs_map[query_vector_id] = query_vector;
                        bool insert_success = cache.insertCEntry2(
                            query_vector_id,
                            new_entry, 
                            float_query
                        );
                    }

                    free(float_query.vector_data);

                    inserted_vector_map_lock.clear(std::memory_order_release);

                    if (i % 1000 == 0)
                    {
                        // Log the progress
                        float progress = (float)i / (float)operation_number * 100.0;
                        logger.getLogger()->info("tid({}): {}%", tid, progress);
                    }
                }
            }
        ));
    }

    for (auto& thread : prefill_threads)
        thread.join();

    logger.getLogger()->info("Generated entries.");

    // Filling means that these are what we have in the base data.
    
    std::atomic<bool> trigger(false);
    
    // We tract what is in the cache.

    std::vector<topkache::Vector2*> current_vectors;
    cache.getPooledVecs(current_vectors);

    size_t current_size = current_vectors.size();
    logger.getLogger()->info("Current vector list size: {}", current_size);

    // We delete half of the entries randomly.

    std::vector<topkache::vector_id_t> delete_list;
    for (int i = 0; i < current_size / 2; i++)
    {
        size_t random_index = rand() % current_size;
        topkache::Vector2* vector = current_vectors[random_index];

        delete_list.push_back(vector->getVecId());
    }

    // We delete the entries.
    for (auto& vector_id : delete_list)
    {
        cache.markVecDeleted(vector_id);
    }

    // We check the cache status.
    logger.getLogger()->info(" >> After delete, mid-check");
    logger.getLogger()->info("ResultCache2 status: {}", cache.toString());
    
    logger.getLogger()->info("Checking exGetCEntryActive");
    logger.getLogger()->info(" >> Vector pool map size: {}", inserted_slotvecs_map.size());

    size_t invalid_count = 0;
    for (auto& vector : inserted_qvecs_map)
    {
        topkache::vector_id_t vector_id = vector.first;

        bool is_invalid;

        topkache::result_cache_entry_t* found_entry = cache.exGetCEntryActive(vector_id, is_invalid);
        if (found_entry != nullptr)
        {
            // logger.getLogger()->info("Found entry: {}, version: {}", 
            //     found_entry->query_vector->getVecId(),
            //     found_entry->version
            // );
        }
        else
        {
            if (is_invalid)
            {
                // logger.getLogger()->info("Entry is invalid (DEL): {}", vector_id);
                invalid_count++;
            }
            else
            {

            }
        }
    }

    logger.getLogger()->info("Invalid count: {}", invalid_count, " / {}", inserted_qvecs_map.size());
    logger.getLogger()->info("ResultCache2 status: {}", cache.toString());
}
TEST_FUNCTION_END()


TEST_FUNCTION_START(test_deletes_2)
{
    std::uint32_t dimension = 100;
    size_t operation_number = 10000;

    topkache::result_cache_parameter_t parameter_info = {
        .vector_dim = dimension,
        .vector_pool_size = (100 * 500) - 20, // The number is about 500K entries.
        .vector_list_size = 96,
        .vector_data_size = sizeof(float) * 100,
        .vector_intopk = 50,
        .vector_extras = 46
    };

    topkache::ResultCache2 cache(parameter_info);

    // We record all the inserted vectors in a map
    std::atomic_flag inserted_vector_map_lock = ATOMIC_FLAG_INIT;

    std::map<topkache::vector_id_t, topkache::Vector2*> inserted_qvecs_map;
    std::map<topkache::vector_id_t, topkache::Vector2*> inserted_slotvecs_map;

    // First, we randomly fills out some data to the cache.
    std::vector<std::thread> prefill_threads;
    for (int tid = 0; tid < 1; tid++)
    {
        prefill_threads.push_back(std::thread(
            [&parameter_info, &logger, &cache, operation_number, tid,
                &inserted_vector_map_lock, &inserted_qvecs_map,
                &inserted_slotvecs_map
            ]()
            {
                const uint32_t vector_list_size = parameter_info.vector_list_size;
                
                std::vector<topkache::Vector2*> prefill_list;

                generateSampleQueryVectors(
                    prefill_list,
                    operation_number,
                    parameter_info,
                    10000000
                );

                for (int i = 0; i < operation_number; i++)
                {

                    // Lock the map before inserting
                    while (inserted_vector_map_lock.test_and_set(std::memory_order_acquire))
                        ;            

                    topkache::Vector2* query_vector = prefill_list[i];
                    topkache::vector_id_t query_vector_id = query_vector->getVecId();

                    topkache::Vector2** vectors = prepareResultSampleVector(vector_list_size, 10000000);
                    topkache::result_cache_entry_t* new_entry = cache.makeCEntry(query_vector, vector_list_size, vectors);

                    for (int j = 0; j < vector_list_size; j++)
                    {
                        topkache::vector_id_t result_id = vectors[j]->getVecId();
                        topkache::Vector2* alloc_vec = new topkache::Vector2(parameter_info.vector_data_size);

                        alloc_vec->setVecId(result_id);
                        std::memcpy(
                            alloc_vec->getVecData(),
                            vectors[j]->getVecData(),
                            parameter_info.vector_data_size
                        );
                        
                        inserted_slotvecs_map[result_id] = alloc_vec;
                    }

                    new_entry->min_distance = 12.0;
                    new_entry->max_distance = 15.0;

                    topkache::float_qvec_t float_query;
                    convertVec2ToFVec(query_vector, float_query, parameter_info);

                    bool is_present = (inserted_qvecs_map.find(query_vector_id) != inserted_qvecs_map.end());
                    if (is_present)
                    {
                        delete query_vector;
                    }
                    else
                    {
                        inserted_qvecs_map[query_vector_id] = query_vector;
                        bool insert_success = cache.insertCEntry2(
                            query_vector_id,
                            new_entry, 
                            float_query
                        );
                    }

                    free(float_query.vector_data);

                    inserted_vector_map_lock.clear(std::memory_order_release);

                    if (i % 1000 == 0)
                    {
                        // Log the progress
                        float progress = (float)i / (float)operation_number * 100.0;
                        logger.getLogger()->info("tid({}): {}%", tid, progress);
                    }
                }
            }
        ));
    }

    for (auto& thread : prefill_threads)
        thread.join();

    logger.getLogger()->info("Generated entries.");

    // Filling means that these are what we have in the base data.
    
    std::atomic<bool> trigger(false);
    size_t invalid_count = 0;    
    
    // We tract what is in the cache.

    std::vector<topkache::Vector2*> current_vectors;
    cache.getPooledVecs(current_vectors);

    size_t current_size = current_vectors.size();
    logger.getLogger()->info("Current vector list size: {}", current_size);

    // We delete half of the entries randomly.

    std::vector<topkache::vector_id_t> delete_list;
    for (int i = 0; i < current_size / 2; i++)
    {
        size_t random_index = rand() % current_size;
        topkache::Vector2* vector = current_vectors[random_index];

        delete_list.push_back(vector->getVecId());
    }

    // We delete the entries.
    for (auto& vector_id : delete_list)
    {
        cache.markVecDeleted(vector_id);
    }

    // We check the cache status.
    logger.getLogger()->info(" >> After delete, mid-check");
    logger.getLogger()->info("ResultCache2 status: {}", cache.toString());
    
    logger.getLogger()->info("Checking simGetCEntry");
    logger.getLogger()->info("Vector query list size: {}", inserted_qvecs_map.size());

    for (auto& vector : inserted_qvecs_map)
    {
        topkache::vector_id_t vector_id = vector.first;
        topkache::Vector2* query_vector = vector.second;

        topkache::float_qvec_t float_query;
        convertVec2ToFVec(
            query_vector,
            float_query, parameter_info
        );

        // We find the element in the cacne.
        bool is_similar = false;
        bool is_invalid = false;

        topkache::result_cache_entry_t* found_entry = cache.simGetCEntry(
            float_query,
            is_similar,
            is_invalid,
            topkache::sampleL2Dist
        );

        if (found_entry != nullptr)
        {

        }
        else
        {
            if (is_invalid)
            {
                invalid_count++;
            }
        }
    }

    logger.getLogger()->info(" >> Invalid count: {}", invalid_count, " / {}", inserted_qvecs_map.size());
    logger.getLogger()->info("ResultCache2 status: {}", cache.toString());
}
TEST_FUNCTION_END()