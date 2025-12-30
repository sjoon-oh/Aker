#include "../test-common.hh"

// 
// Second, is to test the basic read operation
TEST_FUNCTION_START(test_exact_search_approx_insert)
{
    std::uint32_t dimension = 100;
    size_t operation_number = 1000;

    topkache::result_cache_parameter_t parameter_info = {
        .vector_dim = dimension,
        .vector_pool_size = (96 * 20) - 5,
        .vector_list_size = 96,
        .vector_data_size = sizeof(float) * 100,
        .vector_intopk = 10,
        .vector_extras = 5
    };

    topkache::ResultCache2 cache(parameter_info);
    std::vector<std::thread> threads;

    std::vector<std::uint8_t> operation_type_list;
    for (int i = 0; i < operation_number; i++)
        operation_type_list.push_back(generateUniformDist(100) % 3);

    std::atomic<bool> trigger(false);
    int number_of_threads = 8;

    for (int i = 0; i < number_of_threads; i++)
    {
        threads.push_back(std::thread(
            [&parameter_info, &logger, &trigger, &cache, &operation_type_list, i, operation_number]()
            {
                // Shuffle
                std::vector<std::uint8_t> operation_type_list_copy = operation_type_list;
                std::random_shuffle(operation_type_list_copy.begin(), operation_type_list_copy.end());

                const uint32_t vector_list_size = parameter_info.vector_list_size;
                std::vector<topkache::Vector2**> vector_list;
                std::vector<std::uint8_t*> convert_list;
                std::vector<topkache::vector_id_t> vector_id_list_history;

                logger.getLogger()->info("Thread({}) started.", i);

                while (!trigger.load())
                    ;

                for (int j = 0; j < operation_number; j++)
                {
                    topkache::vector_id_t query_vector_id = generateUniformDist(1000000);
                    topkache::vector_id_t previous_vector_id 
                        = vector_id_list_history.size() > 0 ? vector_id_list_history.back() : 0;

                    vector_id_list_history.push_back(query_vector_id);

                    // Randomly reuse the previous Id to simulate the cache hit
                    if ((generateUniformDist(100) % 2 == 0))
                    {
                        vector_id_list_history.push_back(query_vector_id);
                        query_vector_id = previous_vector_id;
                    }

                    topkache::result_cache_entry_t* found_entry = cache.exGetCEntryPassive(query_vector_id);

                    if (found_entry != nullptr)
                    {
                        // logger.getLogger()->info("Thread({}) GET-INSERT: Found {} -> {}", i, query_vector_id, true);
                        continue;
                    }
                    else
                    {
                        topkache::Vector2* query_vector = new topkache::Vector2(sizeof(float) * 100);
                        for (int i = 0; i < 100; i++)
                        {
                            float value = generateNormalDistFloat(0.0, 1.0);
                            float* array = (float*)query_vector->getVecData();

                            array[i] = value;
                        }

                        topkache::Vector2** vectors = prepareResultSampleVector(vector_list_size, 1000000);
                        vector_list.push_back(vectors);

                        topkache::result_cache_entry_t* new_entry =
                            cache.makeCEntry(query_vector, vector_list_size, vectors);

                        // This is where this test differs,
                        // We insert the entry with the conversion function
                        // Here we randomly selected function, thus we do not convert the data from a form into a different form.
                        topkache::float_qvec_t query_vector_data = {
                            .vector_id = query_vector_id,
                            .vector_data = (std::uint8_t*)malloc(sizeof(float) * 100),
                            .vector_dim = 100,
                            .conversion_function = [](void* src, size_t src_size, size_t dim, void* dst, std::uint8_t* aux) -> bool
                            {
                                float* src_array = (float*)src;
                                float* dst_array = (float*)dst;

                                for (int i = 0; i < 100; i++)
                                    dst_array[i] = src_array[i];

                                return true;
                            }
                        };

                        convert_list.push_back(query_vector_data.vector_data);

                        for (int i = 0; i < 100; i++)
                        {
                            float* array = (float*)query_vector->getVecData();
                            query_vector_data.vector_data[i] = array[i];
                        }
                        
                        bool insert_success = cache.insertCEntry2(
                            query_vector_id, 
                            new_entry, 
                            query_vector_data
                        );

                        delete query_vector;
                    }

                }

                for (auto list : vector_list)
                {
                    removeResultSampleVector(list, vector_list_size);
                }

                for (auto data : convert_list)
                {
                    free(data);
                }

                logger.getLogger()->info("Thread({}) finished.", i);
            }
        ));
    }

    trigger.store(true);

    for (auto& thread : threads)
        thread.join();


    logger.getLogger()->info("ResultCache2 status: {}", cache.toString());

}
TEST_FUNCTION_END()
