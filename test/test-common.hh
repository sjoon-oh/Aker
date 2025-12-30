#include "logger.hh"
#include "core/ResultCache2.hh"

#include <thread>
#include <random>
#include <atomic>

#ifndef TOPKACHE_TEST_COMMON_H
#define TOPKACHE_TEST_COMMON_H

#pragma region UTILS

std::uint64_t
generateUniformDist(std::uint64_t max)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<std::uint64_t> dis(0, max);

    return dis(gen);
}

std::uint64_t
generateNormalDist(std::uint64_t max)
{
    std::uint64_t mean = max / 2;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dis(mean, mean / 32);

    return static_cast<std::uint64_t>(dis(gen));
}

float
generateNormalDistFloat(float mean, float std)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dis(mean, std);

    return static_cast<float>(dis(gen));
}

topkache::Vector2**
prepareResultSampleVector(size_t vector_list_size, std::uint64_t max)
{
    topkache::vector_data_t* vector_data = (topkache::vector_data_t*)malloc(100 * sizeof(float)); // Data dimension is fixed to 100, in float
    std::memset(vector_data, 0, 100 * sizeof(topkache::vector_data_t));

    std::map<topkache::vector_id_t, int> duplicate_check;

    topkache::Vector2** vectors = (topkache::Vector2**)malloc(vector_list_size * sizeof(topkache::Vector2*));

    int i = 0;
    while (i < vector_list_size)
    {
        topkache::vector_id_t vector_id = generateUniformDist(max);
    
        // Unique list of vectors
        if (duplicate_check.find(vector_id) != duplicate_check.end())
            continue;

        vectors[i] = new topkache::Vector2(100 * sizeof(float));        // New vector

        duplicate_check[vector_id] = 1;

        vectors[i]->setVecId(
            vector_id
        );                                          // Set vector id
        vectors[i]->setVecVersion(0);            // Set vector version
        vectors[i]->setVecData(vector_data);     // Set vector data
    
        i++;
    }

    // Fill randomized float-array data
    float* array = (float*)vector_data;
    for (int i = 0; i < 100; i++)
    {
        array[i] = generateNormalDistFloat(0.0, 1.0);
    }

    return vectors;
}

void
removeResultSampleVector(topkache::Vector2** vectors, size_t vector_list_size)
{
    topkache::vector_data_t* data = vectors[0]->getVecData();
    free(data);

    for (int i = 0; i < vector_list_size; i++)
    {
        delete vectors[i];
    }

    delete[] vectors;
}

// 
void
generateSampleQueryVectors(
    std::vector<topkache::Vector2*>& query_vector_list,
    size_t operation_number,
    topkache::result_cache_parameter_t& parameter_info, 
    std::uint64_t max
)
{
    for (int i = 0; i < operation_number; i++)
    {
        topkache::Vector2* query_vector = new topkache::Vector2(parameter_info.vector_data_size);
        
        topkache::vector_id_t vector_id = generateUniformDist(max);
        query_vector->setVecId(vector_id);           // Set vector id
        query_vector->setVecVersion(0);              // Set vector version
        
        float* array = (float*)query_vector->getVecData();
        for (int j = 0; j < parameter_info.vector_dim ; j++)
        {
            float value = generateNormalDistFloat(0.0, 1.0);
            array[j] = value;
        }

        query_vector_list.push_back(query_vector);
    }
}

void
cleanupSampleQueryVectors(
    std::vector<topkache::Vector2*>& query_vector_list
)
{
    for (auto& query_vector : query_vector_list)
    {
        delete query_vector;
    }
}

void
generateSampleFloatVectors(
    std::vector<topkache::float_qvec_t>& query_vector_list,
    size_t vector_list_size,
    size_t vector_dim,
    std::uint64_t max
)
{
    for (int i = 0; i < vector_list_size; i++)
    {
        topkache::float_qvec_t query_vector;
        query_vector.vector_id = generateUniformDist(max);
        query_vector.vector_data = (topkache::vector_data_t*)malloc(vector_dim * sizeof(float));
        query_vector.vector_data_size = vector_dim * sizeof(float);
        query_vector.vector_dim = vector_dim;

        float* array = (float*)query_vector.vector_data;
        for (int j = 0; j < vector_dim; j++)
        {
            float value = generateNormalDistFloat(0.0, 1.0);
            array[j] = value;
        }

        query_vector_list.push_back(query_vector);
    }
}

void
convertVec2ToFVec(
    topkache::Vector2* vector,
    topkache::float_qvec_t& query_vector,
    topkache::result_cache_parameter_t& parameter_info
)
{
    query_vector.vector_id = vector->getVecId();
    query_vector.vector_data = (topkache::vector_data_t*)malloc(parameter_info.vector_data_size);
    query_vector.vector_data_size = parameter_info.vector_data_size;
    query_vector.vector_dim = parameter_info.vector_dim;

    query_vector.conversion_function = [](void* src, size_t src_size, size_t dim, void* dst, std::uint8_t* aux) -> bool
    {
        float* src_array = (float*)src;
        float* dst_array = (float*)dst;

        for (int i = 0; i < dim; i++)
            dst_array[i] = src_array[i];

        return true;
    };

    std::memcpy(
        query_vector.vector_data,
        vector->getVecData(),
        parameter_info.vector_data_size
    );
}

void
cleanupSampleFloatVectors(
    std::vector<topkache::float_qvec_t>& query_vector_list
)
{
    for (auto& query_vector : query_vector_list)
    {
        free(query_vector.vector_data);
    }
}

// Macro to generate test functions
#define TEST_FUNCTION_START(FUNC_NAME) \
bool FUNC_NAME() { \
    std::uint32_t random = generateUniformDist(1000); \
    std::string logger_name = __func__ + std::string("(") + std::to_string(random) + std::string(")"); \
    topkache::Logger logger(logger_name.c_str()); \
    try {

#define TEST_FUNCTION_END() \
    } catch (const std::exception& e) { \
        logger.getLogger()->error("Exception: {}", e.what()); \
        return false; \
    } \
    return true; \
}

#pragma endregion

#endif
