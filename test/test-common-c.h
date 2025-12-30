#include "ResultCache2CWrapper.h"

#ifndef TOPKACHE_TEST_COMMON_C_H
#define TOPKACHE_TEST_COMMON_C_H

#include <stdint.h>   /* uint64_t*/
#include <stdlib.h>   /* rand(), srand() */
#include <math.h>     /* log(), sqrt(), sin(), cos(), M_PI */
#include <time.h>     /* time() */

void
rng_seed_once(void)
{
    static int seeded = 0;
    if (!seeded) {
        /* Fallback to time-based seed if /dev/urandom is unavailable.       */
#if defined(_POSIX_VERSION)
        FILE *f = fopen("/dev/urandom", "rb");
        unsigned int seed = (unsigned int)time(NULL);
        if (f) {
            (void)fread(&seed, sizeof(seed), 1, f);
            fclose(f);
        }
        srand(seed);
#else
        srand((unsigned int)time(NULL));
#endif
        seeded = 1;
    }
}

uint64_t
generate_uniform_dist(uint64_t max)
{
    rng_seed_once();
    /* 64-bit result by stitching two 31-bit rand() outputs.                 */
    uint64_t hi = (uint64_t)rand();          /* 0 … RAND_MAX (≥ 2³¹–1)       */
    uint64_t lo = (uint64_t)rand();
    uint64_t rnd = (hi << 31) ^ lo;

    return rnd % (max + 1);                  /* modulo bias is fine for tests */
}

uint64_t
generate_normal_dist(uint64_t max)
{
    rng_seed_once();
    const double mean = (double)max / 2.0;
    const double std  = mean / 32.0;

    /* Box-Muller transform */
    double u1, u2;
    do { u1 = (rand() + 1.0) / (RAND_MAX + 2.0); } while (u1 <= 1e-9);
    u2 = (rand() + 1.0) / (RAND_MAX + 2.0);

    double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    double val = mean + z0 * std;

    if (val < 0.0)        val = 0.0;
    if (val > (double)max) val = (double)max;
    return (uint64_t)val;
}

float
generate_normal_dist_float(float mean, float stddev)
{
    rng_seed_once();

    double u1, u2;
    do { u1 = (rand() + 1.0) / (RAND_MAX + 2.0); } while (u1 <= 1e-9);
    u2 = (rand() + 1.0) / (RAND_MAX + 2.0);

    double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return (float)(mean + z0 * (double)stddev);
}

static int
id_already_chosen(uint64_t id, uint64_t* chosen, size_t n)
{
    for (size_t i = 0; i < n; ++i)
        if (chosen[i] == id) return 1;
    return 0;
}


char**
prepare_result_sample_vector(size_t vector_list_size, uint64_t max, result_cache_parameter_c_t* param)
{
    /* output array ------------------------------------------------------- */
    char** vectors = malloc(vector_list_size * sizeof(char*));
    if (!vectors) return NULL;

    uint64_t* chosen_ids = malloc(vector_list_size * sizeof(uint64_t));
    if (!chosen_ids) { free(vectors); return NULL; }

    /* build each Vector2 -------------------------------------------------- */
    size_t produced = 0;
    while (produced < vector_list_size) {
        /* 1. unique id */
        uint64_t id = generate_uniform_dist(max);
        if (id_already_chosen(id, chosen_ids, produced))
            continue;

        /* 2. payload buffer */
        float* tmp = malloc(param->vector_data_size);
        if (!tmp) goto oom;

        for (size_t d = 0; d < param->vector_dim; ++d)
            tmp[d] = generate_normal_dist_float(0.0f, 1.0f);

        /* 3. wrap into Vector2 (wrapper copies the bytes) */
        char* v = create_vector_2_c_wrapper(id, param->vector_data_size,
                                            (char*)tmp, 0, 0, 0);
        free(tmp);
        if (!v) goto oom;

        vectors[produced]   = v;
        chosen_ids[produced] = id;
        ++produced;
    }

    free(chosen_ids);
    return vectors;

oom:
    for (size_t i = 0; i < produced; ++i)
        destroy_vector_2_c_wrapper(vectors[i]);
    free(vectors);
    free(chosen_ids);
    return NULL;
}


// void
// register_result_sample_vector(char* query_vector, char** vectors)
// {
//     if (!query_vector || !vectors) return;

//     /* wrap into Vector2 (wrapper copies the bytes) */
//     char* v = create_vector_2_c_wrapper(0, 0, query_vector);
//     if (!v) return;

//     /* register the vector */
//     for (size_t i = 0; i < 96; ++i)
//         vectors[i] = v;
// }


void
remove_result_sample_vector(char** vectors, size_t vector_list_size)
{
    if (!vectors) return;
    for (size_t i = 0; i < vector_list_size; ++i)
        destroy_vector_2_c_wrapper(vectors[i]);
    free(vectors);
}

char**
generate_sample_query_vectors(size_t count,
                              const result_cache_parameter_c_t* param,
                              uint64_t max)
{
    if (!param || !count) return NULL;

    char**    out   = calloc(count, sizeof(char*));
    uint64_t* ids   = calloc(count, sizeof(uint64_t));
    if (!out || !ids) goto oom;

    size_t produced = 0;
    while (produced < count) {
        uint64_t id = generate_uniform_dist(max);

        /* ensure id uniqueness */
        int dup = 0;
        for (size_t j = 0; j < produced; ++j)
            if (ids[j] == id) { dup = 1; break; }
        if (dup) continue;

        /* allocate & fill raw payload ----------------------------------- */
        float* tmp = malloc(param->vector_data_size);
        if (!tmp) goto oom;

        for (size_t d = 0; d < param->vector_dim; ++d)
            tmp[d] = generate_normal_dist_float(0.0f, 1.0f);

        /* create Vector2 wrapper --------------------------------------- */
        char* v = create_vector_2_c_wrapper(id,
                                            param->vector_data_size,
                                            (char*)tmp, 0, 0, 0);
        free(tmp);
        if (!v) goto oom;

        out[produced] = v;
        ids[produced] = id;
        ++produced;
    }

    free(ids);
    return out;

oom:
    if (out) {
        for (size_t i = 0; i < count; ++i)
            if (out[i]) destroy_vector_2_c_wrapper(out[i]);
    }
    free(out);
    free(ids);
    return NULL;
}

void
cleanup_sample_query_vectors(char** list, size_t count)
{
    if (!list) return;
    for (size_t i = 0; i < count; ++i)
        destroy_vector_2_c_wrapper(list[i]);
    free(list);
}

#endif