// 
// Author: Sukjoon Oh
// This header defines two flavours of the similarity filter that TopKache uses
// before consulting its full result‑set cache.
//
//  1. ApproxFilterHNSW   – a thin wrapper around a single `IndexHNSWFlat` that
//     stores *representative* (query) vectors, not the full corpus.  It supports
//     inserts, lazy deletions (tomb‑stones), similarity search and background
//     compaction.
//
//  2. ApproxFilterDualHNSW – maintains **two** HNSW instances so that the cache
//     can hot‑swap: while one index is active for queries / inserts, the other
//     can be rebuilt or compacted without blocking the fast path.

#include <vector>
#include <numeric>

#include <boost/unordered/concurrent_flat_map.hpp>
#include <faiss/IndexLSH.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIDMap.h>
#include <faiss/impl/IDSelector.h>

#include "Commons.hh"
#include "Vector2.hh"

#ifndef TOPKACHE_APPROXFILTER_H
#define TOPKACHE_APPROXFILTER_H

namespace topkache
{

    // Signature for user‑supplied routines that convert a raw `vector_data_t`
    // blob into a contiguous float array usable by FAISS.  The `aux` pointer
    // allows the caller to reuse scratch space to avoid heap churn.
    typedef std::function<
        bool(vector_data_t*, size_t, std::uint32_t, float*, std::uint8_t*)>
                                                    conversion_function_t;

    // FloatQueryVector – simple POD holding a converted query / representative
    // vector plus metadata needed inside the filter layer.
    typedef struct FloatQueryVector
    {
        vector_id_t                                 vector_id;
        vector_data_t*                              vector_data;
        std::uint32_t                               vector_dim;
        std::uint32_t                               vector_data_size;

        std::uint64_t                               aux_data_1; // e.g., distance
        std::uint64_t                               aux_data_2; // e.g., tid

        // Conversion function to float*
        conversion_function_t                       conversion_function;
        std::uint8_t*                               aux;

    } float_qvec_t;


    typedef struct ApproxFilterStat
    {
        size_t                                      counts;
        double                                      refresh_latency;
    } apprx_filter_stat_t;

    // This class owns one FAISS `IndexHNSWFlat` and two bidirectional maps that
    // translate between FAISS internal sequence ids and external `vector_id_t`s.
    //
    // Thread‑safety: a simple spin lock (`filter_lock`) protects the critical
    // sections.  
    class ApproxFilterHNSW
    {
    private:
        std::atomic<bool>                           filter_lock;
        result_cache_parameter_t                    parameter;

        size_t                                      eff_repr_vec_num;

        std::vector<float>                          representative_vecs;
        
        boost::concurrent_flat_map<
            faiss::idx_t, vector_id_t>              vector_id_map;
        boost::concurrent_flat_map<
            vector_id_t, faiss::idx_t>              reverse_vector_id_map;
        
        std::unique_ptr<faiss::IndexHNSWFlat>       hnsw_index;

        void                                       __lock() noexcept;
        bool                                       __tryLock() noexcept;
        void                                       __unlock() noexcept;

    public:
        ApproxFilterHNSW(result_cache_parameter_t parameter_info, faiss::MetricType metric = faiss::METRIC_L2) noexcept;
        virtual ~ApproxFilterHNSW() noexcept = default;

        void                                        addVec(float_qvec_t query) noexcept;
        // void                                        deleteVector(vector_id_t vector_id) noexcept;
        int                                         deleteVecs(std::vector<vector_id_t>& vector_id_list) noexcept;
        
        void                                        searchSimVecs(
                                                            const float* x,
                                                            faiss::idx_t k, 
                                                            float* distances,
                                                            faiss::idx_t* labels) noexcept;

        bool                                        seqToVecId(faiss::idx_t seq_id, vector_id_t& vector_id) noexcept;

        void                                        compressIndex() noexcept;

        void                                        clear() noexcept;

        std::string                                 toString() noexcept;

        size_t                                      getNumReprVecs() noexcept;
        size_t                                      getReprVecsArrSize() noexcept;
    };

    class ApproxFilterHNSW2
    {
    private:
        std::atomic<bool>                           filter_lock;
        result_cache_parameter_t                    parameter;
       
        size_t                                      add_count;
        boost::concurrent_flat_map<
            vector_id_t, bool>                      reg_map; // Maps vector_id to bool (registered or not)
        std::unique_ptr<faiss::IndexIDMap>          hnsw_index_wrapper;

        void                                       __lock() noexcept;
        bool                                       __tryLock() noexcept;
        void                                       __unlock() noexcept;

    public:
        ApproxFilterHNSW2(result_cache_parameter_t parameter_info, faiss::MetricType metric = faiss::METRIC_L2) noexcept;
        virtual ~ApproxFilterHNSW2() noexcept = default;

        void                                        addVec(float_qvec_t query) noexcept;
        int                                         deleteVecs(std::vector<vector_id_t>& vector_id_list) noexcept;
        
        void                                        searchSimVecs(
                                                            const float* x,
                                                            faiss::idx_t k, 
                                                            float* distances,
                                                            faiss::idx_t* labels) noexcept;

        bool                                        isRegistered(vector_id_t vector_id) noexcept;
        size_t                                      getReprVecNum() noexcept;
        size_t                                      getAddedCounts() noexcept;

        void                                        clear() noexcept;

        std::string                                 toString() noexcept;
    };

    class ApproxFilterDualHNSW
    {
    private:
        result_cache_parameter_t                    parameter;
        std::atomic<bool>                           filter_lock;

        const size_t                                num_filters = 2;
        ApproxFilterHNSW**                          filter_list;

        void                                        __lock() noexcept;
        bool                                        __tryLock() noexcept;
        void                                        __unlock() noexcept;

        void                                        __switchFilter() noexcept;


    public:
        ApproxFilterDualHNSW(result_cache_parameter_t parameter_info, faiss::MetricType metric = faiss::METRIC_L2) noexcept;
        virtual ~ApproxFilterDualHNSW() noexcept;

        void                                        addVec(float_qvec_t query) noexcept;
        int                                         deleteVecs(std::vector<vector_id_t>& vector_id_list) noexcept;
        void                                        searchSimVecs(
                                                            const float* x,
                                                            faiss::idx_t k, 
                                                            float* distances,
                                                            faiss::idx_t* labels,
                                                            int* filter_id
                                                        ) noexcept;
        bool                                        seqToVecId(faiss::idx_t seq_id, int filter_id, vector_id_t& vector_id) noexcept;

        void                                        clear() noexcept;

        std::string                                 toString() noexcept; 

        size_t                                      getNumReprVecs() noexcept;
        size_t                                      getReprVecsArrSize() noexcept;

        bool                                        needSwitch() noexcept;
    };

#define INVALID_DISTANCE    std::numeric_limits<float>::max()

    class ApproxFilterDualHNSW2
    {
    private:
        result_cache_parameter_t                    parameter;
        std::atomic<bool>                           filter_lock;

        const size_t                                num_filters = 2;
        ApproxFilterHNSW2**                         filter_list;

        void                                        __lock() noexcept;
        bool                                        __tryLock() noexcept;
        void                                        __unlock() noexcept;

        void                                        __switchFilter() noexcept;

    public:
        ApproxFilterDualHNSW2(result_cache_parameter_t parameter_info) noexcept;
        virtual ~ApproxFilterDualHNSW2() noexcept;

        void                                        addVec(float_qvec_t query) noexcept;
        int                                         deleteVecs(std::vector<vector_id_t>& vector_id_list) noexcept;
        void                                        searchSimVecs(
                                                            const float* x,
                                                            faiss::idx_t k,
                                                            float* distances,
                                                            faiss::idx_t* labels 
                                                        ) noexcept;

        void                                        clear() noexcept;

        size_t                                      getReprVecNum() noexcept;
        size_t                                      getAddedCounts() noexcept;
        std::string                                 toString() noexcept;

        bool                                        needSwitch() noexcept;
    };

}

#endif