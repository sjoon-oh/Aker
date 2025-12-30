// #include "VectorPool.hh"

#include "Vector2.hh"
#include "VectorPool2.hh"
#include "EvictionStrategy.hh"

#include "Commons.hh"

#include "WriteLog.hh"
#include "ApproxFilter.hh"
#include "Timer.hh"

#ifndef TOPKACHE_RESULTCACHE2_H
#define TOPKACHE_RESULTCACHE2_H


#include <ctime>
#include <cassert>
#include <sys/stat.h>

#include <list>
#include <map>

#include <string>
#include <fstream>
#include <iomanip>

#include <iostream>

#include <atomic>
#include <mutex>
#include <memory>
#include <boost/lockfree/queue.hpp>

#include <faiss/impl/IDSelector.h>

namespace topkache
{
    typedef std::uint8_t                            pool_operation_status_t;
    typedef std::uint8_t                            pool_operation_type_t;
    
    enum class PoolOperationStatus : pool_operation_status_t
    {
        POOL_OPERATION_STATUS_PENDING               = 0,
        POOL_OPERATION_STATUS_DONE
    };
    
    enum class PoolOperationType : pool_operation_type_t
    {
        POOL_OPERATION_ALLOCATE                     = 0,
        POOL_OPERATION_DEALLOCATE
    };

    /* Result Cache Entry
     */
    typedef std::uint8_t                            result_cache_entry_status_t;

    // Used in locks
    enum
    {
        RESULT_CACHE_ENTRY_STATUS_VALID             = 0,            // Entry is valid
        RESULT_CACHE_ENTRY_STATUS_INMOD                             // Entry is being modified
    };

    /* Result Cache Lookup Entry
     *  Every entry is maintained by the result cache.
     */
    typedef struct ResultCacheLookupEntry2
    {
        result_cache_entry_status_t                 entry_status;                   // Entry status
        std::int32_t                                version;                        // Version of the entry
        Vector2*                                    query_vector;                   // Query vector data
        std::uint32_t                               vector_list_size;               // Size of the vector list
        Vector2**                                   vector_slot_ref_list;           // Reference to the vector list in the pool

        float                                       threshold;                      // Per-query threshold
        float                                       min_distance;                   // Minimum distance from the query vector
        float                                       max_distance;
        struct ResultCacheLookupEntry2*             prev;
        struct ResultCacheLookupEntry2*             next;
        float                                       risk_factor;
        write_log_entry_t*                          checkpoint;                     // Log iterator

    } result_cache_entry_t;


    /* Result Cache Lookup Table
     */
    typedef struct ResultCacheLookupTable
    {
        boost::concurrent_flat_map<
            vector_id_t, result_cache_entry_t*>     map;
    } result_cache_table_t;


    typedef struct ResultCacheLookupTable2
    {
        boost::concurrent_flat_map<
            vector_id_t, result_cache_entry_t*>    map;
    } result_cache_table2_t;

    class ResultCacheStats
    {
    public:
        size_t                                      cache_hit;
        size_t                                      cache_miss;
        size_t                                      cache_invalid_detect;
        size_t                                      cache_evict;
        size_t                                      cache_sim_hit;

        std::map<std::string, std::vector<ElapsedPair>> stat_level_0;   // Highest view of latency measurement
        std::map<std::string, std::vector<ElapsedPair>> stat_level_1;   // Detailed view of latency measurement
        std::map<std::string, std::vector<ElapsedPair>> stat_level_2;   // Detailed view of latency measurement

        std::vector<float>                          cache_tot_hits;
        std::vector<float>                          cache_exact_hits;

        std::vector<size_t>                         apprx_added;
        std::vector<size_t>                         apprx_nrepr;

        std::string                                 directory_path;

        virtual inline void recordHitHistory() noexcept 
        {
            float hitratio_exact = (cache_hit) / (float)(cache_hit + cache_sim_hit + cache_miss);
            float hitratio_total = (cache_hit + cache_sim_hit) / (float)(cache_hit + cache_sim_hit + cache_miss);

            cache_exact_hits.push_back(hitratio_exact);
            cache_tot_hits.push_back(hitratio_total);
        }

        virtual void clear() noexcept;

        ResultCacheStats() noexcept;
        virtual ~ResultCacheStats();

        virtual void exportAll() noexcept;
        virtual void printAll() noexcept;
    };
    
    typedef ResultCacheStats result_cache_stats_t; 


    class ResultCache2
    {
    private:

        result_cache_parameter_t                    parameter;
        result_cache_stats_t                        stats;

        size_t                                      repr_entry_cnt;

        std::unique_ptr<result_cache_table2_t>      lookup_table;
        std::unique_ptr<VectorPool2>                vector_pool;
        std::unique_ptr<EvictionStrategy>           eviction_strategy;
        // std::unique_ptr<ApproxFilterDualHNSW>       apprx_filter;
        std::unique_ptr<ApproxFilterDualHNSW2>      apprx_filter;
        
        std::uint32_t                               evict_entry_cnt;

        // std::unique_ptr<WriteLog>                   write_log;
        std::unique_ptr<RfWriteLog>                 rf_write_log;

        std::uint32_t                               try_read_cnt;

        std::atomic<bool>                           cache_lock;

        bool                                        used;

        /* Locks to block all the operations
         */
        void                                        _test_existance(vector_id_t vector_id);

        result_cache_entry_t*                       _copy_cache_entry(
                                                        result_cache_entry_t* entry) noexcept;

        void                                        _lock() noexcept;
        bool                                        _tryLock() noexcept;
        void                                        _unlock() noexcept;

        bool                                        _tryLockCEntry(result_cache_entry_t* entry) noexcept;
        bool                                        _unlockCEntry(result_cache_entry_t* entry) noexcept;

        bool                                        _needEvict(size_t vector_list_size) noexcept;
        void                                        _evictVecs2(size_t to_evicts, std::vector<vector_id_t>& evicted_list) noexcept;

        result_cache_entry_t*                       _getCEntry(vector_id_t vector_id) noexcept;
        void                                        _getSimCEntryEx(
                                                        float_qvec_t query_vector_data,
                                                        distance_function_t distance_function,
                                                        std::vector<faiss::idx_t>& labels,
                                                        std::vector<float>& distances
                                                        ) noexcept;

        bool                                        _linkSimCEntryLin(
                                                        result_cache_entry_t* allocated_entry,
                                                        distance_function_t distance_function) noexcept;

        bool                                        _linkSimCEntryApprx(
                                                        result_cache_entry_t* allocated_entry,
                                                        float_qvec_t query_vector_data,
                                                        distance_function_t distance_function) noexcept;

        bool                                        _handleInvalidCEntry(result_cache_entry_t* entry) noexcept;

        void                                        _updateWLEntryFastPath(
                                                        float_qvec_t write_vector, 
                                                        distance_function_t distance_function,
                                                        result_conversion_function_t result_conversion_function = nullptr) noexcept;
        void                                        _incrBatchUpdateWLog2(
                                                        distance_function_t distance_function,
                                                        result_conversion_function_t result_conversion_function = nullptr) noexcept;

    public:

        ResultCache2(result_cache_parameter_t& parameter_info) noexcept;
        // ResultCache2(size_t vector_pool_size, size_t free_size);
        virtual ~ResultCache2();

        result_cache_entry_t*                       makeCEntry(
                                                        Vector2* query_vector,
                                                        std::uint32_t list_size, 
                                                        Vector2** vector_local_reference_list) noexcept;

        void                                        freeCEntry(result_cache_entry_t* entry) noexcept;

        // 
        // Before fetching the vectors, we first search for the existing search result by calling
        // getCacheEntry. If the entry does not exist, we allocate the entry by calling makeCEntry.
        // Using the allocated entry, we call insertCEntryLin to insert the entry into the cache.
        // 
        // In getCacheEntry, the entry is locked and should be unlocked by releaseCEntry.
        // 
        // When inserting the cache entry, it is possible that the entry is already inserted by another, or 
        // the entry may be semantically equivalent to the existing entry. In this case, the entry is not inserted
        // but the existing entry is used. The function _linkSimCEntryLin is used to check the semantic 
        // equivalence. However, if the distance function is not provided, the semantic equivalence is not checked.
        //
        
        // exGetCEntryPassive : Even if some vectors are deleted, the entry is returned.
        //  The caller should check the validity of the entry.
        // exGetCEntryActive : Only the active entries are returned.
        //  If the vectors are deleted in the entry, it checks the validity and rearrange the vector_slot_ref_list.
        result_cache_entry_t*                      exGetCEntryPassive(vector_id_t vector_id) noexcept;
        result_cache_entry_t*                      exGetCEntryActive(vector_id_t vector_id, bool& is_invalid) noexcept;
        
        // simGetCEntryxx : First, it checks whether the vector id is contained in the map, 
        //  and then returns the entry if found. If not found, it searches the approximate filter.
        //  The first float_qvec_t contains the query id.
        // result_cache_entry_t*                      simGetCEntryPassive(
        //                                                 float_qvec_t query_vector_data,
        //                                                 bool& similar_entry,
        //                                                 distance_function_t distance_function) noexcept;
        result_cache_entry_t*                      simGetCEntry(
                                                        float_qvec_t query_vector_data,
                                                        bool& similar_entry, bool& is_invalid,
                                                        distance_function_t distance_function,
                                                        bool exhaustive_search = false
                                                    ) noexcept;

        // When inserting an entry, we have two options: Exhaustive and Approximate
        //  Two functions both does two things:
        //  1) Find the similar entry in the cache
        //  2) If the similar entry is found, the entry is linked to the found entry.
        //      If not found, the entry is inserted directly to the map.
        // An exhaustive search scans all the data in the cache, while the approximate search
        //  uses the approximate filter to find the similar entry.
        bool                                        insertCEntry2(
                                                        vector_id_t vector_id, result_cache_entry_t* entry, 
                                                        float_qvec_t query_vector_data) noexcept;

        bool                                        linkCEntry(
                                                        result_cache_entry_t* allocated_entry,
                                                        vector_id_t found_id) noexcept;

        void                                        markVecDeleted(vector_id_t vector_id) noexcept;

        void                                        insertWLEntry3(float_qvec_t write_vector,
                                                        distance_function_t distance_function,
                                                        result_conversion_function_t result_conversion_function = nullptr
                                                        ) noexcept;
        void                                        consumeAgedWLEntry(
                                                        distance_function_t distance_function,
                                                        result_conversion_function_t result_conversion_function = nullptr) noexcept;

        void                                        resetCache() noexcept;

        // Test functions
        //
        void                                        stressTestInvalidateRandom(float percent) noexcept;

        // These functions are only for testing purpose. 
        // 
        void                                        getPooledVecs(std::vector<Vector2*>& pooled_list) noexcept;

        std::string                                 toString() noexcept;
        std::string                                 toStringCsv() noexcept;
        void                                        exportPrintAll() noexcept;
    };

}

#endif
