
#include "Vector2.hh"

#ifndef TOPKACHE_INSERTLOG_H
#define TOPKACHE_INSERTLOG_H

// #include <list>
#include <queue>
#include <vector>
#include <atomic>

#include <list>

#include <boost/unordered/concurrent_flat_map.hpp>

namespace topkache
{

    enum class WriteLogType
    {
        INSERT_LOG_TYPE_INSERT              = 0,
        INSERT_LOG_TYPE_DELETE
    };

    typedef std::uint64_t                   epoch_t;
    typedef struct WriteLogEntry
    {
        epoch_t                             epoch;
        Vector2                             vector;

        std::uint32_t                       reference_count;

        struct WriteLogEntry*               prev;
        struct WriteLogEntry*               next;
        
        WriteLogEntry(size_t vector_list_size) noexcept
            : epoch(0), vector(vector_list_size), reference_count(0),
                prev(nullptr), next(nullptr)
        {

        }
        ~WriteLogEntry() noexcept
        {
            vector.freeVecData();
        }
    } write_log_entry_t;

    typedef struct WriteLogStats
    {
        std::uint32_t                       insert_count;
        std::uint32_t                       delete_count;
        std::uint32_t                       sp_checked_count; // Only counts for slow paths

        inline void clear() noexcept
        {
            insert_count                    = 0;
            delete_count                    = 0;
            sp_checked_count                = 0;
        }

        WriteLogStats() noexcept
        {
            clear();
        }

    } write_log_stats_t;

    typedef struct CloseCandidates
    {
        float                               distance;
        write_log_entry_t*                  entry;
    } close_candidates_t;
    

    /* Function that do distance calculation
     */
    typedef std::function<float(vector_data_t*, vector_data_t*, size_t)> 
                                            distance_function_t;
    typedef std::function<void(vector_id_t, vector_data_t*, size_t, uint64_t, uint64_t)>
                                            result_conversion_function_t;
    typedef std::uint8_t                    rev_centry_t;

    class WriteLog
    {
    private:

        const size_t                        scan_threshold;
        const size_t                        force_age;

        write_log_stats_t                   stats;

        epoch_t                             log_epoch;
        size_t                              log_delta_bound;

        size_t                              log_current_size;

        std::atomic<bool>                   log_lock;
        write_log_entry_t*                  log_head;
        write_log_entry_t*                  log_tail;

        std::list<rev_centry_t*>*           sealed;
        std::list<rev_centry_t*>*           unsealed;

        size_t                              heartbeat;
        

        struct EntryLocation
        {
            std::list<rev_centry_t*>*               list;
            std::list<rev_centry_t*>::iterator      it;
        };

        boost::concurrent_flat_map<vector_id_t, write_log_entry_t*>
                                            log_map;
        boost::concurrent_flat_map<rev_centry_t*, struct EntryLocation> 
                                            centry_map;
        
        void                                _lock() noexcept;
        bool                                _tryLock() noexcept;
        void                                _unlock() noexcept;

        void                                _insertLEntry(Vector2& vector, size_t vector_data_size) noexcept;

    public:

        WriteLog(size_t delta_bound) noexcept;
        virtual ~WriteLog() noexcept;

        void                                insertLEntry(Vector2& vector, size_t vector_data_size) noexcept;

        void                                addToUnsealed(rev_centry_t* cache_entry) noexcept;
        void                                arrangeSealed(rev_centry_t* observed_front) noexcept;
        write_log_entry_t*                  sweepLEntry(vector_data_t* query_vector_data, 
                                                size_t query_vector_data_size, float entry_max_dist,
                                                std::vector<close_candidates_t>& found_entries,
                                                distance_function_t distance_function,
                                                write_log_entry_t* scan_start
                                                ) noexcept;

        bool                                deregFromList(rev_centry_t* cache_entry) noexcept;
        bool                                safeTrims(write_log_entry_t* scan_start) noexcept;
        void                                deleteLEntry(Vector2& vector) noexcept;

        void                                clearLog() noexcept;

        std::string                         toString() noexcept;

        write_log_entry_t*                  getHead() noexcept;
        write_log_entry_t*                  getTail() noexcept;

        rev_centry_t*                       getSealedNext() noexcept;
        const size_t                        getScanThreshold() noexcept;

        write_log_stats_t                   getStats() noexcept;

        size_t                              getMappedSize() noexcept;
    };

    float                                   sampleL2Dist(vector_data_t* vector1, vector_data_t* vector2, size_t dimension) noexcept;

    class DdfWriteLog
    {
    private:
        const size_t                        scan_threshold;
        const size_t                        k;

        write_log_stats_t                   stats;
        size_t                              log_current_size;
        
        // Delta-bounded Risks
        const double                        allowed_risk;
        epoch_t                             position;

        struct Score 
        {
            double                          curr_risk;
            double                          total_ddf;
            double                          avg_ddf;
            double                          total_unseen;
            double                          total_centry;
        };
        struct Score                        risk_status;

        std::atomic<bool>                   log_lock;
        write_log_entry_t*                  log_head;
        write_log_entry_t*                  log_tail;

        boost::concurrent_flat_map<vector_id_t, write_log_entry_t*>
                                            log_map;

        struct EntryLocation
        {
            std::list<rev_centry_t*>::iterator      it;
        };
        boost::concurrent_flat_map<rev_centry_t*, struct EntryLocation>
                                            centry_map;
        std::list<rev_centry_t*>*           rr_list;
        
        void                                _lock() noexcept;
        bool                                _tryLock() noexcept;
        void                                _unlock() noexcept;

        void                                _insertLEntry(Vector2& vector, size_t vector_data_size) noexcept;
        void                                _increaseCEntryDdf(double ddf, size_t unseen, size_t curr_n_centry) noexcept;
        void                                _decreaaseCEntryDdf(double ddf, size_t unseen, size_t curr_n_centry) noexcept;

    public:
        DdfWriteLog(size_t k, size_t scan_thresh, double risk_thresh) noexcept;
        virtual ~DdfWriteLog() noexcept;

        void                                insertLEntry(Vector2& vector, size_t vector_data_size) noexcept;
        
        void                                addToRr(rev_centry_t* cache_entry) noexcept;
        rev_centry_t*                       getNextRr() noexcept;
        bool                                removeFromRr(rev_centry_t* cache_entry) noexcept;

        bool                                safeTrims(write_log_entry_t* scan_start) noexcept;

        void                                increaseCEntryDdf(double ddf, size_t unseen, size_t curr_n_centry) noexcept;
        void                                decreaseCEntryDdf(double ddf, size_t unseen, size_t curr_n_centry) noexcept;

        write_log_entry_t*                  sweepLEntryBatch(
                                                vector_data_t* query_vector_data, 
                                                size_t query_vector_data_size, float entry_max_dist,
                                                std::vector<close_candidates_t>& found_entries,
                                                distance_function_t distance_function,
                                                write_log_entry_t* scan_start
                                            ) noexcept;

        void                                clearLog() noexcept;

        write_log_entry_t*                  getHead() noexcept;
        write_log_entry_t*                  getTail() noexcept;

        bool                                needRunSlowPath() noexcept;
        std::uint64_t                       getDistance(write_log_entry_t* curr_entry) noexcept;

        write_log_stats_t                   getStats() noexcept;
        size_t                              getMappedSize() noexcept;
        std::string                         toString() noexcept;
    };
}

#endif