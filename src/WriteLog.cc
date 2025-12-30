#include "WriteLog.hh"
#include "core/ResultCache2.hh"

#include <cmath>
#include <cassert>
#include <cstdlib>
#include <ctime>

#include <algorithm>

// Helper
static inline 
topkache::result_cache_entry_t* 
as_centry(void* entry) noexcept 
{
    return reinterpret_cast<topkache::result_cache_entry_t*>(entry);
}

static inline
topkache::rev_centry_t*
as_uint8_ptr(void* entry) noexcept
{
    return reinterpret_cast<topkache::rev_centry_t*>(entry);
}

namespace topkache
{
    float
    sampleL2Dist(vector_data_t* vector1, vector_data_t* vector2, size_t dimension) noexcept
    {
        float distance = 0.0;

        float* first_data = (float*)vector1;
        float* second_data = (float*)vector2;

        for (size_t i = 0; i < dimension; i++)
        {
            float diff = first_data[i] - second_data[i];
            distance += diff * diff;

            // printf("    %f - %f = %f\n", first_data[i], second_data[i], diff);
        }

        return std::sqrt(distance);
    }

    // 
    // -- RfWriteLog
    RfWriteLog::RfWriteLog(size_t in_topk, size_t scan_thresh, double risk_thresh) noexcept
        : k(in_topk), scan_threshold(scan_thresh), allowed_risk(risk_thresh), position(0)
    {
        log_lock.store(0);
        
        log_current_size = 0;

        // score values

        log_head = nullptr;
        log_tail = nullptr;

        std::memset(&risk_status, 0, sizeof(Score));
        rr_list = new std::list<rev_centry_t*>();
    }

    RfWriteLog::~RfWriteLog() noexcept
    {
        write_log_entry_t* log_entry = log_head;
        while (log_entry != nullptr)
        {
            write_log_entry_t* next_log_entry = log_entry->next;
            delete log_entry;

            log_entry = next_log_entry;
        }
    }

    void
    RfWriteLog::_lock() noexcept
    {
        while (_tryLock() == false)
            ;
    }

    bool
    RfWriteLog::_tryLock() noexcept
    {
        bool expected = false;
        bool desired = true;

        return log_lock.compare_exchange_strong(expected, desired, std::memory_order_acquire);
    }

    void
    RfWriteLog::_unlock() noexcept
    {
        log_lock.store(false, std::memory_order_release);
    }

    void
    RfWriteLog::_insertLEntry(Vector2& vector, size_t vector_data_size) noexcept
    {
        write_log_entry_t* log_entry = nullptr;
        vector_id_t vector_id = vector.getVecId();

        bool insert_success = log_map.try_emplace_or_visit(
            vector_id, nullptr,
            [&](const auto& pair) { log_entry = pair.second; }
        );

        position++;

        // Make a log entry
        if (log_entry == nullptr)
        {
            log_entry = new write_log_entry_t(vector_data_size);

            log_entry->vector.lock();

            log_entry->vector.setVecId(vector_id);
            log_entry->vector.setVecVersion(vector.getVecVersion());

            log_entry->vector.setAuxData1(vector.getAuxData1());
            log_entry->vector.setAuxData2(vector.getAuxData2());

            log_entry->vector.decreaseRefCount();
            
            log_entry->epoch = position;
            
            std::memcpy(log_entry->vector.getVecData(), vector.getVecData(), vector_data_size);

            log_map.visit(vector_id, [&](auto& pair) { pair.second = log_entry; });

            // Link the log entry to the log list
            if (log_head == nullptr)
            {
                log_head = log_entry;
                log_tail = log_entry;
            }
            else
            {
                log_entry->prev = log_tail;
                log_tail->next = log_entry;
                log_tail = log_entry;
            }

            log_entry->vector.unlock();
        }

        log_current_size++;
    }

    inline double
    estimatedRiskyWLE(double avg_rf, size_t total_unseen, size_t total_entry) noexcept
    {
        // With each risk (rf), we can estimate the number of WLEs that can be risky.
        // With having average risk, and total_unseen entries, we have:
        //    avg_rf * (total_unseen / total_entry), which is the average number of risky WLEs per cache entry.
        // This implies the number of WLEs that can affect the in-top-k inserts.
        // For example, we have avg_rf - 0.8 (the 80% risk of touching in-top-k),
        // and total_unseen - 2367, total_entry - 500, then we have:
        //    0.8 * (2367 / 500) = 3.788, which means that we can expect only 3.788 entries to be risky.
        // (normally, we can expect the number of unseeen will exceed the number of entries)
        // In this case, we can expect 3.788 entries to be risky.
        // Let's say we have 200 WLEs in the log.
        // The total ratio of risky WLEs is 3.788 / 200 = 0.01894, which implies that 1.894% of the WLEs
        // can touch the in-top-k entries.

        // So, the curr_risk is the ratio of risky WLEs to the log size.
        // If this ratio exceeds some predefined threshold, we run the slow path to reduce:
        // the number of total unseen entries, so that the risk is reduced.

        return avg_rf * (total_unseen / total_entry) ;
    }

    void
    RfWriteLog::_increaseCEntryRf(double rf, size_t unseen, size_t curr_n_centry) noexcept
    {
        risk_status.total_rf += rf;
        risk_status.total_unseen += unseen;

        assert((rf >= 0) && (rf <= 1));

        risk_status.total_centry = curr_n_centry;
        risk_status.avg_rf = risk_status.total_rf / risk_status.total_centry;

        // Estimated number of IDs in the worst-case
        double est_risky_wle = estimatedRiskyWLE(risk_status.avg_rf, risk_status.total_unseen, risk_status.total_centry);
        risk_status.curr_risk = est_risky_wle / log_current_size;
    }

    void
    RfWriteLog::_decreaaseCEntryRf(double rf, size_t unseen, size_t curr_n_centry) noexcept
    {
        risk_status.total_rf -= rf;
        risk_status.total_unseen -= unseen;

        assert((rf >= 0) && (rf <= 1));

        risk_status.total_centry = curr_n_centry;
        risk_status.avg_rf = risk_status.total_rf / risk_status.total_centry;

        // Estimated number of IDs in the worst-case
        double est_risky_wle = estimatedRiskyWLE(risk_status.avg_rf, risk_status.total_unseen, risk_status.total_centry);
        risk_status.curr_risk = est_risky_wle / log_current_size;
    }

    void
    RfWriteLog::insertLEntry(Vector2& vector, size_t vector_data_size) noexcept
    {
        _lock();
        _insertLEntry(vector, vector_data_size);
        stats.insert_count++;
        _unlock();
    }

    void
    RfWriteLog::addToRr(rev_centry_t* cache_entry) noexcept
    {
        _lock();

        rr_list->push_back(cache_entry);

        struct EntryLocation location = { .it = --(rr_list->end()) };
        centry_map.try_emplace_or_visit(cache_entry, location, [](auto& pair) { });

        _unlock();
    }

    rev_centry_t*
    RfWriteLog::getNextRr() noexcept
    {
        _lock();
        if (rr_list->size() == 0)
            return nullptr;

        rev_centry_t* front = rr_list->front();

        // Move to the back using splice
        if (rr_list->size() == 1)
        {
            // If there is only one element, we do not need to move it.
            _unlock();
            return front;
        }

        rr_list->splice(rr_list->end(), *rr_list, rr_list->begin());

        _unlock();
        return front;
    }

    bool
    RfWriteLog::removeFromRr(rev_centry_t* cache_entry) noexcept
    {
        _lock();

        // int visited = centry_map.visit(
        //     cache_entry, [&](const auto& pair) { 
        //         rr_list->erase(pair.second.it); 
        //     });

        std::list<rev_centry_t*>::iterator it;

        int visited = centry_map.visit(
            cache_entry, [&](const auto& pair) { 
                it = pair.second.it;
            });

        assert(visited != 0); 
        rr_list->erase(it);

        centry_map.erase(cache_entry);

        _unlock();
        return true;
    }

    bool
    RfWriteLog::safeTrims(write_log_entry_t* scan_start) noexcept
    {
        _lock();
        write_log_entry_t* log_entry = scan_start;

        // Safe trim should always start from the log_head, 
        // since it is the only starting place it can be trimmed
        if (log_entry == nullptr)
        {
            _unlock();
            return false;
        }

        while (log_entry != log_tail)
        {
            log_entry->vector.lock();
            if (log_entry->vector.getRefCount() == 0)
            {
                write_log_entry_t* next_log_entry = log_entry->next;

                // 
                // Get vector ID of the log entry
                vector_id_t vector_id = log_entry->vector.getVecId();

                // Remove the entry from the map
                log_map.erase(vector_id);
                log_entry->vector.unlock();

                delete log_entry;

                log_entry = next_log_entry;
                log_head = log_entry;
                log_head->prev = nullptr;
                log_current_size--;

                stats.delete_count++;
            }
            else
            {
                log_entry->vector.unlock();
                break;
            }
        }

        // At this point, log_entry is either 
        // log_tail or somewhere in the middle of the list.
        _unlock();

        return true;
    }

    void
    RfWriteLog::increaseCEntryRf(double rf, size_t unseen, size_t curr_n_centry) noexcept
    {
        _lock();
        _increaseCEntryRf(rf, unseen, curr_n_centry);
        _unlock();
    }

    void
    RfWriteLog::decreaseCEntryRf(double rf, size_t unseen, size_t curr_n_centry) noexcept
    {
        _lock();
        _decreaaseCEntryRf(rf, unseen, curr_n_centry);
        _unlock();
    }

    write_log_entry_t*
    RfWriteLog::sweepLEntryBatch(
        vector_data_t* query_vector_data,
        size_t vector_dim,
        float entry_max_dist,
        std::vector<close_candidates_t>& found_entries,
        distance_function_t distance_function,
        write_log_entry_t* scan_start) noexcept {
        _lock();

        write_log_entry_t* log_entry = scan_start;
        write_log_entry_t* prev_log_entry = nullptr;

        // If the scan start is null, start from the head
        if (log_entry == nullptr)
            log_entry = log_head;

        else
        {
            assert(log_entry->vector.getRefCount() != 0);

            log_entry->vector.lock();
            log_entry->vector.decreaseRefCount();
            log_entry->vector.unlock();
        }

        float dist = 0.0;

        // Maximum scan window size is by 32.
        int scanned = 0;
        for (int scan_count = 0; scan_count < scan_threshold; scan_count++)
        {
            if (log_entry == nullptr)
                break;

            log_entry->vector.lock();
            dist = distance_function(query_vector_data, log_entry->vector.getVecData(), vector_dim);
            if (dist < entry_max_dist)
            {
                found_entries.push_back({ .distance = dist, .entry = log_entry });
            }

            prev_log_entry = log_entry;
            log_entry = log_entry->next;

            prev_log_entry->vector.unlock();

            scanned++;
        }

        stats.sp_checked_count += scanned;

        // In case the sweep ended reaching tail,
        // return the tail so that the cache entry does not scan from the log_head 
        // in the future.
        if (log_entry == nullptr)
            log_entry = log_tail;

        // Increase the last reference point
        if (log_entry != nullptr)
            log_entry->vector.increaseRefCount();

        _unlock();

        return log_entry;
    }

    void
    RfWriteLog::clearLog() noexcept
    {
        _lock();
        
        log_map.clear();
        centry_map.clear();

        log_head = nullptr;
        log_tail = nullptr;

        std::memset(&risk_status, 0, sizeof(struct Score));

        _unlock();
    }

    write_log_entry_t*
    RfWriteLog::getHead() noexcept
    {
        return log_head;
    }

    write_log_entry_t*
    RfWriteLog::getTail() noexcept
    {
        return log_tail;
    }

    bool
    RfWriteLog::needRunSlowPath() noexcept
    {
        return (allowed_risk < risk_status.curr_risk);
    }

    std::uint64_t
    RfWriteLog::getDistance(write_log_entry_t* curr_entry) noexcept
    {
        write_log_entry_t* tail = getTail();

        if (tail == nullptr)
            return 0;

        std::uint64_t curr_epoch = 0;
        std::uint64_t latest_epoch = position;

        if (curr_entry == nullptr)
        {
            std::uint64_t head_epoch = log_head->epoch;
            return latest_epoch - head_epoch;
        }
        else
        {
            std::uint64_t curr_epoch = curr_entry->epoch;
            return latest_epoch - curr_epoch;
        }
    }

    write_log_stats_t
    RfWriteLog::getStats() noexcept
    {
        return stats;
    }

    write_log_stats_t*
    RfWriteLog::getStatsPtr() noexcept
    {
        return &stats;
    }

    size_t
    RfWriteLog::getMappedSize() noexcept
    {
        return centry_map.size();
    }

    std::string
    RfWriteLog::toString() noexcept
    {
        std::string buffer = "WriteLog status: total elements (";
        buffer += std::to_string(log_current_size) + ")\n";

        // Only print head and tail vector ID
        if (log_head != nullptr)
        {
            buffer += "        Head: " + std::to_string(log_head->vector.getVecId()) + ", ";
            buffer += "Head Ref: " + std::to_string(log_head->vector.getRefCount()) + "\n";
            buffer += "        Tail: " + std::to_string(log_tail->vector.getVecId()) + ", ";
            buffer += "Tail Ref: " + std::to_string(log_tail->vector.getRefCount()) + "\n";
            buffer += "        Refs from tail (tail-10): ";

            write_log_entry_t* log_entry = log_tail;
            for (int i = 0; i < 10; i++)
            {
                if (log_entry == nullptr)
                    break;

                buffer += std::to_string(log_entry->vector.getRefCount()) + ", ";
                log_entry = log_entry->prev;
            }

            buffer += "...\n";

            std::uint32_t total_ref_count = 0;
            log_entry = log_head;
            while (log_entry != nullptr)
            {
                total_ref_count += log_entry->vector.getRefCount();
                log_entry = log_entry->next;
            }
            buffer += "            Total Ref Count: " + std::to_string(total_ref_count) + "\n";
        }

        // risk scores?
        buffer += "        Risk Status: \n";
        buffer += "            Current Risk: " + std::to_string(risk_status.curr_risk) + "\n";
        buffer += "            Total DDF: " + std::to_string(risk_status.total_rf) + "\n";
        buffer += "            Average DDF: " + std::to_string(risk_status.avg_rf) + "\n";
        buffer += "            Total Unseen: " + std::to_string(risk_status.total_unseen) + "\n";
        buffer += "            Total CEntry: " + std::to_string(risk_status.total_centry) + "\n";
        buffer += "            LEntry insert counts: " + std::to_string(stats.insert_count) + "\n";
        buffer += "            LEntry trim counts: " + std::to_string(stats.delete_count) + "\n";
        buffer += "            Scanned counts (SP): " + std::to_string(stats.sp_checked_count) + "\n";
        buffer += "            Refresh counts: " + std::to_string(stats.refresh_count) + "\n";

        return buffer;
    }

    // Sample result conversion function
    void
    sampleResultConversion(
        char* vector_data, size_t vector_data_size
    )
    {
        // Given the vector data (raw buffer),
        // we can convert it to a result format.

    }
}