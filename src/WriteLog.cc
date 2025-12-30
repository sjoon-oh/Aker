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
    WriteLog::WriteLog(size_t delta_bound) noexcept
        : scan_threshold(32), log_lock(false), 
            log_epoch(0), log_delta_bound(delta_bound), 
                log_current_size(0), 
                heartbeat(0), force_age(100000) // 1K Calls, we age.
    {
        log_head = nullptr;

        sealed = new std::list<rev_centry_t*>();
        unsealed = new std::list<rev_centry_t*>();
    }

    WriteLog::~WriteLog() noexcept
    {
        write_log_entry_t* log_entry = log_head;
        while (log_entry != nullptr)
        {
            write_log_entry_t* next_log_entry = log_entry->next;
            delete log_entry;

            log_entry = next_log_entry;
        }

        delete sealed;
        delete unsealed;
    }

    void
    WriteLog::_lock() noexcept
    {
        while (_tryLock() == false)
            ;
    }

    bool
    WriteLog::_tryLock() noexcept
    {
        bool expected = false;
        bool desired = true;

        return log_lock.compare_exchange_strong(expected, desired, std::memory_order_acquire);
    }

    void
    WriteLog::_unlock() noexcept
    {
        log_lock.store(false, std::memory_order_release);
    }

    void
    WriteLog::_insertLEntry(Vector2& vector, size_t vector_data_size) noexcept
    {
        write_log_entry_t* log_entry = nullptr;
        vector_id_t vector_id = vector.getVecId();

        bool insert_success = log_map.try_emplace_or_visit(
            vector_id, nullptr,
            [&](const auto& pair)
            {
                // If the entry already exists, the entry is not allocated.
                // We convey the existing entry to the allocated_entry.
                log_entry = pair.second;
            }
        );

        // Make a log entry
        if (log_entry == nullptr)
        {
            log_entry = new write_log_entry_t(vector_data_size);

            log_entry->vector.lock();

            log_entry->vector.setVecId(vector_id);
            log_entry->vector.setVecVersion(vector.getVecVersion());

            if (log_current_size > log_delta_bound)
            {
                // 
                // At this moment, we wait until the sealed 
                // size becomes zero, and we swap it.
                if (sealed->size() == 0)
                {
                    auto* temporary = sealed;
                    sealed = unsealed;
                    unsealed = temporary;
                }
            }

            log_entry->epoch = log_epoch;

            // Make the initial reference count to zero.
            log_entry->vector.decreaseRefCount(); 
            
            std::memcpy(log_entry->vector.getVecData(), vector.getVecData(), vector_data_size);

            log_map.visit(
                vector_id,
                [&](auto& pair)
                {
                    pair.second = log_entry;                    
                }
            );

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

    write_log_entry_t*
    WriteLog::sweepLEntry(
        vector_data_t* query_vector_data,
        size_t vector_dim,
        float entry_max_dist,
        std::vector<close_candidates_t>& found_entries,
        distance_function_t distance_function,
        write_log_entry_t* scan_start
        ) noexcept
    {
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

            // printf("    Found distance %f, max distance %f\n", dist, entry_max_dist);

            prev_log_entry = log_entry;
            log_entry = log_entry->next;

            prev_log_entry->vector.unlock();
        }

        if (log_entry == nullptr)
            log_entry = log_tail;

        if (log_entry != nullptr)
            log_entry->vector.increaseRefCount();

        _unlock();

        return log_entry;
    }

    void
    WriteLog::insertLEntry(Vector2& vector, size_t vector_data_size) noexcept
    {
        _lock();
        _insertLEntry(vector, vector_data_size);
        stats.insert_count++;
        _unlock();
    }

    void
    WriteLog::addToUnsealed(rev_centry_t* entry) noexcept
    {
        _lock();

        unsealed->push_back(entry);

        struct EntryLocation location = {
            .list = unsealed,
            .it = --(unsealed->end())
        };

        centry_map.try_emplace_or_visit(entry, 
            location, [](auto& pair) { });

        _unlock();
    }

    void
    WriteLog::arrangeSealed(rev_centry_t* saved_entry) noexcept
    {
        _lock();

        rev_centry_t* front = sealed->front();
        assert(front == saved_entry);

        result_cache_entry_t* cache_entry = as_centry(saved_entry);
        int visited = centry_map.visit(
            saved_entry,
            [&](auto& pair) { 

                if (cache_entry->checkpoint->epoch != log_epoch)
                {
                    // We move to the unsealed list.
                    sealed->pop_front();
                    unsealed->push_back(saved_entry);
                    
                    pair.second.list = unsealed;
                    pair.second.it = --(unsealed->end());
                }
                else
                {
                    sealed->splice(sealed->end(), *sealed, sealed->begin());
                }
            }
        );

        assert(visited != 0);

        _unlock();
    }

    bool
    WriteLog::deregFromList(rev_centry_t* cache_entry) noexcept
    {
        _lock();

        int visited = centry_map.visit(
            cache_entry, 
            [&](const auto& pair) { 
                pair.second.list->erase(pair.second.it);
            });
        assert(visited != 0);

        // Remove it from the map
        centry_map.erase(cache_entry);

        _unlock();

        return true;
    }

    /* Update the references of the log entries
     *  If the reference count of the log entry is zero, the entry is deleted.
     *  This 
     */
    bool
    WriteLog::safeTrims(write_log_entry_t* scan_start) noexcept
    {
        _lock();
        write_log_entry_t* log_entry = scan_start;

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
    WriteLog::clearLog() noexcept
    {
        _lock();
        
        log_map.clear();
        // centry_map.clear();

        sealed->clear();
        unsealed->clear();

        log_head = nullptr;
        log_tail = nullptr;

        _unlock();
    }

    std::string
    WriteLog::toString() noexcept
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

            buffer += "        Refs from tail: ";

            // Show sample references of top 10 from tail.
            write_log_entry_t* log_entry = log_tail;
            for (int i = 0; i < 10; i++)
            {
                if (log_entry == nullptr)
                    break;

                buffer += std::to_string(log_entry->vector.getRefCount()) + ", ";
                log_entry = log_entry->prev;
            }

            buffer += "...\n";

            // Summation of the reference counts
            std::uint32_t total_ref_count = 0;
            log_entry = log_head;
            while (log_entry != nullptr)
            {
                total_ref_count += log_entry->vector.getRefCount();
                log_entry = log_entry->next;
            }
            buffer += "            Total Ref Count: " + std::to_string(total_ref_count) + "\n";
        }

        // Check the sealed and unsealed list
        buffer += "        Sealed List: " + std::to_string(sealed->size()) + "\n";
        buffer += "        Unsealed List: " + std::to_string(unsealed->size()) + "\n";

        return buffer;
    }

    write_log_entry_t*
    WriteLog::getHead() noexcept
    {
        return log_head;
    }

    write_log_entry_t*
    WriteLog::getTail() noexcept
    {
        return log_tail;
    }

    write_log_stats_t
    WriteLog::getStats() noexcept
    {
        return stats;
    }

    rev_centry_t*
    WriteLog::getSealedNext() noexcept
    {
        if (sealed->size() == 0)
            return nullptr;

        rev_centry_t* front = sealed->front();
        return front;
    }

    const size_t
    WriteLog::getScanThreshold() noexcept
    {
        return scan_threshold;
    }

    size_t
    WriteLog::getMappedSize() noexcept
    {
        return centry_map.size();
    }

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
    // -- DdfWriteLog
    DdfWriteLog::DdfWriteLog(size_t in_topk, size_t scan_thresh, double risk_thresh) noexcept
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

    DdfWriteLog::~DdfWriteLog() noexcept
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
    DdfWriteLog::_lock() noexcept
    {
        while (_tryLock() == false)
            ;
    }

    bool
    DdfWriteLog::_tryLock() noexcept
    {
        bool expected = false;
        bool desired = true;

        return log_lock.compare_exchange_strong(expected, desired, std::memory_order_acquire);
    }

    void
    DdfWriteLog::_unlock() noexcept
    {
        log_lock.store(false, std::memory_order_release);
    }

    void
    DdfWriteLog::_insertLEntry(Vector2& vector, size_t vector_data_size) noexcept
    {
        write_log_entry_t* log_entry = nullptr;
        vector_id_t vector_id = vector.getVecId();

        bool insert_success = log_map.try_emplace_or_visit(
            vector_id, nullptr,
            [&](const auto& pair) { log_entry = pair.second; }
        );

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
    estimatedSubs(double avg_ddf, size_t total_unseen, size_t total_entry) noexcept
    {
        return avg_ddf * (total_unseen / total_entry) ;
    }

    void
    DdfWriteLog::_increaseCEntryDdf(double ddf, size_t unseen, size_t curr_n_centry) noexcept
    {
        risk_status.total_ddf += ddf;
        risk_status.total_unseen += unseen;

        assert((ddf >= 0) && (ddf <= 1));

        risk_status.total_centry = curr_n_centry;
        risk_status.avg_ddf = risk_status.total_ddf / risk_status.total_centry;

        // Estimated number of IDs in the worst-case
        double est_total_subs = estimatedSubs(risk_status.avg_ddf, risk_status.total_unseen, risk_status.total_centry);

        // Normalize how much error there will be within the top-k
        risk_status.curr_risk = est_total_subs / (k * risk_status.total_centry);
    }

    void
    DdfWriteLog::_decreaaseCEntryDdf(double ddf, size_t unseen, size_t curr_n_centry) noexcept
    {
        risk_status.total_ddf -= ddf;
        risk_status.total_unseen -= unseen;

        assert((ddf >= 0) && (ddf <= 1));

        risk_status.total_centry = curr_n_centry;
        risk_status.avg_ddf = risk_status.total_ddf / risk_status.total_centry;

        // Estimated number of IDs in the worst-case
        double est_total_subs = estimatedSubs(risk_status.avg_ddf, risk_status.total_unseen, risk_status.total_centry);

        // Normalize how much error there will be within the top-k
        risk_status.curr_risk = est_total_subs / (k * risk_status.total_centry);
    }

    void
    DdfWriteLog::insertLEntry(Vector2& vector, size_t vector_data_size) noexcept
    {
        _lock();
        _insertLEntry(vector, vector_data_size);
        stats.insert_count++;
        _unlock();
    }

    void
    DdfWriteLog::addToRr(rev_centry_t* cache_entry) noexcept
    {
        _lock();

        rr_list->push_back(cache_entry);

        struct EntryLocation location = { .it = --(rr_list->end()) };
        centry_map.try_emplace_or_visit(cache_entry, location, [](auto& pair) { });

        _unlock();
    }

    rev_centry_t*
    DdfWriteLog::getNextRr() noexcept
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
    DdfWriteLog::removeFromRr(rev_centry_t* cache_entry) noexcept
    {
        _lock();

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
    DdfWriteLog::safeTrims(write_log_entry_t* scan_start) noexcept
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
    DdfWriteLog::increaseCEntryDdf(double ddf, size_t unseen, size_t curr_n_centry) noexcept
    {
        _lock();
        _increaseCEntryDdf(ddf, unseen, curr_n_centry);
        _unlock();
    }

    void
    DdfWriteLog::decreaseCEntryDdf(double ddf, size_t unseen, size_t curr_n_centry) noexcept
    {
        _lock();
        _decreaaseCEntryDdf(ddf, unseen, curr_n_centry);
        _unlock();
    }

    write_log_entry_t*
    DdfWriteLog::sweepLEntryBatch(
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
    DdfWriteLog::clearLog() noexcept
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
    DdfWriteLog::getHead() noexcept
    {
        return log_head;
    }

    write_log_entry_t*
    DdfWriteLog::getTail() noexcept
    {
        return log_tail;
    }

    bool
    DdfWriteLog::needRunSlowPath() noexcept
    {
        return (allowed_risk < risk_status.curr_risk);
    }

    std::uint64_t
    DdfWriteLog::getDistance(write_log_entry_t* curr_entry) noexcept
    {
        write_log_entry_t* tail = getTail();

        std::uint64_t curr_epoch = curr_entry->epoch;
        std::uint64_t latest_epoch = tail->epoch;

        // epoch only increases, thus latest epoch may be smaller than the currently viewing epoch.
        std::uint64_t distance = latest_epoch - curr_epoch;

        return distance;
    }

    write_log_stats_t
    DdfWriteLog::getStats() noexcept
    {
        return stats;
    }

    size_t
    DdfWriteLog::getMappedSize() noexcept
    {
        return centry_map.size();
    }

    std::string
    DdfWriteLog::toString() noexcept
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
        buffer += "            Total DDF: " + std::to_string(risk_status.total_ddf) + "\n";
        buffer += "            Average DDF: " + std::to_string(risk_status.avg_ddf) + "\n";
        buffer += "            Total Unseen: " + std::to_string(risk_status.total_unseen) + "\n";
        buffer += "            Total CEntry: " + std::to_string(risk_status.total_centry) + "\n";
        buffer += "            LEntry insert counts: " + std::to_string(stats.insert_count) + "\n";
        buffer += "            LEntry trim counts: " + std::to_string(stats.delete_count) + "\n";
        buffer += "            Processed counts (SP): " + std::to_string(stats.sp_checked_count) + "\n";

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