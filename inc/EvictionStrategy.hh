#include <memory>
#include <boost/lockfree/queue.hpp>
#include <boost/intrusive/list.hpp>
#include <unordered_map>

#include "VectorPool2.hh"

#ifndef TOPKACHE_EVICTSTRATEGY_FIFO_H
#define TOPKACHE_EVICTSTRATEGY_FIFO_H

namespace topkache
{
    class EvictionStrategy
    {
    public:
        EvictionStrategy() noexcept = default;
        ~EvictionStrategy() noexcept = default;

        size_t  size;

        virtual bool nextEvictCandidate(vector_id_t* candidate_key) noexcept = 0;
        virtual bool addEvictCandidate(vector_id_t candidate_key) noexcept = 0;
        virtual bool recentlyAccessed(vector_id_t candidate_key) noexcept = 0;

        virtual inline size_t getCurrSize() 
        {
            return size;
        }
    };

    class EvictionStrategyFifo : public EvictionStrategy
    {
    private:
        // FIFO queue
        std::unique_ptr<boost::lockfree::queue<std::uint64_t>>  fifo_queue;

    public:
        EvictionStrategyFifo() noexcept
        {
            fifo_queue = std::make_unique<boost::lockfree::queue<std::uint64_t>>(1024);
            size = 0;
        }

        ~EvictionStrategyFifo() noexcept
        {
            
        }

        virtual bool
        nextEvictCandidate(vector_id_t* candidate_key) noexcept 
        {
            vector_id_t key = 0;
            bool success = fifo_queue->pop(key);

            if (success)
            {
                *candidate_key = key;
            }

            size--;
            return success;
        }
        
        virtual bool 
        addEvictCandidate(vector_id_t candidate_key) noexcept 
        {
            size++;
            return fifo_queue->push(candidate_key);
        }

        virtual bool
        recentlyAccessed(vector_id_t candidate_key) noexcept 
        {
            // FIFO strategy does not track recently accessed items
            return false;
        }
    };

    // 
    // LRU eviction strategy
    class EvictionStrategyLru : public EvictionStrategy
    {
    private:
        struct Node : public boost::intrusive::list_base_hook<>
        {
            vector_id_t key;
            Node(vector_id_t k) : key(k) {}
        };

        boost::intrusive::list<Node>                            lru_list;
        std::unordered_map<vector_id_t, std::unique_ptr<Node>>  lru_map;

    public:
        EvictionStrategyLru() noexcept
        {
            lru_list.clear();
            lru_map.clear();
            size = 0;
        }

        ~EvictionStrategyLru() noexcept
        {
            lru_list.clear();
            lru_map.clear();
        }

        virtual bool
        nextEvictCandidate(vector_id_t* candidate_key) noexcept 
        {
            if (lru_list.empty())
                return false;

            Node& node = lru_list.back();
            *candidate_key = node.key;
            lru_list.pop_back();
            
            size--;
            return true;
        }
        
        virtual bool 
        addEvictCandidate(vector_id_t candidate_key) noexcept 
        {
            auto it = lru_map.find(candidate_key);
            if (it != lru_map.end())
                return false;

            size++;
            auto node = std::make_unique<Node>(candidate_key);
            lru_list.push_front(*node);
            lru_map[candidate_key] = std::move(node);

            return true;
        }

        virtual bool
        recentlyAccessed(vector_id_t candidate_key) noexcept 
        {
            auto it = lru_map.find(candidate_key);
            if (it == lru_map.end())
                return false;

            Node& node = *(it->second);
            lru_list.erase(lru_list.iterator_to(node));
            lru_list.push_front(node);

            return true;
        }
    };
    
}

#endif