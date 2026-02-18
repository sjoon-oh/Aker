#pragma once

#include <cstddef>
#include <cstdint>

#include <memory>
#include <unordered_map>

#include <boost/intrusive/list.hpp>
#include <boost/lockfree/queue.hpp>

#include "ak_vector_slot.hh"

namespace aker
{
    /**
     * @brief Eviction strategy interface.
     */
    class EvictionStrategy
    {
    public:
        /**
         * @brief Constructor.
         */
        EvictionStrategy() noexcept = default;

        /**
         * @brief Virtual destructor for safe deletion via base pointer.
         */
        virtual ~EvictionStrategy() noexcept = default;

        EvictionStrategy(const EvictionStrategy&) = delete;
        EvictionStrategy& operator=(const EvictionStrategy&) = delete;

        /**
         * @brief Returns the next eviction candidate via output parameter.
         */
        virtual bool nextEvictCandidate(vector_id_t* candidate_key) noexcept = 0;

        /**
         * @brief Adds an eviction candidate.
         */
        virtual bool addEvictCandidate(vector_id_t candidate_key) noexcept = 0;

        /**
         * @brief Marks a candidate as recently accessed.
         */
        virtual bool recentlyAccessed(vector_id_t candidate_key) noexcept = 0;

        /**
         * @brief Convenience wrapper returning the next candidate key, or 0 on failure.
         */
        inline vector_id_t nextEvictCandidate() noexcept
        {
            vector_id_t key = 0;
            bool success = nextEvictCandidate(&key);
            return success ? key : 0;
        }

        /**
         * @brief Returns the current size of the eviction structure.
         */
        inline size_t getCurrSize() const noexcept
        {
            return size_;
        }

    protected:
        /**
         * @brief Current number of tracked candidates.
         */
        size_t size_ = 0;
    };

    /**
     * @brief FIFO eviction strategy.
     */
    class EvictionStrategyFifo final : public EvictionStrategy
    {
    public:
        /**
         * @brief Constructs a FIFO strategy.
         */
        EvictionStrategyFifo() noexcept
        {
            fifo_queue_ = std::make_unique<boost::lockfree::queue<std::uint64_t>>(k_default_fifo_capacity);
            size_ = 0;
        }

        /**
         * @brief Destructor.
         */
        ~EvictionStrategyFifo() noexcept override = default;

        /**
         * @brief Pops the next candidate from the FIFO queue.
         */
        bool nextEvictCandidate(vector_id_t* candidate_key) noexcept override
        {
            vector_id_t key = 0;
            bool success = fifo_queue_->pop(key);

            if (success && candidate_key != nullptr)
                *candidate_key = key;

            size_--;
            return success;
        }

        /**
         * @brief Pushes a candidate into the FIFO queue.
         */
        bool addEvictCandidate(vector_id_t candidate_key) noexcept override
        {
            size_++;
            return fifo_queue_->push(candidate_key);
        }

        /**
         * @brief FIFO strategy does not track recently accessed items.
         */
        bool recentlyAccessed(vector_id_t candidate_key) noexcept override
        {
            (void)candidate_key;
            return false;
        }

    private:
        static constexpr std::size_t k_default_fifo_capacity = 1024;

        std::unique_ptr<boost::lockfree::queue<std::uint64_t>> fifo_queue_;
    };

    /**
     * @brief LRU eviction strategy.
     */
    class EvictionStrategyLru final : public EvictionStrategy
    {
    public:
        /**
         * @brief Constructs an LRU strategy.
         */
        EvictionStrategyLru() noexcept
        {
            lru_list_.clear();
            lru_map_.clear();
            size_ = 0;
        }

        /**
         * @brief Destructor.
         */
        ~EvictionStrategyLru() noexcept override
        {
            lru_list_.clear();
            lru_map_.clear();
        }

        /**
         * @brief Pops the least-recently used candidate.
         */
        bool nextEvictCandidate(vector_id_t* candidate_key) noexcept override
        {
            if (lru_list_.empty())
                return false;

            Node& node = lru_list_.back();

            if (candidate_key != nullptr)
                *candidate_key = node.key;

            lru_list_.pop_back();

            size_--;
            return true;
        }

        /**
         * @brief Adds a new candidate to the LRU list.
         */
        bool addEvictCandidate(vector_id_t candidate_key) noexcept override
        {
            auto it = lru_map_.find(candidate_key);
            if (it != lru_map_.end())
                return false;

            size_++;

            std::unique_ptr<Node> node = std::make_unique<Node>(candidate_key);
            lru_list_.push_front(*node);
            lru_map_[candidate_key] = std::move(node);

            return true;
        }

        /**
         * @brief Updates a candidate as recently accessed.
         */
        bool recentlyAccessed(vector_id_t candidate_key) noexcept override
        {
            auto it = lru_map_.find(candidate_key);
            if (it == lru_map_.end())
                return false;

            Node& node = *(it->second);
            lru_list_.erase(lru_list_.iterator_to(node));
            lru_list_.push_front(node);

            return true;
        }

    private:
        /**
         * @brief Node stored in the intrusive LRU list.
         */
        struct Node : public boost::intrusive::list_base_hook<>
        {
            explicit Node(vector_id_t k)
                : key(k)
            {
            }

            vector_id_t key;
        };

        boost::intrusive::list<Node> lru_list_;
        std::unordered_map<vector_id_t, std::unique_ptr<Node>> lru_map_;
    };
}
