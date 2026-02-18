#pragma once

#include <string>

#include "core/ak_ann_cache2_context.hh"

namespace aker
{
    /**
     * @brief Telemetry module for ANNCache2.
     *
     * This module builds a concise cache summary and coordinates trace export.
     */
    class ANNCache2Telemetry
    {
    public:
        /**
         * @brief Constructs the module with the shared cache context.
         */
        explicit ANNCache2Telemetry(ANNCache2Context* context) noexcept;

        /**
         * @brief Builds a human-readable status string.
         */
        std::string buildStatusText() noexcept;

        /**
         * @brief Builds a concise summary in CSV (key,value) form.
         */
        std::string buildSummaryCsv() noexcept;

        /**
         * @brief Exports all telemetry files under /tmp/aker_trace_<timestamp>/.
         */
        void exportTraceToFiles() noexcept;

    private:
        ANNCache2Context* context_;
    };
}
