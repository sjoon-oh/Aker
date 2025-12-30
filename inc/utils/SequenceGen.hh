/*
 * YcsbSeqGenerator.hh
 * Author: Sukjoon Oh (sjoon@kaist.ac.kr)
 */

#ifndef _YCSB_SEQ_GENERATOR_H
#define _YCSB_SEQ_GENERATOR_H

#include <cstdint>
#include <vector>
#include <string>

#include <memory>

#include "YCSB-C/core/generator.h"
#include "YCSB-C/core/uniform_generator.h"
#include "YCSB-C/core/zipfian_generator.h"
#include "YCSB-C/core/discrete_generator.h"
#include "YCSB-C/core/scrambled_zipfian_generator.h"
#include "YCSB-C/core/skewed_latest_generator.h"
#include "YCSB-C/core/const_generator.h"
#include "YCSB-C/core/core_workload.h"


class YcsbSeqGenerator
{
private:
    std::vector<std::uint64_t>                              keySequence;
    std::vector<ycsbc::Operation>                           opSequence;

    std::unique_ptr<ycsbc::CounterGenerator>                insertKeySequence;

    std::unique_ptr<ycsbc::CounterGenerator>                keyGenerator;
    std::unique_ptr<ycsbc::Generator<std::uint64_t>>        keyChooser;

    std::unique_ptr<ycsbc::DiscreteGenerator<ycsbc::Operation>>    opChooser;

    // 
    // Generated sequence
    std::uint64_t generateNextKey() noexcept;
    std::uint64_t chooseNextKey() noexcept;

    ycsbc::Operation chooseNextOp() noexcept;

public:

    YcsbSeqGenerator() noexcept;

    virtual ~YcsbSeqGenerator() = default;

    virtual bool setGenerator(
        size_t recordCount, std::string distType,
        double insertRatio = 0.0, double updateRatio = 0.0, double readRatio = 0.0) noexcept;
    virtual void resetGenerator() noexcept;

    virtual std::vector<std::uint64_t>& generateKeySequence(size_t numQueryVec) noexcept;

    virtual size_t checkUniqueIds(std::vector<std::pair<std::uint64_t, size_t>>& idsByFreq) noexcept;

    virtual void exportFrequency() noexcept;
    virtual void exportSequence() noexcept;
};

#endif