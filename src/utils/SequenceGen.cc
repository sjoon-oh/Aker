/*
 * YcsbSeqGenerator.
 * Author: Sukjoon Oh (sjoon@kaist.ac.kr)
 */

#include <cstdint>
#include <unordered_map>

#include <fstream>

#include "utils/SequenceGen.hh"

YcsbSeqGenerator::YcsbSeqGenerator() noexcept
{
    resetGenerator();
}

std::uint64_t 
YcsbSeqGenerator::generateNextKey() noexcept
{
    std::uint64_t keyNum = keyGenerator->Next();
    return keyNum;
}

std::uint64_t 
YcsbSeqGenerator::chooseNextKey() noexcept
{
    std::uint64_t keyNum = 0;
    if (keyChooser.get() != nullptr)
    {   
        do
        {
            keyNum = keyChooser->Next();
        } 
        while (keyNum > insertKeySequence->Last());
    }
    return keyNum;
}

ycsbc::Operation
YcsbSeqGenerator::chooseNextOp() noexcept
{
    ycsbc::Operation op = ycsbc::Operation::INSERT;
    if (opChooser.get() != nullptr)
    {
        op = opChooser->Next();
    }

    return op;
}

bool
YcsbSeqGenerator::setGenerator(
    size_t recordCount, 
    std::string distType,
    double insertRatio,
    double updateRatio,
    double readRatio
) noexcept
{
    insertKeySequence->Set(recordCount);

    // Make the distType lowercase.
    for (char &c : distType) c = c | ' ';

    if (distType == "uniform")
        keyChooser.reset(new ycsbc::UniformGenerator(0, recordCount - 1));

    else if (distType == "zipfian")
        keyChooser.reset(new ycsbc::ScrambledZipfianGenerator(recordCount));

    else if (distType == "latest")
        keyChooser.reset(new ycsbc::SkewedLatestGenerator(*insertKeySequence));

    else
    {
        keyChooser.reset();
        return false;
    }

    if (insertRatio > 0)
        opChooser->AddValue(ycsbc::Operation::INSERT, insertRatio);

    if (updateRatio > 0)
        opChooser->AddValue(ycsbc::Operation::UPDATE, updateRatio);

    if (readRatio > 0)
        opChooser->AddValue(ycsbc::Operation::READ, readRatio);

    return true;
}

/**
 * Resets the generator to its initial state.
 */
void 
YcsbSeqGenerator::resetGenerator() noexcept
{
    keySequence.clear();
    opSequence.clear();

    // Make default
    insertKeySequence.reset(new ycsbc::CounterGenerator(3));
    keyGenerator.reset(new ycsbc::CounterGenerator(0));

    opChooser.reset(
        new ycsbc::DiscreteGenerator<ycsbc::Operation>()
    );

    keyChooser.reset();
}

std::vector<std::uint64_t>&
YcsbSeqGenerator::generateKeySequence(size_t numVec) noexcept
{
    for (size_t count = 0; count < numVec; count++)
        keySequence.emplace_back(chooseNextKey());

    // This class just generates the sequence with the given distribution set.
    // Mapping of external vectors should be done by the caller.
    // Use retuned sequence ID to map the vectors.

    return keySequence;
}

size_t
YcsbSeqGenerator::checkUniqueIds(
    std::vector<std::pair<std::uint64_t, size_t>>& idsByFreq
) noexcept
{
    // Extract unique keys
    std::unordered_map<std::uint64_t, size_t> uniqueKeys;
    for (std::uint64_t& key: keySequence)
    {
        if (uniqueKeys.find(key) == uniqueKeys.end())
            uniqueKeys[key] = 1;
            
        else 
            uniqueKeys[key]++;
    }

    // Insert to the vector
    idsByFreq = std::vector<std::pair<std::uint64_t, size_t>>(
        uniqueKeys.begin(), uniqueKeys.end()
    );

    // Sort the vector by frequency (descending order)
    std::sort(
        idsByFreq.begin(),
        idsByFreq.end(),
        [](const std::pair<std::uint64_t, size_t>& a, const std::pair<std::uint64_t, size_t>& b) {
            return a.second > b.second;
        }
    );

    return uniqueKeys.size();
}


void
YcsbSeqGenerator::exportFrequency() noexcept
{

    std::vector<std::pair<std::uint64_t, size_t>> uniqueKeys;
    checkUniqueIds(uniqueKeys);

    // 
    // Export to files to visualize in descending count order.
    std::fstream outputFile("sequence-freqs.csv", std::ios::out);
    if (!outputFile)
        return;

    for (const auto& pair: uniqueKeys)
        outputFile << pair.first << "\t" << pair.second << std::endl;

    outputFile.close();
}

void
YcsbSeqGenerator::exportSequence() noexcept
{
    std::fstream outputFile("sequence.csv", std::ios::out);
    if (!outputFile)
        return;

    for (const auto& key: keySequence)
        outputFile << key << "\t" << chooseNextOp() << std::endl;

    outputFile.close();
}