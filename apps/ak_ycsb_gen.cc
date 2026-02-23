#include <boost/program_options.hpp>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>

#include "utils/ak_sequence_gen.hh"

namespace po = boost::program_options;

namespace {

constexpr double kInsertRatio = 0.0;
constexpr double kUpdateRatio = 0.0;
constexpr double kReadRatio = 1.0;

constexpr const char* kDefaultDistType = "zipfian";
constexpr const char* kDefaultOutputDir = "apps/ak_ycsb_gen";

} // namespace

int
main(int argc, char* argv[])
{
    po::options_description desc("Allowed options");
    desc.add_options()
        ("total,t", po::value<uint32_t>()->default_value(1000000), "Total Record Counts")
        ("insert,i", po::value<double>()->default_value(0.0), "Insert Ratio")
        ("update,u", po::value<double>()->default_value(0.0), "Update Ratio")
        ("read,r", po::value<double>()->default_value(1.0), "Read Ratio")
        ("query,q", po::value<uint32_t>()->default_value(1000000), "Query Count");

    // Keep the original behavior: require explicit CLI invocation.
    // The generated output format must remain stable.
    if (argc < 6) {
        std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
        std::cout << desc << std::endl;
        return 1;
    }

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    const std::uint32_t total_record_count = vm["total"].as<uint32_t>();
    const std::uint32_t query_count = vm["query"].as<uint32_t>();

    double insert_ratio = vm["insert"].as<double>();
    double update_ratio = vm["update"].as<double>();
    double read_ratio = vm["read"].as<double>();

    // Force read-only workload (all READ operations).
    // Keep CLI options for compatibility, but ignore insert/update ratios.
    insert_ratio = kInsertRatio;
    update_ratio = kUpdateRatio;
    read_ratio = kReadRatio;

    YcsbSeqGenerator seq_gen;
    seq_gen.setGenerator(total_record_count, kDefaultDistType, insert_ratio, update_ratio, read_ratio);
    seq_gen.generateKeySequence(query_count);

    std::error_code ec;
    std::filesystem::create_directories(kDefaultOutputDir, ec);
    if (ec) {
        std::cerr << "[ak_ycsb_gen] Failed to create output directory: " << kDefaultOutputDir
                  << " (" << ec.message() << ")" << std::endl;
        return 2;
    }

    const std::string frequency_path = std::string(kDefaultOutputDir) + "/sequence-freqs.csv";
    const std::string sequence_path = std::string(kDefaultOutputDir) + "/sequence.csv";

    seq_gen.exportFrequency(frequency_path);
    seq_gen.exportSequence(sequence_path);

    // 0: INSERT,
    // 1: UPDATE,
    // 2: READ,
    return 0;
}
