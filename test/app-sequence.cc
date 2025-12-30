#include <boost/program_options.hpp>
#include <iostream>

#include "utils/SequenceGen.hh"

namespace po = boost::program_options;

int
main(int argc, char* argv[])
{
    po::options_description desc("Allowed options");
    desc.add_options()
        ("total,t", po::value<uint32_t>()->default_value(1000000), "Total Record Counts")
        ("insert,i", po::value<double>()->default_value(0.0), "Insert Ratio")
        ("update,u", po::value<double>()->default_value(0.0), "Update Ratio")
        ("read,r", po::value<double>()->default_value(1.0), "Read Ratio")
        ("query,q", po::value<uint32_t>()->default_value(1000000), "Query Count")
        ;

    if (argc < 6)
    {
        std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
        std::cout << desc << std::endl;
        return 1;
    }

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    std::uint32_t total_record_count = vm["total"].as<uint32_t>();
    std::uint32_t query_count = vm["query"].as<uint32_t>();

    double insert_ratio = vm["insert"].as<double>();
    double update_ratio = vm["update"].as<double>();
    double read_ratio = vm["read"].as<double>();

    YcsbSeqGenerator seq_gen;
    seq_gen.setGenerator(total_record_count, "zipfian", insert_ratio, update_ratio, read_ratio);
    
    auto keys = seq_gen.generateKeySequence(query_count);
    
    seq_gen.exportFrequency();
    seq_gen.exportSequence();

    // 0: INSERT,
    // 1: UPDATE,
    // 2: READ,
    
    return 0;
}