
#include <vector>
#include <string>
#include <limits>
#include <algorithm>

#include <cstdio>

#include "core/ResultCache2.hh"

namespace topkache
{
    void
    ResultCacheStats::clear() noexcept
    {
        cache_hit                               = 0;
        cache_miss                              = 0;
        cache_evict                             = 0;
        cache_sim_hit                           = 0;
        cache_dropout                           = 0;
        cache_refreshed                         = 0;
        cache_invalid                           = 0;  

        for (auto& pair : stat_level_0)
            pair.second.clear();

        for (auto& pair : stat_level_1)
            pair.second.clear();
    }

    ResultCacheStats::ResultCacheStats() noexcept
    {

        // 
        // Add the latency measurement here
        stat_level_0["simGetCEntry"]        = std::vector<ElapsedPair>(); 
        stat_level_0["insertCEntry"]        = std::vector<ElapsedPair>();
        stat_level_0["linkCEntry"]          = std::vector<ElapsedPair>();
        stat_level_0["insertWLEntry2"]      = std::vector<ElapsedPair>();
        stat_level_0["markVecDeleted"]      = std::vector<ElapsedPair>();
        stat_level_0["handleAgedWLEntry"]   = std::vector<ElapsedPair>();

        stat_level_1["simGetCEntry-1"]      = std::vector<ElapsedPair>();
        stat_level_1["simGetCEntry-2"]      = std::vector<ElapsedPair>();
        stat_level_1["simGetCEntry-3"]      = std::vector<ElapsedPair>();
        stat_level_1["insertCEntry-1"]      = std::vector<ElapsedPair>();
        stat_level_1["insertCEntry-2"]      = std::vector<ElapsedPair>();
        stat_level_1["insertCEntry-3"]      = std::vector<ElapsedPair>();

        stat_level_2["_evictVecs-1"]        = std::vector<ElapsedPair>();
        stat_level_2["_evictVecs-2"]        = std::vector<ElapsedPair>();
        stat_level_2["_evictVecs-3"]        = std::vector<ElapsedPair>();

        directory_path = "/tmp/";
        std::string trace_dir = "/tmp/topkache-trace-";
        
        std::time_t t = std::time(nullptr);
        std::tm tm;

        localtime_r(&t, &tm);

        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");

        directory_path = trace_dir + oss.str();

        // Read the envirnoment variable, TOPKACHE_TRACE_DIR
        const char* env_trace_dir = std::getenv("TOPKACHE_TRACE_DIR");
        if (env_trace_dir != nullptr)
        {
            directory_path = "/tmp/" + std::string(env_trace_dir);
            fprintf(stdout, "Using TOPKACHE_TRACE_DIR: %s\n", directory_path.c_str());
            fflush(stdout);
        }

        clear();
    }

    void
    ResultCacheStats::exportAll() noexcept
    {
        std::vector<std::map<std::string, std::vector<ElapsedPair>>*> all_stats{
            &stat_level_0,
            &stat_level_1,
            &stat_level_2
        };

        printf("Exporting latency measurement...\n");


        // Make the subdirectory
        mkdir(directory_path.c_str(), 0777);

        // Export the latency measurement
        for (auto& stat_level : all_stats)
        {
            for (auto& pair : *stat_level)
            {
                std::string file_name = directory_path + "/" + pair.first + "-raw.csv";
                std::ofstream file(file_name, std::ios::out);

                printf("    Exporting %s\n", file_name.c_str());
                
                std::vector<double> elapsed_ms;
                if (file.is_open())
                {
                    
                    for (auto& elapsed_pair : pair.second)
                    {
                        elapsed_pair.elapsedMs();
                        
                        // Set floating point precision to 3
                        file << std::fixed << std::setprecision(3);
                        file << elapsed_pair.getElapsedMs() << 
                            "\t" << elapsed_pair.getAux1() << "\t" << elapsed_pair.getAux2() << "\n";

                        elapsed_ms.push_back(elapsed_pair.getElapsedMs());
                    }

                    file.close();
                }

                // Make statistics: Average, Min, 1%ile, Median, 99%ile, Max
                file_name = directory_path + "/" + pair.first + "-stat.csv";
                file.open(file_name, std::ios::out);

                if (elapsed_ms.size() == 0)
                {
                    file.close();
                    continue;
                }

                // Sort ascending order
                std::sort(elapsed_ms.begin(), elapsed_ms.end());
                
                double avg_ms           = std::accumulate(elapsed_ms.begin(), elapsed_ms.end(), 0.0) / elapsed_ms.size();

                double min_ms           = elapsed_ms[0];
                double max_ms           = elapsed_ms[pair.second.size() - 1];

                double total_count      = elapsed_ms.size();

                double median_ms        = elapsed_ms[total_count / 2];
                double percentile_1_ms  = elapsed_ms[total_count / 100];
                double percentile_99_ms = elapsed_ms[total_count - (total_count / 100)];

                file << "Avg\tMin\t1%ile\tMedian\t99%ile\tMax\n";
                file << avg_ms << "\t" << min_ms << "\t" << percentile_1_ms << "\t" << median_ms << "\t" << percentile_99_ms << "\t" << max_ms << "\n";

                file.close();                    
            }
        }

        // Export the hit ratio history
        {
            std::string file_name = directory_path + "/" + "hit_ratio_history.csv";
            std::ofstream file(file_name, std::ios::out);

            if (file.is_open())
            {
                file << "Hit Ratio Exact\tHit Ratio Total\n";
                for (size_t i = 0; i < cache_exact_hits.size(); i++)
                {
                    file << cache_exact_hits[i] << "\t" << cache_tot_hits[i] << "\n";
                }
                file.close();
            }
            else
            {
                printf("Failed to open %s for writing.\n", file_name.c_str());
            }
        }

        { 
            std::string file_name = directory_path + "/" + "apprx_added_nrepr.csv";
            std::ofstream file(file_name, std::ios::out);

            if (file.is_open())
            {
                file << "Approx Added\tApprox NRepr\n";
                for (size_t i = 0; i < apprx_added.size(); i++)
                {
                    file << apprx_added[i] << "\t" << apprx_nrepr[i] << "\n";
                }
                file.close();
            }
            else
            {
                printf("Failed to open %s for writing.\n", file_name.c_str());
            }
        }

        {
            std::string file_name = directory_path + "/" + "global_threshold_history.csv";
            std::ofstream file(file_name, std::ios::out);

            if (file.is_open())
            {
                file << "Global Threshold\n";
                for (size_t i = 0; i < global_threshold_history.size(); i++)
                {
                    file << global_threshold_history[i] << "\n";
                }
                file.close();
            }
        }

        printf("Exported latency measurement.\n");
    }

    void
    ResultCacheStats::printAll() noexcept
    {
        std::vector<std::map<std::string, std::vector<ElapsedPair>>*> all_stats{
            &stat_level_0,
            &stat_level_1,
            &stat_level_2
        };

// Export the latency measurement
        for (auto& stat_level : all_stats)
        {
            for (auto& pair : *stat_level)
            {
                printf(" >>> STAT-BEGIN(%s)<<<\n", pair.first.c_str());
                std::vector<double> elapsed_ms;
                    
                for (auto& elapsed_pair : pair.second)
                {
                    elapsed_pair.elapsedMs();
                    printf("%.3f ms (aux1: %lu, aux2: %lu)\n", elapsed_pair.getElapsedMs(), elapsed_pair.getAux1(), elapsed_pair.getAux2());
                    elapsed_ms.push_back(elapsed_pair.getElapsedMs());
                }

                if (elapsed_ms.size() == 0)
                {
                    continue;
                }

                // Sort ascending order
                std::sort(elapsed_ms.begin(), elapsed_ms.end());
                
                double avg_ms           = std::accumulate(elapsed_ms.begin(), elapsed_ms.end(), 0.0) / elapsed_ms.size();

                double min_ms           = elapsed_ms[0];
                double max_ms           = elapsed_ms[pair.second.size() - 1];

                double total_count      = elapsed_ms.size();

                double median_ms        = elapsed_ms[total_count / 2];
                double percentile_1_ms  = elapsed_ms[total_count / 100];
                double percentile_99_ms = elapsed_ms[total_count - (total_count / 100)];
                
                printf("Statistics for %s:\n", pair.first.c_str());
                printf("    Avg: %.3f ms\n", avg_ms);
                printf("    Min: %.3f ms\n", min_ms);
                printf("    1%%ile: %.3f ms\n", percentile_1_ms);
                printf("    Median: %.3f ms\n", median_ms);
                printf("    99%%ile: %.3f ms\n", percentile_99_ms);
                printf("    Max: %.3f ms\n", max_ms);         
                
                printf(" >>> STAT-END <<<\n");
            }
        }

        // Export the hit ratio history
        {
            printf(" >>> HIT-RATIO-HISTORY-BEGIN <<<\n");
            for (size_t i = 0; i < cache_exact_hits.size(); i++)
            {
                printf("%.3f\t%.3f\n", cache_exact_hits[i], cache_tot_hits[i]);
            }
        }
    }

    ResultCacheStats::~ResultCacheStats()
    {
        exportAll();
    }
}