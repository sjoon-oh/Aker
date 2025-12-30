#include "test-common.hh"

#include "runs/test_exact_search_approx_insert.hh"
#include "runs/test_write_logs.hh"
#include "runs/test_deletes.hh"

#include "runs/test_scratchpad.hh"


typedef     bool (*test_func_t)(void);

int 
main()
{
    test_func_t tests[] = {
        // test_get_and_insert,
        // test_exact_search_approx_insert,
        // test_similar_postlinks_exhaustive,
        // test_similar_postlinks_approx,
        // test_similarity_get,
        test_write_logs,
        // test_write_logs_small,
        // test_deletes_1,
        // test_deletes_2,
        // test_scratchpad
    };

    topkache::Logger logger("MAIN");

    size_t test_count = sizeof(tests) / sizeof(test_func_t);
    for (size_t i = 0; i < test_count; i++)
    {   
        logger.getLogger()->info("Running test {}", i);
        bool success = tests[i]();
        if (success)
        {
            logger.getLogger()->info("Test {} passed", i);
        }
        else
        {
            logger.getLogger()->info("Test {} failed", i);
        }
    
        printf("\n\n");
    }

    return 0;
}

