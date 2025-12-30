cmake_minimum_required(VERSION 3.10)
project(TopKcache)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_VERBOSE_MAKEFILE true)

add_compile_options(
    # "-Wall" "-Wextra" "$<$<CONFIG:DEBUG>:-O0;-g3;-ggdb>"
    "-w" "-Wextra" "$<$<CONFIG:DEBUG>:-O0;-g3;-ggdb>"
)
# cmake . -DCMAKE_BUILD_TYPE=Debug

set(CMAKE_BUILD_TYPE Debug)

set(BOOST_ROOT "/opt/boost_1_86")
set(Boost_INCLUDE_DIR "/opt/boost_1_86/include")
set(Boost_LIBRARY_DIR "/opt/boost_1_86/lib")

# Include boost library
find_package(
    Boost 1.80 REQUIRED 
    COMPONENTS program_options
)
# if (Boost_FOUND)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIR})

set(Boost_USE_STATIC_LIBS ON)
# endif()

# Add submodules
# add_subdirectory(extern/YCSB-C)

# External sources
include_directories(${PROJECT_SOURCE_DIR}/extern/spdlog/include)
include_directories(${PROJECT_SOURCE_DIR}/extern) # For YCSB-C, include all for extern directory

# Set FAISS library
find_library(FAISSLIB NAMES faiss PATHS /usr/local/lib REQUIRED)
# if (FAISSLIB)
#     message(STATUS "Found FAISS library: ${FAISSLIB}")
# else()
#     message(FATAL_ERROR "mylib not found")
# endif()

# Internal sources
# include_directories(${PROJECT_SOURCE_DIR}/inc)
file(GLOB_RECURSE HDR_FILES 
    CONFIGURE_DEPENDS
    inc/*.h inc/*.hh
    inc/utils/*.h inc/utils/*.hh
    inc/core/*.h inc/core/*.hh
    )
file(GLOB_RECURSE SRC_FILES
    CONFIGURE_DEPENDS
    src/*.c src/*.cc
    src/utils/*.c src/utils/*.cc
    src/core/*.c src/core/*.cc
    )


set(TOPKACHE topkache)
add_library(${TOPKACHE} 
    SHARED ${HDR_FILES} ${SRC_FILES}
)
set_target_properties(${TOPKACHE} 
    PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/lib
    LINKER_LANGUAGE CXX
)
target_include_directories(${TOPKACHE} 
    PUBLIC ${PROJECT_SOURCE_DIR}/inc
)
target_link_libraries(${TOPKACHE} 
    PUBLIC pthread ${Boost_LIBRARIES} faiss
)

add_library(${TOPKACHE}_static STATIC ${HDR_FILES} ${SRC_FILES})
set_target_properties(${TOPKACHE}_static PROPERTIES 
    OUTPUT_NAME "topkache"
    ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/lib
    LINKER_LANGUAGE CXX
)
target_include_directories(${TOPKACHE}_static PUBLIC ${PROJECT_SOURCE_DIR}/inc)
target_compile_options(${TOPKACHE}_static PUBLIC -fPIC)
target_link_libraries(${TOPKACHE}_static PUBLIC pthread ${Boost_LIBRARIES} faiss)

file(GLOB_RECURSE TEST_RUNS_HDRS 
    CONFIGURE_DEPENDS
    test/runs/*.h test/runs/*.hh
    )

# Test binary
option(TEST_BINARY "TEST-BINARY" ON)
if (TEST_BINARY)

    add_executable(
        test-topkache ${HDR_FILES} ${SRC_FILES}
            ${PROJECT_SOURCE_DIR}/test/test-common.hh
            ${TEST_RUNS_HDRS}
            ${PROJECT_SOURCE_DIR}/test/test-topkache.cc)
    set_target_properties(
        test-topkache PROPERTIES 
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/bin
        LINKER_LANGUAGE CXX
    )

    target_include_directories(test-topkache PUBLIC ${PROJECT_SOURCE_DIR}/inc)
    target_link_libraries(test-topkache PUBLIC ${Boost_LIBRARIES} ${TOPKACHE})
    add_dependencies(test-topkache ${Boost_LIBRARIES} ${TOPKACHE})

    add_executable(
        test-topkache-c ${HDR_FILES} ${SRC_FILES}
            ${PROJECT_SOURCE_DIR}/test/test-topkache-c.c)
    set_target_properties(
        test-topkache-c PROPERTIES 
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/bin
        LINKER_LANGUAGE C
    )

    target_include_directories(test-topkache-c PUBLIC ${PROJECT_SOURCE_DIR}/inc)
    target_link_libraries(test-topkache-c PUBLIC ${TOPKACHE})
    add_dependencies(test-topkache-c ${Boost_LIBRARIES} ${TOPKACHE})

    # 
    # 
    add_executable(
        faiss-hnsw-microbenchmark ${HDR_FILES} ${SRC_FILES}
            ${PROJECT_SOURCE_DIR}/test/faiss-microbenchmark.cc)
    
    set_target_properties(
        faiss-hnsw-microbenchmark PROPERTIES 
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/bin
        LINKER_LANGUAGE C
    ) 

    target_include_directories(faiss-hnsw-microbenchmark PUBLIC ${PROJECT_SOURCE_DIR}/inc)
    target_link_libraries(faiss-hnsw-microbenchmark PUBLIC ${TOPKACHE})
    add_dependencies(
        faiss-hnsw-microbenchmark ${TOPKACHE} 
    )

endif()

add_executable(
    app-sequence ${HDR_FILES} ${SRC_FILES}
        ${PROJECT_SOURCE_DIR}/test/app-sequence.cc
    )
set_target_properties(
    app-sequence PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/bin
    LINKER_LANGUAGE CXX
)

target_include_directories(app-sequence PUBLIC ${PROJECT_SOURCE_DIR}/inc)
target_link_libraries(app-sequence PUBLIC ${Boost_LIBRARIES} ${TOPKACHE})
add_dependencies(app-sequence ${Boost_LIBRARIES} ${TOPKACHE})
