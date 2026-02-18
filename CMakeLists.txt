cmake_minimum_required(VERSION 3.10)

project(TopKcache LANGUAGES C CXX)

/**
 * Build configuration
 * - Use C++17
 * - Keep default build type as Debug for single-config generators
 */
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_VERBOSE_MAKEFILE ON)

# Use Debug as the default for single-config generators.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

/**
 * Compile options
 * - Preserve the existing behavior (warnings suppressed by default)
 * - Provide a switch to enable warnings when needed
 */
option(AKER_SUPPRESS_WARNINGS "Suppress compiler warnings (-w)" ON)

add_library(aker_build_options INTERFACE)

# Debug symbols and no optimization in Debug.
target_compile_options(aker_build_options INTERFACE
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0;-g3;-ggdb>
)

if(AKER_SUPPRESS_WARNINGS)
    target_compile_options(aker_build_options INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-w>
    )
else()
    target_compile_options(aker_build_options INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    )
endif()

/**
 * Dependencies
 */
find_package(Threads REQUIRED)

# Boost
set(BOOST_ROOT "/opt/boost_1_86" CACHE PATH "Boost root directory")
set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)

find_package(Boost 1.80 REQUIRED COMPONENTS program_options)

# FAISS
set(FAISS_ROOT "" CACHE PATH "FAISS installation prefix")

find_path(FAISS_INCLUDE_DIR
    NAMES faiss/Index.h
    PATHS
        "${FAISS_ROOT}/include"
        /usr/local/include
        /usr/include
    REQUIRED
)

find_library(FAISS_LIBRARY
    NAMES faiss
    PATHS
        "${FAISS_ROOT}/lib"
        "${FAISS_ROOT}/lib64"
        /usr/local/lib
        /usr/lib
        /usr/lib64
    REQUIRED
)

add_library(faiss::faiss UNKNOWN IMPORTED)
set_target_properties(faiss::faiss PROPERTIES
    IMPORTED_LOCATION "${FAISS_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${FAISS_INCLUDE_DIR}"
)

/**
 * Sources
 */
file(GLOB_RECURSE HDR_FILES
    CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/inc/*.h" "${PROJECT_SOURCE_DIR}/inc/*.hh"
    "${PROJECT_SOURCE_DIR}/inc/utils/*.h" "${PROJECT_SOURCE_DIR}/inc/utils/*.hh"
    "${PROJECT_SOURCE_DIR}/inc/core/*.h" "${PROJECT_SOURCE_DIR}/inc/core/*.hh"
)

file(GLOB_RECURSE SRC_FILES
    CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/src/*.c" "${PROJECT_SOURCE_DIR}/src/*.cc"
    "${PROJECT_SOURCE_DIR}/src/utils/*.c" "${PROJECT_SOURCE_DIR}/src/utils/*.cc"
    "${PROJECT_SOURCE_DIR}/src/core/*.c" "${PROJECT_SOURCE_DIR}/src/core/*.cc"
)

set(TOPKACHE_TARGET topkache)

/**
 * Object library
 * - Avoid compiling the same sources twice for shared/static
 */
add_library(${TOPKACHE_TARGET}_obj OBJECT ${SRC_FILES})
set_target_properties(${TOPKACHE_TARGET}_obj PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_features(${TOPKACHE_TARGET}_obj PUBLIC cxx_std_17)

target_include_directories(${TOPKACHE_TARGET}_obj
    PUBLIC
        "${PROJECT_SOURCE_DIR}/inc"
        "${PROJECT_SOURCE_DIR}/extern"
)

target_link_libraries(${TOPKACHE_TARGET}_obj
    PUBLIC
        aker_build_options
)

/**
 * Shared library
 */
add_library(${TOPKACHE_TARGET} SHARED $<TARGET_OBJECTS:${TOPKACHE_TARGET}_obj>)
set_target_properties(${TOPKACHE_TARGET} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build/lib"
    OUTPUT_NAME "topkache"
)

target_link_libraries(${TOPKACHE_TARGET}
    PUBLIC
        Threads::Threads
        Boost::program_options
        faiss::faiss
)

/**
 * Static library
 */
add_library(${TOPKACHE_TARGET}_static STATIC $<TARGET_OBJECTS:${TOPKACHE_TARGET}_obj>)
set_target_properties(${TOPKACHE_TARGET}_static PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build/lib"
    OUTPUT_NAME "topkache"
)

target_link_libraries(${TOPKACHE_TARGET}_static
    PUBLIC
        Threads::Threads
        Boost::program_options
        faiss::faiss
)

/**
 * Test and example binaries
 * - Compile only the test sources and link against the library
 */
option(TEST_BINARY "Build test binaries" ON)

if(TEST_BINARY)
    file(GLOB_RECURSE TEST_RUNS_HDRS
        CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/test/runs/*.h" "${PROJECT_SOURCE_DIR}/test/runs/*.hh"
    )

    add_executable(test-topkache
        "${PROJECT_SOURCE_DIR}/test/test-topkache.cc"
        "${PROJECT_SOURCE_DIR}/test/test-common.hh"
        ${TEST_RUNS_HDRS}
    )

    set_target_properties(test-topkache PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build/bin"
    )

    target_include_directories(test-topkache PUBLIC "${PROJECT_SOURCE_DIR}/inc")
    target_link_libraries(test-topkache PUBLIC ${TOPKACHE_TARGET})

    add_executable(test-topkache-c
        "${PROJECT_SOURCE_DIR}/test/test-topkache-c.c"
    )

    set_target_properties(test-topkache-c PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build/bin"
        LINKER_LANGUAGE CXX
    )

    target_include_directories(test-topkache-c PUBLIC "${PROJECT_SOURCE_DIR}/inc")
    target_link_libraries(test-topkache-c PUBLIC ${TOPKACHE_TARGET})

    add_executable(faiss-hnsw-microbenchmark
        "${PROJECT_SOURCE_DIR}/test/faiss-microbenchmark.cc"
    )

    set_target_properties(faiss-hnsw-microbenchmark PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build/bin"
    )

    target_include_directories(faiss-hnsw-microbenchmark PUBLIC "${PROJECT_SOURCE_DIR}/inc")
    target_link_libraries(faiss-hnsw-microbenchmark PUBLIC ${TOPKACHE_TARGET})
endif()

add_executable(app-sequence
    "${PROJECT_SOURCE_DIR}/test/app-sequence.cc"
)

set_target_properties(app-sequence PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build/bin"
)

target_include_directories(app-sequence PUBLIC "${PROJECT_SOURCE_DIR}/inc")
target_link_libraries(app-sequence PUBLIC ${TOPKACHE_TARGET})
