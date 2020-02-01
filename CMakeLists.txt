cmake_minimum_required(VERSION 3.10)

project(Pirate_Detector VERSION 0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_TESTING "Build tests" OFF)

if(BUILD_TESTING)
    # Download and unpack googletest at configure time
    configure_file(CMakeLists.txt.googletest googletest/CMakeLists.txt)
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest)
    if(result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest)
    if(result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
    endif()


    # Download and unpack benchmark at configure time
    configure_file(CMakeLists.txt.benchmark benchmark/CMakeLists.txt)
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/benchmark)
    if(result)
    message(FATAL_ERROR "CMake step for benchmark failed: ${result}")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/benchmark)
    if(result)
    message(FATAL_ERROR "Build step for benchmark failed: ${result}")
    endif()

    # Prevent overriding the parent project's compiler/linker
    # settings on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    # Add googletest directly to our build. This defines
    # the gtest and gtest_main targets.
    add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                    ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
                    EXCLUDE_FROM_ALL)

    # Add benchmark directly to our build. This defines
    # the benchmark and benchmark_main targets.
    add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/benchmark-src
                    ${CMAKE_CURRENT_BINARY_DIR}/benchmark-build
                    EXCLUDE_FROM_ALL)

    # The gtest/gtest_main targets carry header search path
    # dependencies automatically when using CMake 2.8.11 or
    # later. Otherwise we have to add them here ourselves.
    if (CMAKE_VERSION VERSION_LESS 2.8.11)
    include_directories("${gtest_SOURCE_DIR}/include")
    endif()
endif(BUILD_TESTING)

find_package( OpenCV REQUIRED core videoio xfeatures2d highgui )

add_executable(main main.cxx)

target_link_libraries( main ${OpenCV_LIBS} )
