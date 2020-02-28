cmake_minimum_required(VERSION 3.10)

project(Pirate_Detector VERSION 0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

include(CMakeLists.dependencies)

include_directories(include)

enable_testing()

option(BUILD_TESTING "Build tests" OFF)
option(PRINT_DEBUG "Add debug messages" OFF)

if(PRINT_DEBUG)
    add_definitions(-DSW_COUT)
endif(PRINT_DEBUG)

add_subdirectory(src)

add_custom_target(sample_videos cp ${CMAKE_CURRENT_SOURCE_DIR}/sample.mp4 ${CMAKE_CURRENT_SOURCE_DIR}/coffee.mp4 ${CMAKE_CURRENT_SOURCE_DIR}/crab.mp4 ${CMAKE_CURRENT_BINARY_DIR})
add_custom_target(python_scripts cp -R ${CMAKE_CURRENT_SOURCE_DIR}/python ${CMAKE_CURRENT_BINARY_DIR})

if(BUILD_TESTING)
    add_subdirectory(tests)
endif(BUILD_TESTING)

add_executable(add add.cxx)
add_executable(query query.cxx)
add_executable(bench bench.cxx)
add_executable(keyframes keyframes.cxx)

target_link_libraries( add database )
target_link_libraries( bench database )
target_link_libraries( keyframes database )

add_dependencies(add sample_videos)
add_dependencies(query python_scripts)

target_link_libraries( query database )
