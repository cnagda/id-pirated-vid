cmake_minimum_required(VERSION 3.10)

include(CMakeLists.dependencies)
include(GoogleTest)

add_executable(database_test database_test.cpp scene_test.cpp)
add_executable(database_benchmark database_benchmark.cpp)
add_executable(sw_benchmark sw_benchmark.cpp)
add_executable(query_test query_test.cpp sw_test.cpp classifier_test.cpp)

target_link_libraries(database_test gtest_main database)
target_link_libraries(query_test gtest_main database Eigen3::Eigen)
target_link_libraries(sw_benchmark benchmark_main database Eigen3::Eigen)
target_link_libraries(database_benchmark benchmark_main database)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg -g")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")
add_executable(sw_profile sw_profile.cxx)
target_link_libraries(sw_profile Eigen3::Eigen)

gtest_discover_tests(database_test)
gtest_discover_tests(query_test)