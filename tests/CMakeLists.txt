include(CMakeLists.dependencies)

add_executable(database_test database_test.cpp)
add_executable(database_benchmark database_benchmark.cpp)
add_executable(sw_test sw_test.cpp)
add_executable(sw_benchmark sw_benchmark.cpp)
add_executable(query_test query_test.cpp)

target_link_libraries(database_test gtest_main database)
add_dependencies(database_test sample_videos)

target_link_libraries(query_test gtest_main database ${OpenCV_LIBS})
add_dependencies(query_test sample_videos)

target_link_libraries(sw_benchmark benchmark_main database)
target_link_libraries(database_benchmark benchmark_main database)
target_link_libraries(sw_test gtest_main database)

add_test(NAME db_test COMMAND database_test)
add_test(NAME sw_test COMMAND sw_test)
add_test(NAME query_test COMMAND query_test)