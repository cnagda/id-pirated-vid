find_package(OpenCV REQUIRED core videoio xfeatures2d)
find_package(OpenMP)
find_package( TBB REQUIRED )

add_library(database 
    database.cpp 
    matcher.cpp 
    instrumentation.cpp 
    filter.cpp)

target_link_libraries(database PUBLIC ${OpenCV_LIBS}
    $<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.0>>:stdc++fs> 
    PRIVATE OpenMP::OpenMP_CXX TBB::tbb)
