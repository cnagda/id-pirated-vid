find_package(OpenCV REQUIRED core videoio xfeatures2d)
find_package(OpenMP)
find_package( TBB REQUIRED )

add_library(database database.cpp 
    matcher.cpp 
    instrumentation.cpp 
    kmeans2.cpp
    matrix.cpp)

if(BUILD_OPTIONAL)
    add_library(kernel kernel.cpp)
    target_link_libraries(kernel PUBLIC raft ${OpenCV_LIBS} )
endif()

target_link_libraries(database PUBLIC ${OpenCV_LIBS}
    $<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.0>>:stdc++fs> 
    PRIVATE OpenMP::OpenMP_CXX ${Boost_LIBRARIES} Eigen3::Eigen TBB::tbb)
