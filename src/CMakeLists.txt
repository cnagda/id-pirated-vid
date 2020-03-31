find_package(OpenCV REQUIRED core videoio xfeatures2d)
find_package(OpenMP)

add_library(database database.cpp 
    matcher.cpp 
    instrumentation.cpp 
    kmeans2.cpp
    matrix.cpp)

add_library(kernel kernel.cpp)

target_link_libraries(kernel PUBLIC raft PRIVATE ${OpenCV_LIBS} )
target_link_libraries(database PUBLIC ${OpenCV_LIBS} stdc++fs PRIVATE OpenMP::OpenMP_CXX)
