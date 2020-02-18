find_package(OpenCV REQUIRED core videoio xfeatures2d)
find_package(OpenMP)

add_library(database database.cpp matcher.cpp kmeans2.cpp instrumentation.cpp matrix.cpp)

target_link_libraries(database PUBLIC ${OpenCV_LIBS} stdc++fs PRIVATE OpenMP::OpenMP_CXX)
