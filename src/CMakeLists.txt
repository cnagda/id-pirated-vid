find_package(OpenCV REQUIRED core videoio xfeatures2d)
find_package(OpenMP)

add_library(database database.cpp matcher.cpp kmeans2.cpp instrumentation.cpp)

target_link_libraries(database PRIVATE OpenMP::OpenMP_CXX ${OpenCV_LIBS} stdc++fs)
