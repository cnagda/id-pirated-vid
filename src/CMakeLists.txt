find_package(OpenCV REQUIRED core videoio xfeatures2d)

add_library(instrumentation instrumentation.cpp)
add_library(database database.cpp bow.cpp)

target_link_libraries(database instrumentation ${OpenCV_LIBS} stdc++fs)
target_link_libraries(instrumentation ${OpenCV_LIBS} stdc++fs)
