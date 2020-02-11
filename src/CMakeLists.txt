find_package(OpenCV REQUIRED core videoio xfeatures2d)

add_library(database database.cpp bow.cpp)

target_link_libraries(database ${OpenCV_LIBS} stdc++fs)
