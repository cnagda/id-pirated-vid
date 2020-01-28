cmake_minimum_required(VERSION 3.10)

project(Pirate_Detector VERSION 0.1)

find_package( OpenCV REQUIRED )

add_executable(main main.cxx)

target_link_libraries( main ${OpenCV_LIBS} )