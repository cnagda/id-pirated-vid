option(BUILD_OPTIONAL "Build testing binaries" OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_executable(add add.cxx)
add_executable(query query.cxx)

if(BUILD_OPTIONAL)
    add_executable(dataflow kernel.cxx)
    add_executable(keyframes keyframes.cxx)
    target_link_libraries( keyframes database )
    target_link_libraries( dataflow kernel database )
endif()

target_link_libraries( add database )
target_link_libraries( query database )
