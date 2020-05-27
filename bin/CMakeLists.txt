option(BUILD_OPTIONAL "Build testing binaries" OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_executable(add add.cxx)
add_executable(info info.cxx)
add_executable(query query.cxx)
add_executable(labscene migrate-labscene.cxx)

if(BUILD_OPTIONAL)
    add_executable(dataflow kernel.cxx)
    target_link_libraries( dataflow database )

    find_package(Boost REQUIRED COMPONENTS iostreams)

    add_executable(visualize visualize.cxx)
    target_link_libraries( visualize database ${Boost_LIBRARIES} )
endif()

target_link_libraries( add database )
target_link_libraries( info database )
target_link_libraries( query database )
target_link_libraries( labscene database )
