cmake_minimum_required(VERSION 3.17)
project(litehub)

message(">>>> Current project is: ${CMAKE_CURRENT_SOURCE_DIR}")

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_BUILD_TYPE debug)

set(THIRDPARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)

set(LITEHUB_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

option(LITEHUB_THIRDPARTY "use third party libraies if exists" ON)
option(LITEHUB_INSTALL_SDK "whether to install litehub sdk." OFF)  # don't make it ON !
option(LITEHUB_COPY_BUILD "whether to copy the built libs to ./build directory." ON)

option(LITEHUB_BUILD_LIB "whether to build shared|static libraries." ON)
option(LITEHUB_BUILD_TEST "whether to build test examples" ON)

include_directories(${LITEHUB_ROOT_DIR})

if (LITEHUB_BUILD_LIB)
    add_subdirectory(ort)
    add_subdirectory(ncnn)
endif()

if (LITEHUB_BUILD_TEST)
    add_subdirectory(examples/ort)
    # add_subdirectory(examples/ncnn)
    # add_subdirectory(examples/mnn)
    # add_subdirectory(examples/tnn)
endif()