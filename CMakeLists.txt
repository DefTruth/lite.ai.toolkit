cmake_minimum_required(VERSION 3.17)
project(litehub)

message(">>>> Current project is: ${CMAKE_CURRENT_SOURCE_DIR}")

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_BUILD_TYPE debug)

set(THIRDPARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

set(LITEHUB_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

option(LITEHUB_THIRDPARTY "use third party libraies if exists" ON)
option(LITEHUB_INSTALL_SDK "whether to install litehub sdk." OFF)  # don't make it ON !
option(LITEHUB_COPY_BUILD "whether to copy the built libs to ./build directory." ON)
option(LITEHUB_BUILD_LIB "whether to build shared|static libraries." ON)
option(LITEHUB_BUILD_TEST "whether to build test examples" ON)
option(LITEHUB_BUILD_TEST_SRCS "whether to build test examples from sources" OFF)  # link liblitehub or not
# engine backend -> config.h.in
option(ENABLE_ONNXRUNTIME "whether to enable ONNXRuntime engine" ON)
option(ENABLE_MNN "whether to enable MNN engine" OFF)
option(ENABLE_NCNN "whether to enable NCNN engine" OFF)
option(ENABLE_TNN "whether to enable TNN engine" OFF)
option(BACKEND_ONNXRUNTIME "whether to set ONNXRuntime as the main backend of litehub" ON)
option(BACKEND_MNN "whether to set MNN as the main backend of litehub" OFF)
option(BACKEND_NCNN "whether to set NCNN as the main backend of litehub" OFF)
option(BACKEND_TNN "whether to set TNN as the main backend of litehub" OFF)

include_directories(${LITEHUB_ROOT_DIR})

if (LITEHUB_BUILD_LIB)
    include(litehub.cmake)
endif()

if (LITEHUB_BUILD_TEST)
    add_subdirectory(examples/ort)
    add_subdirectory(examples/lite)
    # add_subdirectory(examples/ncnn)
    # add_subdirectory(examples/mnn)
    # add_subdirectory(examples/tnn)
endif()