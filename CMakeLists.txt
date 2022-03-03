cmake_minimum_required(VERSION 3.10)
project(lite.ai.toolkit)

set(CMAKE_CXX_STANDARD 11)
set(VERSION_STRING 0.1.1)
set(SOVERSION_STRING 0.1.1)
include(cmake/lite.ai.toolkit-platform.cmake)  # checking platform

message(STATUS "Lite.AI.ToolKit  ${VERSION_STRING}")
message(STATUS "Project: lite.ai.toolkit")
message(STATUS "Version: ${VERSION_STRING}")
message(STATUS "SO Version: ${SOVERSION_STRING}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Platform Name: ${PLATFORM_NAME}")
message(STATUS "Root Path: ${CMAKE_SOURCE_DIR}")

# Linux GCC Compiler Options
if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-std=c++11 -Wno-deprecated ${CMAKE_CXX_FLAGS} ")
    message(STATUS "[Linux GCC Compiler Options]+:-std=c++11 -Wno-deprecated")
endif ()


# root dir
set(LITE_AI_ROOT_DIR ${CMAKE_SOURCE_DIR})
# set default build dir for lite.ai.toolkit
if (NOT DEFINED BUILD_LITE_AI_DIR)
    set(BUILD_LITE_AI_DIR ${LITE_AI_ROOT_DIR}/build/lite.ai.toolkit)
endif ()
set(LIBRARY_OUTPUT_PATH ${BUILD_LITE_AI_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${BUILD_LITE_AI_DIR}/bin)

# compile options for lite.ai.toolkit
option(LITE_AI_BUILD_LIB "build shared libraries." ON) # now, ON only
option(LITE_AI_BUILD_TEST "build test examples." ON)
option(INCLUDE_OPENCV "package OpenCV into lite.ai.toolkit." ON)
# inference engines setups: config.h.in -> config.h
option(ENABLE_ONNXRUNTIME "enable ONNXRuntime engine" ON)
option(ENABLE_MNN "enable MNN engine" ON)  # unstable now, DON'T use
option(ENABLE_NCNN "enable NCNN engine" ON) # unstable now, DON'T use
option(ENABLE_TNN "enable TNN engine" ON) # unstable now, DON'T use
# cuda provider setups: config.h.in -> config.h (only for onnxruntime)
option(ENABLE_ONNXRUNTIME_CUDA "enable ONNXRuntime engine with CUDA provider" OFF)
# videoio interface setups, for future use
option(ENABLE_OPENCV_VIDEOIO "enable opencv videoio modules for detect_video apis" ON) # now, ON only
# inference engines backend setups for lite.ai.toolkit
option(BACKEND_ONNXRUNTIME "set ONNXRuntime as the main backend of lite.ai.toolkit" ON)
option(BACKEND_MNN "set MNN as the main backend of lite.ai.toolkit" OFF)  # now, OFF only
option(BACKEND_NCNN "set NCNN as the main backend of lite.ai.toolkit" OFF)  # now, OFF only
option(BACKEND_TNN "set TNN as the main backend of lite.ai.toolkit" OFF)  # now, OFF only

message(STATUS "Engines Enable Details ... ")
message(STATUS "INCLUDE_OPENCV: ${INCLUDE_OPENCV}")
message(STATUS "ENABLE_ONNXRUNTIME: ${ENABLE_ONNXRUNTIME}")
message(STATUS "ENABLE_MNN: ${ENABLE_MNN}")
message(STATUS "ENABLE_NCNN: ${ENABLE_NCNN}")
message(STATUS "ENABLE_TNN: ${ENABLE_TNN}")

# setup include dir and lib dir
include_directories(${LITE_AI_ROOT_DIR})
link_directories(${LITE_AI_ROOT_DIR}/lib/${PLATFORM_NAME})

# include custom cmake files.
include(cmake/setup_opencv_libs.cmake)
include(cmake/lite.ai.toolkit-cmd.cmake)

# configuration for lite.ai shared lib.
if (LITE_AI_BUILD_LIB)
    include(cmake/lite.ai.toolkit.cmake)
endif ()

# configuration for test examples.
if (LITE_AI_BUILD_LIB AND LITE_AI_BUILD_TEST)
    add_subdirectory(examples/lite)
endif ()
