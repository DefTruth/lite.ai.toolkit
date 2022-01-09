cmake_minimum_required(VERSION 3.17)
project(lite.ai.toolkit)

set(CMAKE_CXX_STANDARD 11)
set(VERSION_STRING 0.1.0)
set(SOVERSION_STRING 0.1.0)
include(cmake/lite.ai.toolkit-platform.cmake)  # checking platform

message("==================================== Lite.AI.ToolKit ${VERSION_STRING} =============================")
message("Project: lite.ai.toolkit")
message("Version: ${VERSION_STRING}")
message("SO Version: ${SOVERSION_STRING}")
message("Build Type: ${CMAKE_BUILD_TYPE}")
message("Platform Name: ${PLATFORM_NAME}")
message("Root Path: ${CMAKE_SOURCE_DIR}")
message("================================================================================")

# root dir
set(LITE_AI_ROOT_DIR ${CMAKE_SOURCE_DIR})
# dir for built files.
set(BUILD_LITE_AI_DIR ${LITE_AI_ROOT_DIR}/build/lite.ai.toolkit)
set(LIBRARY_OUTPUT_PATH ${BUILD_LITE_AI_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${BUILD_LITE_AI_DIR}/bin)

# options
option(LITE_AI_BUILD_LIB "whether to build shared libraries." ON) # now, ON only
option(LITE_AI_BUILD_TEST "whether to build shared libraries." ON) # now, ON only
option(INCLUDE_OPENCV "whether to include OpenCV into lite.ai.toolkit." ON)
# engine backend: config.h.in -> config.h
option(ENABLE_ONNXRUNTIME "whether to enable ONNXRuntime engine" ON)
option(ENABLE_MNN "whether to enable MNN engine" ON)  # unstable now, DON'T use
option(ENABLE_NCNN "whether to enable NCNN engine" ON) # unstable now, DON'T use
option(ENABLE_TNN "whether to enable TNN engine" ON) # unstable now, DON'T use
option(BACKEND_ONNXRUNTIME "whether to set ONNXRuntime as the main backend of lite.ai.toolkit" ON)
option(BACKEND_MNN "whether to set MNN as the main backend of lite.ai.toolkit" OFF)  # now, OFF only
option(BACKEND_NCNN "whether to set NCNN as the main backend of lite.ai.toolkit" OFF)  # now, OFF only
option(BACKEND_TNN "whether to set TNN as the main backend of lite.ai.toolkit" OFF)  # now, OFF only

message("###################################  Engines Enable Details ...  #######################################")
message(STATUS "INCLUDE_OPENCV: ${INCLUDE_OPENCV}")
message(STATUS "ENABLE_ONNXRUNTIME: ${ENABLE_ONNXRUNTIME}")
message(STATUS "ENABLE_MNN: ${ENABLE_MNN}")
message(STATUS "ENABLE_NCNN: ${ENABLE_NCNN}")
message(STATUS "ENABLE_TNN: ${ENABLE_TNN}")
message("######################################################################################################")

# setup include dir and lib dir
include_directories(${LITE_AI_ROOT_DIR})
link_directories(${LITE_AI_ROOT_DIR}/lib)

# include custom cmake files.
include(cmake/setup_opencv_libs.cmake)
include(cmake/lite.ai.toolkit-cmd.cmake)

# configuration for lite.ai shared lib.
if (LITE_AI_BUILD_LIB)
    include(cmake/lite.ai.toolkit.cmake)
endif()

# configuration for test examples.
if (LITE_AI_BUILD_LIB AND LITE_AI_BUILD_TEST)
    add_subdirectory(examples/lite)
endif()
