cmake_minimum_required(VERSION 3.10)
project(lite.ai.toolkit)

set(CMAKE_CXX_STANDARD 17)
set(VERSION_STRING 0.2.0)
set(SOVERSION_STRING 0.2.0)
include(cmake/utils.cmake)

if (NOT (UNIX AND NOT APPLE))
  message(FATAL_ERROR "lite.ai.toolkit>=0.2 not support for windows/mac now!")
endif()

set(THIRD_PARTY_PATH "${CMAKE_SOURCE_DIR}/third_party")
if(NOT EXISTS ${THIRD_PARTY_PATH})
    file(MAKE_DIRECTORY ${THIRD_PARTY_PATH})
endif()

# Linux GCC Compiler Options
if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-std=c++17 -Wno-deprecated ${CMAKE_CXX_FLAGS} ")
    message(STATUS "[Linux GCC Compiler Options]+:-std=c++17 -Wno-deprecated")
endif ()
# message(STATUS "CMAKE_CXX_COMPILER: [${CMAKE_CXX_COMPILER}]")

# root dir
set(LITE_AI_ROOT_DIR ${CMAKE_SOURCE_DIR})

# compile options for lite.ai.toolkit
option(ENABLE_TEST "build test examples." OFF)
# inference engines setups: config.h.in -> config.h
option(ENABLE_DEBUG_STRING "enable DEBUG string or not" ON)
option(ENABLE_ONNXRUNTIME "enable ONNXRuntime engine" ON)
option(ENABLE_MNN "enable MNN engine" OFF)  # unstable now, DON'T use
option(ENABLE_NCNN "enable NCNN engine" OFF) # unstable now, DON'T use
option(ENABLE_TNN "enable TNN engine" OFF) # unstable now, DON'T use
# cuda provider setups: config.h.in -> config.h (only for onnxruntime)
option(ENABLE_ONNXRUNTIME_CUDA "enable ONNXRuntime engine with CUDA provider" OFF)
option(ENABLE_OPENCV_VIDEOIO "enable opencv videoio modules for detect_video apis" ON) # now, ON only

if ((NOT ENABLE_ONNXRUNTIME) AND (NOT ENABLE_MNN))
    message(FATAL_ERROR "One of ONNXRuntime/MNN Backend must be enable!")
endif()

# setup include dir and lib dir
include_directories(${LITE_AI_ROOT_DIR})
configure_file(cmake/lite.ai.toolkit.cmake.in ${CMAKE_SOURCE_DIR}/cmake/lite.ai.toolkit.cmake @ONLY)

# include custom cmake files.
include(cmake/opencv.cmake)

add_lite_ai_toolkit_shared_library(${VERSION_STRING} ${SOVERSION_STRING})
install(TARGETS lite.ai.toolkit LIBRARY DESTINATION lib)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/lite 
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include
        FILES_MATCHING
        PATTERN "*.h")
install(DIRECTORY ${THIRD_PARTY_PATH} DESTINATION ${CMAKE_INSTALL_PREFIX})
install(FILES ${CMAKE_SOURCE_DIR}/cmake/lite.ai.toolkit.cmake 
        DESTINATION ${CMAKE_INSTALL_PREFIX})
install(FILES ${CMAKE_SOURCE_DIR}/cmake/lite.ai.toolkit-config.cmake 
        DESTINATION ${CMAKE_INSTALL_PREFIX})

# configuration for test examples.
if (ENABLE_TEST)
    set(EXECUTABLE_OUTPUT_PATH ${CMAKE_INSTALL_PREFIX}/bin)
    add_subdirectory(examples/lite)
    if ((UNIX AND NOT APPLE))
        file(GLOB_RECURSE ALL_THIRD_LIBS ${THIRD_PARTY_PATH} FOLLOW_SYMLINKS *.so*)
        install(FILES ${ALL_THIRD_LIBS} DESTINATION ${EXECUTABLE_OUTPUT_PATH})
    endif()
endif ()

message(STATUS "-------------------------- lite.ai.toolkit Configuration Summary --------------------------")
message(STATUS "        Version: ${VERSION_STRING}")
message(STATUS "     SO Version: ${SOVERSION_STRING}")
message(STATUS "     Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "      Root Path: ${CMAKE_SOURCE_DIR}")
message(STATUS "         OpenCV: ON Version: ${OpenCV_Version}")
message(STATUS "    ONNXRUNTIME: ${ENABLE_ONNXRUNTIME} Version: ${OnnxRuntime_Version}")
message(STATUS "            MNN: ${ENABLE_MNN} Version: ${MNN_Version}")
message(STATUS "           NCNN: ${ENABLE_NCNN} Version: ${NCNN_Version}")
message(STATUS "            TNN: ${ENABLE_TNN} Version: ${TNN_Version}")
message(STATUS "-------------------------- lite.ai.toolkit Configuration Summary --------------------------")
