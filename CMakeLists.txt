cmake_minimum_required(VERSION 3.17)
project(lite.ai)

set(CMAKE_CXX_STANDARD 11)
set(VERSION_STRING 0.0.1)
set(SOVERSION_STRING 0.0.1)

message("==================================== Lite.AI ${VERSION_STRING} =============================")
message("Project: Lite.AI")
message("Version: ${VERSION_STRING}")
message("SO Version: ${SOVERSION_STRING}")
message("Build Type: ${CMAKE_BUILD_TYPE}")
message("Root Path: ${CMAKE_SOURCE_DIR}")
message("================================================================================")

# dir for third parties.
set(LITE_AI_ROOT_DIR ${CMAKE_SOURCE_DIR})
set(THIRDPARTY_DIR ${LITE_AI_ROOT_DIR}/third_party)
# dir for built files.
set(BUILD_LITE_AI_DIR ${LITE_AI_ROOT_DIR}/build/lite.ai)
set(LIBRARY_OUTPUT_PATH ${BUILD_LITE_AI_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${BUILD_LITE_AI_DIR}/bin)

option(LITE_AI_THIRDPARTY "use third party libraries if exists" ON)
option(LITE_AI_COPY_BUILD "whether to copy the built libs to ./build directory." ON)
option(LITE_AI_BUILD_LIB "whether to build shared|static libraries." ON)
option(LITE_AI_BUILD_TEST "whether to build shared|static libraries." ON)
option(LITE_AI_BUILD_LITE_TEST "whether to build lite test examples" ON)
# engine backend: config.h.in -> config.h
option(ENABLE_ONNXRUNTIME "whether to enable ONNXRuntime engine" ON)
option(ENABLE_MNN "whether to enable MNN engine" OFF)
option(ENABLE_NCNN "whether to enable NCNN engine" OFF)
option(ENABLE_TNN "whether to enable TNN engine" OFF)
option(BACKEND_ONNXRUNTIME "whether to set ONNXRuntime as the main backend of lite.ai" ON)
option(BACKEND_MNN "whether to set MNN as the main backend of lite.ai" OFF)
option(BACKEND_NCNN "whether to set NCNN as the main backend of lite.ai" OFF)
option(BACKEND_TNN "whether to set TNN as the main backend of lite.ai" OFF)

include_directories(${LITE_AI_ROOT_DIR})

# include custom cmake files.
include(cmake/setup_3rdparty.cmake)
include(cmake/lite.ai-cmd.cmake)

# configuration for lite.ai shared lib.
if (LITE_AI_BUILD_LIB)
    include(cmake/lite.ai.cmake)
endif()

# configuration for test examples.
if (LITE_AI_BUILD_LIB AND LITE_AI_BUILD_TEST)

    if (LITE_AI_BUILD_LITE_TEST)
        add_subdirectory(examples/lite)
    endif()

endif()
