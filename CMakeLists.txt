cmake_minimum_required(VERSION 3.17)
project(litehub)

set(CMAKE_CXX_STANDARD 11)
set(VERSION_STRING 0.0.1)
set(SOVERSION_STRING 0.0.1)

message("==================================== LiteHub ${VERSION_STRING} =============================")
message("Project: LiteHub")
message("Version: ${VERSION_STRING}")
message("SO Version: ${SOVERSION_STRING}")
message("Build Type: ${CMAKE_BUILD_TYPE}")
message("Root Path: ${CMAKE_SOURCE_DIR}")
message("================================================================================")

# dir for third parties.
set(LITEHUB_ROOT_DIR ${CMAKE_SOURCE_DIR})
set(THIRDPARTY_DIR ${LITEHUB_ROOT_DIR}/third_party)
# dir for built files.
set(BUILD_LITEHUB_DIR ${LITEHUB_ROOT_DIR}/build/litehub)
set(LIBRARY_OUTPUT_PATH ${BUILD_LITEHUB_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${BUILD_LITEHUB_DIR}/bin)

option(LITEHUB_THIRDPARTY "use third party libraies if exists" ON)
option(LITEHUB_COPY_BUILD "whether to copy the built libs to ./build directory." ON)
option(LITEHUB_BUILD_LIB "whether to build shared|static libraries." ON)
option(LITEHUB_BUILD_TEST "whether to build shared|static libraries." ON)
option(LITEHUB_BUILD_ORT_TEST "whether to build ort test examples" OFF)
option(LITEHUB_BUILD_LITE_TEST "whether to build lite test examples" ON)
# engine backend: config.h.in -> config.h
option(ENABLE_ONNXRUNTIME "whether to enable ONNXRuntime engine" ON)
option(ENABLE_MNN "whether to enable MNN engine" OFF)
option(ENABLE_NCNN "whether to enable NCNN engine" OFF)
option(ENABLE_TNN "whether to enable TNN engine" OFF)
option(BACKEND_ONNXRUNTIME "whether to set ONNXRuntime as the main backend of litehub" ON)
option(BACKEND_MNN "whether to set MNN as the main backend of litehub" OFF)
option(BACKEND_NCNN "whether to set NCNN as the main backend of litehub" OFF)
option(BACKEND_TNN "whether to set TNN as the main backend of litehub" OFF)

include_directories(${LITEHUB_ROOT_DIR})

# include custom cmake files.
include(setup_3rdparty.cmake)
include(litehub-command.cmake)

# configuration for litehub shared lib.
if (LITEHUB_BUILD_LIB)
    include(litehub.cmake)
endif()

# configuration for test examples.
if (LITEHUB_BUILD_LIB AND LITEHUB_BUILD_TEST)

    if (LITEHUB_BUILD_ORT_TEST)
        add_subdirectory(examples/ort)
    endif()

    if (LITEHUB_BUILD_LITE_TEST)
        add_subdirectory(examples/lite)
    endif()

endif()
