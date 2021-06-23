# 1. setup 3rd-party dependences
message(">>>> Current project is [ortcv_resnext] in : ${CMAKE_CURRENT_SOURCE_DIR}")
include(${CMAKE_SOURCE_DIR}/setup_3rdparty.cmake)

if (APPLE)
    set(CMAKE_MACOSX_RPATH 1)
    set(CMAKE_BUILD_TYPE release)
endif ()

# 2. setup onnxruntime include
include_directories(${ONNXRUNTIME_INCLUDE_DIR})
link_directories(${ONNXRUNTIME_LIBRARY_DIR})

# 3. will be include into CMakeLists.txt at examples/ort
set(ORTCV_RESNEXT_SRCS
        cv/test_ortcv_resnext.cpp
        ${LITEHUB_ROOT_DIR}/ort/cv/resnext.cpp
        ${LITEHUB_ROOT_DIR}/ort/core/ort_utils.cpp
        ${LITEHUB_ROOT_DIR}/ort/core/ort_handler.cpp
        )

add_executable(ortcv_resnext ${ORTCV_RESNEXT_SRCS})
target_link_libraries(ortcv_resnext onnxruntime ${OpenCV_LIBS})

if (LITEHUB_COPY_BUILD)
    # "set" only valid in the current directory and subdirectory and does not broadcast
    # to parent and sibling directories
    # CMAKE_SOURCE_DIR means the root path of top CMakeLists.txt
    # CMAKE_CURRENT_SOURCE_DIR the current path of current CMakeLists.txt
    set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/build/litehub/bin)
    message("=================================================================================")
    message("output binary [app: ortcv_resnext] to ${EXECUTABLE_OUTPUT_PATH}")
    message("=================================================================================")
endif ()