# 1. setup 3rd-party dependences
message(">>>> Current project is [ortcv_gender_googlenet] in : ${CMAKE_CURRENT_SOURCE_DIR}")
include(${CMAKE_SOURCE_DIR}/setup_3rdparty.cmake)

if (APPLE)
    set(CMAKE_MACOSX_RPATH 1)  # 开启rpath update qiuyanjun 20210314
    set(CMAKE_BUILD_TYPE release)
endif ()

# 2. setup onnxruntime include
include_directories(${ONNXRUNTIMR_INCLUDE_DIR})
link_directories(${ONNXRUNTIMR_LIBRARY_DIR})

# 3. CMakeLists.txt 在上一级目录 examples/ort
set(ORTCV_FSANET_SRCS
        cv/test_ortcv_gender_googlenet.cpp
        ${LITEHUB_ROOT_DIR}/ort/cv/gender_googlenet.cpp
        ${LITEHUB_ROOT_DIR}/ort/core/ort_utils.cpp
        ${LITEHUB_ROOT_DIR}/ort/core/ort_handler.cpp
        )

add_executable(ortcv_gender_googlenet ${ORTCV_FSANET_SRCS})
target_link_libraries(ortcv_gender_googlenet
        onnxruntime
        opencv_highgui
        opencv_core
        opencv_imgcodecs
        opencv_imgproc)

if (LITEHUB_COPY_BUILD)
    # set 只在当前目录和子目录有效 不会广播到父目录和兄弟目录
    # CMAKE_SOURCE_DIR是指顶层CMakeLists.txt所在目录
    # CMAKE_CURRENT_SOURCE_DIR是指当前CMakeLists.txt所在目录
    set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/build/liteort/bin)
    message("=================================================================================")
    message("output binary [app: ortcv_gender_googlenet] to ${EXECUTABLE_OUTPUT_PATH}")
    message("=================================================================================")
endif ()