# 1. setup 3rd-party dependencies
message("########## Setting up 3rd-party dependencies for: ${CMAKE_CURRENT_SOURCE_DIR} ###########")
set(THIRDPARTY_SET_STATE OFF)

if (EXISTS ${THIRDPARTY_DIR} AND LITE_AI_THIRDPARTY)

    message("Setting Up Custom Dependencies ...")

    set(OpenCV_DIR ${THIRDPARTY_DIR}/${PLATFORM_NAME}/opencv/4.5.2/x86_64/lib/cmake/opencv4)
    set(OpenCV_LIBRARY_DIR ${THIRDPARTY_DIR}/${PLATFORM_NAME}/opencv/4.5.2/x86_64/lib)

    if (ENABLE_ONNXRUNTIME)
        set(ONNXRUNTIME_DIR ${THIRDPARTY_DIR}/${PLATFORM_NAME}/onnxruntime/1.7.0/x86_64)
        set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_DIR}/include)
        set(ONNXRUNTIME_LIBRARY_DIR ${ONNXRUNTIME_DIR}/lib)
        include_directories(${ONNXRUNTIME_INCLUDE_DIR})
        link_directories(${ONNXRUNTIME_LIBRARY_DIR})
    endif()
    if (ENABLE_MNN)
        set(MNN_DIR ${THIRDPARTY_DIR}/${PLATFORM_NAME}/1.2.0/MNN/x86_64)
        set(MNN_INCLUDE_DIR ${MNN_DIR}/include)
        set(MNN_LIBRARY_DIR ${MNN_DIR}/lib)
        include_directories(${MNN_INCLUDE_DIR})
        link_directories(${MNN_LIBRARY_DIR})
    endif()
    if (ENABLE_NCNN)
        set(NCNN_DIR ${THIRDPARTY_DIR}/${PLATFORM_NAME}/ncnn/x86_64)
        set(NCNN_INCLUDE_DIR ${NCNN_DIR}/include)
        set(NCNN_LIBRARY_DIR ${NCNN_DIR}/lib)
        include_directories(${NCNN_INCLUDE_DIR})
        link_directories(${NCNN_LIBRARY_DIR})
    endif()
    if (ENABLE_TNN)
        set(TNN_DIR ${THIRDPARTY_DIR}/${PLATFORM_NAME}/TNN/x86_64)
        set(TNN_INCLUDE_DIR ${TNN_DIR}/include)
        set(TNN_LIBRARY_DIR ${TNN_DIR}/lib)
        include_directories(${TNN_INCLUDE_DIR})
        link_directories(${TNN_LIBRARY_DIR})
    endif()

else ()

    message(FATAL_ERROR "Dependencies Setting Up Error!")

endif ()

# 2. check if OpenCV is available.

if (OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    set(OpenCV_LIBS opencv_highgui opencv_core opencv_imgcodecs opencv_imgproc opencv_video opencv_videoio) # need only
    message("=================================================================================")
    message(STATUS "    OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    libraries: ${OpenCV_LIBS}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
    message("=================================================================================")
else ()
    # find at first time
    find_package(OpenCV 4 REQUIRED)
    if (OpenCV_FOUND)
        include_directories(${OpenCV_INCLUDE_DIRS})
        set(OpenCV_LIBS opencv_highgui opencv_core opencv_imgcodecs opencv_imgproc opencv_video opencv_videoio) # need only
        message("=================================================================================")
        message(STATUS "    OpenCV library status:")
        message(STATUS "    version: ${OpenCV_VERSION}")
        message(STATUS "    libraries: ${OpenCV_LIBS}")
        message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
        message("=================================================================================")
    else ()
     message(FATAL_ERROR "OpenCV library not found")
    endif ()

endif ()

set(THIRDPARTY_SET_STATE ON)
