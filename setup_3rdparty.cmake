# 1. setup 3rd-party dependences
message("########## Setting up 3rd-party dependences for: ${CMAKE_CURRENT_SOURCE_DIR} ###########")
if (EXISTS ${THIRDPARTY_DIR} AND LITEHUB_THIRDPARTY)
    # 由于rpath无法修改 因此以下设置暂时无用
#    set(OpenCV_DIR ${THIRDPARTY_DIR}/opencv/4.1.2/lib/cmake/opencv4)
#
#    set(ONNXRUNTIME_DIR ${THIRDPARTY_DIR}/onnxruntime/1.7.1)
#    set(ONNXRUNTIMR_INCLUDE_DIR ${ONNXRUNTIME_DIR}/include)
#    set(ONNXRUNTIMR_LIBRARY_DIR ${ONNXRUNTIME_DIR}/lib)

    set(OpenCV_DIR /usr/local/Cellar/opencv/4.5.1_3/lib/cmake/opencv4)
    set(ONNXRUNTIME_DIR /usr/local/Cellar/onnxruntime/1.7.1)
    set(ONNXRUNTIMR_INCLUDE_DIR ${ONNXRUNTIME_DIR}/include)
    set(ONNXRUNTIMR_LIBRARY_DIR ${ONNXRUNTIME_DIR}/lib)

else ()
    set(OpenCV_DIR /usr/local/Cellar/opencv/4.5.1_3/lib/cmake/opencv4)
    set(ONNXRUNTIME_DIR /usr/local/Cellar/onnxruntime/1.7.1)
    set(ONNXRUNTIMR_INCLUDE_DIR ${ONNXRUNTIME_DIR}/include)
    set(ONNXRUNTIMR_LIBRARY_DIR ${ONNXRUNTIME_DIR}/lib)
endif ()

# 2. check if OpenCV is available.
find_package(OpenCV 4 REQUIRED)

if (OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    message(${OpenCV_INCLUDE_DIRS})
    message(${OpenCV_LIBRARIES})
    message("=================================================================================")
    message(STATUS "    OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    libraries: ${OpenCV_LIBS}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
    message("=================================================================================")
else (OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV library not found")
endif (OpenCV_FOUND)
