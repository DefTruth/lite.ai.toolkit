set(OpenCV_Version "4.9.0-ffmpeg4.2.2" CACHE STRING "OpenCV version" FORCE)
set(OpenCV_DIR ${THIRD_PARTY_PATH}/opencv)
# download from github if opencv library is not exists
if (NOT EXISTS ${OpenCV_DIR})
    set(OpenCV_Filename "opencv-${OpenCV_Version}-linux-x86_64.tgz")
    set(OpenCV_URL https://github.com/DefTruth/lite.ai.toolkit/releases/download/v0.2.0-rc0/${OpenCV_Filename})
    message(STATUS "Downloading  library: ${OpenCV_URL}")
    download_and_decompress(${OpenCV_URL} ${OpenCV_Filename} ${OpenCV_DIR}) 
else() 
    message(STATUS "Found local OpenCV library: ${OpenCV_DIR}")
endif() 

message(STATUS "Setting up OpenCV libs for: ${CMAKE_CURRENT_SOURCE_DIR}")
if(NOT DEFINED OpenCV_DIR)
    message(FATAL_ERROR "OpenCV_DIR is not defined!")
endif()

include_directories(${OpenCV_DIR}/include/opencv4)
link_directories(${OpenCV_DIR}/lib)

if (NOT WIN32)
    if (ENABLE_OPENCV_VIDEOIO OR LITE_AI_BUILD_TEST)
        set(
                OpenCV_LIBS
                opencv_core
                opencv_imgproc
                opencv_imgcodecs
                opencv_video
                opencv_videoio
        )
    else ()
        set(
                OpenCV_LIBS
                opencv_core
                opencv_imgproc
                opencv_imgcodecs
        ) # no videoio, video module
    endif ()
else ()
    set(OpenCV_LIBS opencv_world490)
endif()

message(STATUS "Setting up OpenCV libs done! OpenCV_LIBS:+[${OpenCV_LIBS}]")
