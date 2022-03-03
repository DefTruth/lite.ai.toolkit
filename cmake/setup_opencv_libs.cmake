# 1. setup 3rd-party dependencies
message(STATUS "Setting up OpenCV libs for: ${CMAKE_CURRENT_SOURCE_DIR}")

if(NOT WIN32)
    if(ENABLE_OPENCV_VIDEOIO)
        set(OpenCV_LIBS opencv_core opencv_imgproc opencv_imgcodecs opencv_video opencv_videoio)
    else()
        set(OpenCV_LIBS opencv_core opencv_imgproc opencv_imgcodecs) # no videoio, video module
    endif()
else()
    set(OpenCV_LIBS opencv_world452)
endif()

message(STATUS "Setting up OpenCV libs done! OpenCV_LIBS:+[${OpenCV_LIBS}]")
