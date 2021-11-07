# 1. setup 3rd-party dependencies
message("########## Setting up OpenCV libs for: ${CMAKE_CURRENT_SOURCE_DIR} ###########")
set(OpenCV_LIBS
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
        opencv_video
        opencv_videoio
        ) # need only
message("###########################################################################################")

