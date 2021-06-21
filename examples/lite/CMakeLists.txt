option(BUILD_LITE_YOLOV5 "whether to build lite_yolov5 binary." ON)

if (BUILD_LITE_YOLOV5)
    include(cv/test_lite_yolov5.cmake)
endif()