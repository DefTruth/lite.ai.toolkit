option(BUILD_LITE_YOLOV5 "whether to build lite_yolov5 binary." ON)
option(BUILD_LITE_AGE_GOOGLENET "whether to build lite_age_googlenet binary." ON)

if (BUILD_LITE_YOLOV5)
    include(cv/test_lite_yolov5.cmake)
endif()

if (BUILD_LITE_AGE_GOOGLENET)
    include(cv/test_lite_age_googlenet.cmake)
endif()