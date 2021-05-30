option(BUILD_ORT_CV_FSANET "whether to build ortcv_fsanet binary." ON)
option(BUILD_ORT_CV_ULTRAFACE "whether to build ortcv_utraface binary." ON)
option(BUILD_ORT_CV_PFLD "whether to build ortcv_pfld binary." ON)
option(BUILD_ORT_CV_AGE_GOOGLENET "whether to build ortcv_age_googlenet binary." ON)
option(BUILD_ORT_CV_GENDER_GOOGLENET "whether to build ortcv_gender_googlenet binary." ON)
option(BUILD_ORT_CV_EMOTION_FERPLUS "whether to build ortcv_emotion_ferplus binary." ON)
option(BUILD_ORT_CV_VGG16_AGE "whether to build ortcv_vgg16_age binary." ON)
option(BUILD_ORT_CV_VGG16_GENDER "whether to build ortcv_vgg16_gender binary." ON)
option(BUILD_ORT_CV_ARCFACE_RESNET "whether to build ortcv_arcface_resnet binary." ON)
option(BUILD_ORT_CV_FAST_STYLE_TRANSFER "whether to build ortcv_fast_style_transfer binary." ON)
option(BUILD_ORT_CV_SUBPIXEL_CNN "whether to build ortcv_subpixel_cnn binary." ON)
option(BUILD_ORT_CV_SSRNET "whether to build ortcv_ssrnet binary." ON)
option(BUILD_ORT_CV_COLORIZER "whether to build ortcv_colorizer binary." ON)
option(BUILD_ORT_CV_YOLOV5 "whether to build ortcv_yolov5 binary." ON)
option(BUILD_ORT_CV_YOLOV4 "whether to build ortcv_yolov4 binary." ON)
option(BUILD_ORT_CV_YOLOV3 "whether to build ortcv_yolov3 binary." ON)
option(BUILD_ORT_CV_TINY_YOLOV3 "whether to build ortcv_tiny_yolov3 binary." ON)
option(BUILD_ORT_CV_SHUFFLENETV2 "whether to build ortcv_shufflenetv2 binary." ON)

if (BUILD_ORT_CV_FSANET)
    include(cv/test_ortcv_fsanet.cmake)
endif()

if (BUILD_ORT_CV_ULTRAFACE)
    include(cv/test_ortcv_ultraface.cmake)
endif()

if (BUILD_ORT_CV_PFLD)
    include(cv/test_ortcv_pfld.cmake)
endif()

if (BUILD_ORT_CV_AGE_GOOGLENET)
    include(cv/test_ortcv_age_googlenet.cmake)
endif()

if (BUILD_ORT_CV_GENDER_GOOGLENET)
    include(cv/test_ortcv_gender_googlenet.cmake)
endif()

if (BUILD_ORT_CV_EMOTION_FERPLUS)
    include(cv/test_ortcv_emotion_ferplus.cmake)
endif()

if (BUILD_ORT_CV_VGG16_AGE)
    include(cv/test_ortcv_vgg16_age.cmake)
endif()

if (BUILD_ORT_CV_VGG16_GENDER)
    include(cv/test_ortcv_vgg16_gender.cmake)
endif()

if (BUILD_ORT_CV_ARCFACE_RESNET)
    include(cv/test_ortcv_arcface_resnet.cmake)
endif()

if (BUILD_ORT_CV_FAST_STYLE_TRANSFER)
    include(cv/test_ortcv_fast_style_transfer.cmake)
endif()

if (BUILD_ORT_CV_SUBPIXEL_CNN)
    include(cv/test_ortcv_subpixel_cnn.cmake)
endif()

if (BUILD_ORT_CV_SSRNET)
    include(cv/test_ortcv_ssrnet.cmake)
endif()

if (BUILD_ORT_CV_COLORIZER)
    include(cv/test_ortcv_colorizer.cmake)
endif()

if (BUILD_ORT_CV_YOLOV5)
    include(cv/test_ortcv_yolov5.cmake)
endif()

if (BUILD_ORT_CV_YOLOV4)
    include(cv/test_ortcv_yolov4.cmake)
endif()

if (BUILD_ORT_CV_YOLOV3)
    include(cv/test_ortcv_yolov3.cmake)
endif()

if (BUILD_ORT_CV_TINY_YOLOV3)
    include(cv/test_ortcv_tiny_yolov3.cmake)
endif()

if (BUILD_ORT_CV_SHUFFLENETV2)
    include(cv/test_ortcv_shufflenetv2.cmake)
endif()