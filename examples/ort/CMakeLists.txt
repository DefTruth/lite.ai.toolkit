option(BUILD_ORT_CV_FSANET "whether to build ortcv_fsanet binary." ON)
option(BUILD_ORT_CV_ULTRAFACE "whether to build ortcv_utraface binary." ON)
option(BUILD_ORT_CV_PFLD "whether to build ortcv_pfld binary." ON)
option(BUILD_ORT_CV_AGE_GOOGLENET "whether to build ortcv_age_googlenet binary." ON)
option(BUILD_ORT_CV_GENDER_GOOGLENET "whether to build ortcv_gender_googlenet binary." ON)
option(BUILD_ORT_CV_EMOTION_FERPLUS "whether to build ortcv_emotion_ferplus binary." ON)
option(BUILD_ORT_CV_VGG16_AGE "whether to build ortcv_vgg16_age binary." ON)
option(BUILD_ORT_CV_VGG16_GENDER "whether to build ortcv_vgg16_gender binary." ON)

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