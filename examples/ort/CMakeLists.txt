option(BUILD_ORT_CV_FSANET "whether to build ortcv_fsanet binary." ON)
option(BUILD_ORT_CV_ULTRAFACE "whether to build ortcv_utraface binary." ON)
option(BUILD_ORT_CV_PFLD "whether to build ortcv_pfld binary." ON)
option(BUILD_ORT_CV_AGE_GOOGLENET "whether to build ortcv_age_googlenet binary." ON)
option(BUILD_ORT_CV_GENDER_GOOGLENET "whether to build ortcv_gender_googlenet binary." ON)

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