option(BUILD_ORT_CV_FSANET "whether to build ortcv_fsanet binary." ON)
option(BUILD_ORT_CV_ULTRAFACE "whether to build ortcv_utraface binary." ON)
option(BUILD_ORT_CV_PFLD "whether to build ortcv_pfld binary." OFF)

if (BUILD_ORT_CV_FSANET)
    include(cv/test_ortcv_fsanet.cmake)
endif()

if (BUILD_ORT_CV_ULTRAFACE)
    include(cv/test_ortcv_ultraface.cmake)
endif()

if (BUILD_ORT_CV_PFLD)
    include(cv/test_ortcv_pfld.cmake)
endif()