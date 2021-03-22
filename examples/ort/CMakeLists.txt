option(BUILD_ORT_CV_FSANET "whether to build ortcv_fsanet binary." ON)
option(BUILD_ORT_CV_UTRAFACE "whether to build ortcv_utraface binary." OFF)
option(BUILD_ORT_CV_PFLD "whether to build ortcv_pfld binary." OFF)

if (BUILD_ORT_CV_FSANET)
    include(cv/test_ortcv_fsanet.cmake)
endif()

if (BUILD_ORT_CV_UTRAFACE)
    include(cv/test_ortcv_utraface.cmake)
endif()

if (BUILD_ORT_CV_PFLD)
    include(cv/test_ortcv_pfld.cmake)
endif()