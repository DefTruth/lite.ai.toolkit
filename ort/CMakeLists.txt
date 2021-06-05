# 1. setup 3rd-party dependences
message(">>>> Current project is [liteort] in : ${CMAKE_CURRENT_SOURCE_DIR}")
include(${CMAKE_SOURCE_DIR}/setup_3rdparty.cmake)

# 2. setup onnxruntime include
include_directories(${ONNXRUNTIMR_INCLUDE_DIR})
link_directories(${ONNXRUNTIMR_LIBRARY_DIR})

file(GLOB ONNXRUNTIME_CORE_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp)
file(GLOB ONNXRUNTIME_CV_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/cv/*.cpp)
file(GLOB ONNXRUNTIME_NLP_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/nlp/*.cpp)
file(GLOB ONNXRUNTIME_ASR_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/asr/*.cpp)
file(GLOB LITE_ONNXRUNTIME_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB ONNXRUNTIME_CORE_HEAD ${CMAKE_CURRENT_SOURCE_DIR}/core/*.h)
file(GLOB ONNXRUNTIME_CV_HEAD ${CMAKE_CURRENT_SOURCE_DIR}/cv/*.h)
file(GLOB ONNXRUNTIME_NLP_HEAD ${CMAKE_CURRENT_SOURCE_DIR}/nlp/*.h)
file(GLOB ONNXRUNTIME_ASR_HEAD ${CMAKE_CURRENT_SOURCE_DIR}/asr/*.h)
file(GLOB LITE_ONNXRUNTIME_HEAD ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

set(ORT_SRCS
        ${ONNXRUNTIME_CV_SRCS} ${ONNXRUNTIME_NLP_SRCS}
        ${ONNXRUNTIME_ASR_SRCS} ${LITE_ONNXRUNTIME_SRCS}
        ${ONNXRUNTIME_CORE_SRCS})
set(ORT_HEAD
        ${ONNXRUNTIME_CV_HEAD} ${ONNXRUNTIME_NLP_HEAD}
        ${ONNXRUNTIME_ASR_HEAD} ${LITE_ONNXRUNTIME_HEAD}
        ${ONNXRUNTIME_CORE_HEAD})

add_library(liteort SHARED ${ORT_SRCS})
# TODO: remove OpenCV dependences.
target_link_libraries(liteort onnxruntime ${OpenCV_LIBS})

# 4. copy
if (LITEHUB_COPY_BUILD)
    message("install ${ORT_HEAD} ...")
    # "INSTALL" can copy all files from the list to the specified path.
    # "COPY" only copies one file to a specified path
    file(INSTALL ${ORT_HEAD} DESTINATION ${CMAKE_SOURCE_DIR}/build/liteort/include)
    set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/build/liteort/lib)
    set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/build/liteort/bin)
    message("=================================================================================")
    message("CMAKE_SOURCE_DIR in onnxruntime is ${CMAKE_SOURCE_DIR}")
    message("LIBRARY_OUTPUT_PATH in onnxruntime is ${LIBRARY_OUTPUT_PATH}")
    message("EXECUTABLE_OUTPUT_PATH in onnxruntime is ${EXECUTABLE_OUTPUT_PATH}")
    message("=================================================================================")
endif ()

# 5. install liteort
if (LITEHUB_INSTALL_SDK)
    message("CMAKE_INSTALL_PREFIX is ${CMAKE_INSTALL_PREFIX}")
    install(TARGETS liteort EXPORT liteort
            ARCHIVE DESTINATION lib
            LIBRARY DESTINATION lib
            RUNTIME DESTINATION bin)
    install(FILES ${ORT_HEAD} DESTINATION include/liteort)
endif ()