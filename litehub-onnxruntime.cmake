############################## Source Files of LiteHub Based on ONNXRuntime #################################
# 1. glob sources files
file(GLOB ONNXRUNTIME_CORE_SRCS ${CMAKE_SOURCE_DIR}/ort/core/*.cpp)
file(GLOB ONNXRUNTIME_CV_SRCS ${CMAKE_SOURCE_DIR}/ort/cv/*.cpp)
file(GLOB ONNXRUNTIME_NLP_SRCS ${CMAKE_SOURCE_DIR}/ort/nlp/*.cpp)
file(GLOB ONNXRUNTIME_ASR_SRCS ${CMAKE_SOURCE_DIR}/ort/asr/*.cpp)
# 2. glob headers files
file(GLOB ONNXRUNTIME_CORE_HEAD ${CMAKE_SOURCE_DIR}/ort/core/*.h)
file(GLOB ONNXRUNTIME_CV_HEAD ${CMAKE_SOURCE_DIR}/ort/cv/*.h)
file(GLOB ONNXRUNTIME_NLP_HEAD ${CMAKE_SOURCE_DIR}/ort/nlp/*.h)
file(GLOB ONNXRUNTIME_ASR_HEAD ${CMAKE_SOURCE_DIR}/ort/asr/*.h)

set(ORT_SRCS
        ${ONNXRUNTIME_CV_SRCS}
        ${ONNXRUNTIME_NLP_SRCS}
        ${ONNXRUNTIME_ASR_SRCS}
        ${ONNXRUNTIME_CORE_SRCS})

# 3. copy
if (LITEHUB_COPY_BUILD)
    message("Installing LiteHub Headers for ONNXRuntime Backend ...")
    # "INSTALL" can copy all files from the list to the specified path.
    # "COPY" only copies one file to a specified path
    file(INSTALL ${ONNXRUNTIME_CORE_HEAD} DESTINATION ${CMAKE_SOURCE_DIR}/build/litehub/include/ort/core)
    file(INSTALL ${ONNXRUNTIME_CV_HEAD} DESTINATION ${CMAKE_SOURCE_DIR}/build/litehub/include/ort/cv)
    file(INSTALL ${ONNXRUNTIME_ASR_HEAD} DESTINATION ${CMAKE_SOURCE_DIR}/build/litehub/include/ort/asr)
    file(INSTALL ${ONNXRUNTIME_NLP_HEAD} DESTINATION ${CMAKE_SOURCE_DIR}/build/litehub/include/ort/nlp)
endif ()
