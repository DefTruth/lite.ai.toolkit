############################## Source Files of LiteHub Based on ONNXRuntime #################################
# 1. glob sources files
file(GLOB ONNXRUNTIME_CORE_SRCS ${CMAKE_SOURCE_DIR}/lite/ort/core/*.cpp)
file(GLOB ONNXRUNTIME_CV_SRCS ${CMAKE_SOURCE_DIR}/lite/ort/cv/*.cpp)
file(GLOB ONNXRUNTIME_NLP_SRCS ${CMAKE_SOURCE_DIR}/lite/ort/nlp/*.cpp)
file(GLOB ONNXRUNTIME_ASR_SRCS ${CMAKE_SOURCE_DIR}/lite/ort/asr/*.cpp)
# 2. glob headers files
file(GLOB ONNXRUNTIME_CORE_HEAD ${CMAKE_SOURCE_DIR}/lite/ort/core/*.h)
file(GLOB ONNXRUNTIME_CV_HEAD ${CMAKE_SOURCE_DIR}/lite/ort/cv/*.h)
file(GLOB ONNXRUNTIME_NLP_HEAD ${CMAKE_SOURCE_DIR}/lite/ort/nlp/*.h)
file(GLOB ONNXRUNTIME_ASR_HEAD ${CMAKE_SOURCE_DIR}/lite/ort/asr/*.h)

set(ORT_SRCS
        ${ONNXRUNTIME_CV_SRCS}
        ${ONNXRUNTIME_NLP_SRCS}
        ${ONNXRUNTIME_ASR_SRCS}
        ${ONNXRUNTIME_CORE_SRCS})

# 3. copy
message("Installing Lite.AI.ToolKit Headers for ONNXRuntime Backend ...")
# "INSTALL" can copy all files from the list to the specified path.
# "COPY" only copies one file to a specified path
file(INSTALL ${ONNXRUNTIME_CORE_HEAD} DESTINATION ${CMAKE_SOURCE_DIR}/build/lite.ai.toolkit/include/lite/ort/core)
file(INSTALL ${ONNXRUNTIME_CV_HEAD} DESTINATION ${CMAKE_SOURCE_DIR}/build/lite.ai.toolkit/include/lite/ort/cv)
file(INSTALL ${ONNXRUNTIME_ASR_HEAD} DESTINATION ${CMAKE_SOURCE_DIR}/build/lite.ai.toolkit/include/lite/ort/asr)
file(INSTALL ${ONNXRUNTIME_NLP_HEAD} DESTINATION ${CMAKE_SOURCE_DIR}/build/lite.ai.toolkit/include/lite/ort/nlp)
