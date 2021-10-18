############################## Source Files of LiteHub Based on NCNN #################################
# 1. glob sources files
file(GLOB NCNN_CORE_SRCS ${CMAKE_SOURCE_DIR}/lite/ncnn/core/*.cpp)
file(GLOB NCNN_CV_SRCS ${CMAKE_SOURCE_DIR}/lite/ncnn/cv/*.cpp)
file(GLOB NCNN_NLP_SRCS ${CMAKE_SOURCE_DIR}/lite/ncnn/nlp/*.cpp)
file(GLOB NCNN_ASR_SRCS ${CMAKE_SOURCE_DIR}/lite/ncnn/asr/*.cpp)
# 2. glob headers files
file(GLOB NCNN_CORE_HEAD ${CMAKE_SOURCE_DIR}/lite/ncnn/core/*.h)
file(GLOB NCNN_CV_HEAD ${CMAKE_SOURCE_DIR}/lite/ncnn/cv/*.h)
file(GLOB NCNN_NLP_HEAD ${CMAKE_SOURCE_DIR}/lite/ncnn/nlp/*.h)
file(GLOB NCNN_ASR_HEAD ${CMAKE_SOURCE_DIR}/lite/ncnn/asr/*.h)

set(NCNN_SRCS
        ${NCNN_CV_SRCS}
        ${NCNN_NLP_SRCS}
        ${NCNN_ASR_SRCS}
        ${NCNN_CORE_SRCS})

# 3. copy
message("Installing Lite.AI.ToolKit Headers for NCNN Backend ...")
# "INSTALL" can copy all files from the list to the specified path.
# "COPY" only copies one file to a specified path
file(INSTALL ${NCNN_CORE_HEAD} DESTINATION ${CMAKE_SOURCE_DIR}/build/lite.ai.toolkit/include/lite/ncnn/core)
file(INSTALL ${NCNN_CV_HEAD} DESTINATION ${CMAKE_SOURCE_DIR}/build/lite.ai.toolkit/include/lite/ncnn/cv)
file(INSTALL ${NCNN_ASR_HEAD} DESTINATION ${CMAKE_SOURCE_DIR}/build/lite.ai.toolkit/include/lite/ncnn/asr)
file(INSTALL ${NCNN_NLP_HEAD} DESTINATION ${CMAKE_SOURCE_DIR}/build/lite.ai.toolkit/include/lite/ncnn/nlp)
