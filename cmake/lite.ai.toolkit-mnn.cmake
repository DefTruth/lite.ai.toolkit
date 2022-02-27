############################## Source Files of LiteHub Based on MNN #################################
# 1. glob sources files
file(GLOB MNN_CORE_SRCS ${CMAKE_SOURCE_DIR}/lite/mnn/core/*.cpp)
file(GLOB MNN_CV_SRCS ${CMAKE_SOURCE_DIR}/lite/mnn/cv/*.cpp)
file(GLOB MNN_NLP_SRCS ${CMAKE_SOURCE_DIR}/lite/mnn/nlp/*.cpp)
file(GLOB MNN_ASR_SRCS ${CMAKE_SOURCE_DIR}/lite/mnn/asr/*.cpp)
# 2. glob headers files
file(GLOB MNN_CORE_HEAD ${CMAKE_SOURCE_DIR}/lite/mnn/core/*.h)
file(GLOB MNN_CV_HEAD ${CMAKE_SOURCE_DIR}/lite/mnn/cv/*.h)
file(GLOB MNN_NLP_HEAD ${CMAKE_SOURCE_DIR}/lite/mnn/nlp/*.h)
file(GLOB MNN_ASR_HEAD ${CMAKE_SOURCE_DIR}/lite/mnn/asr/*.h)

set(MNN_SRCS
        ${MNN_CV_SRCS}
        ${MNN_NLP_SRCS}
        ${MNN_ASR_SRCS}
        ${MNN_CORE_SRCS})

# 3. copy
message("Installing Lite.AI.ToolKit Headers for MNN Backend ...")
# "INSTALL" can copy all files from the list to the specified path.
# "COPY" only copies one file to a specified path
file(INSTALL ${MNN_CORE_HEAD} DESTINATION ${BUILD_LITE_AI_DIR}/include/lite/mnn/core)
file(INSTALL ${MNN_CV_HEAD} DESTINATION ${BUILD_LITE_AI_DIR}/include/lite/mnn/cv)
file(INSTALL ${MNN_ASR_HEAD} DESTINATION ${BUILD_LITE_AI_DIR}/include/lite/mnn/asr)
file(INSTALL ${MNN_NLP_HEAD} DESTINATION ${BUILD_LITE_AI_DIR}/include/lite/mnn/nlp)
