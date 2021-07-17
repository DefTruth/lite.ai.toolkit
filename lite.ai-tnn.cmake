############################## Source Files of LiteHub Based on TNN #################################
# 1. glob sources files
file(GLOB TNN_CORE_SRCS ${CMAKE_SOURCE_DIR}/tnn/core/*.cpp)
file(GLOB TNN_CV_SRCS ${CMAKE_SOURCE_DIR}/tnn/cv/*.cpp)
file(GLOB TNN_NLP_SRCS ${CMAKE_SOURCE_DIR}/tnn/nlp/*.cpp)
file(GLOB TNN_ASR_SRCS ${CMAKE_SOURCE_DIR}/tnn/asr/*.cpp)
# 2. glob headers files
file(GLOB TNN_CORE_HEAD ${CMAKE_SOURCE_DIR}/tnn/core/*.h)
file(GLOB TNN_CV_HEAD ${CMAKE_SOURCE_DIR}/tnn/cv/*.h)
file(GLOB TNN_NLP_HEAD ${CMAKE_SOURCE_DIR}/tnn/nlp/*.h)
file(GLOB TNN_ASR_HEAD ${CMAKE_SOURCE_DIR}/tnn/asr/*.h)

set(TNN_SRCS
        ${TNN_CV_SRCS}
        ${TNN_NLP_SRCS}
        ${TNN_ASR_SRCS}
        ${TNN_CORE_SRCS})

# 3. copy
if (LITE_AI_COPY_BUILD)
    message("Installing Lite.AI Headers for TNN Backend ...")
    # "INSTALL" can copy all files from the list to the specified path.
    # "COPY" only copies one file to a specified path
    file(INSTALL ${TNN_CORE_HEAD} DESTINATION ${CMAKE_SOURCE_DIR}/build/lite.ai/include/tnn/core)
    file(INSTALL ${TNN_CV_HEAD} DESTINATION ${CMAKE_SOURCE_DIR}/build/lite.ai/include/tnn/cv)
    file(INSTALL ${TNN_ASR_HEAD} DESTINATION ${CMAKE_SOURCE_DIR}/build/lite.ai/include/tnn/asr)
    file(INSTALL ${TNN_NLP_HEAD} DESTINATION ${CMAKE_SOURCE_DIR}/build/lite.ai/include/tnn/nlp)
endif ()
