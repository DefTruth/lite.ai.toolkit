set(TNN_DIR ${THIRD_PARTY_PATH}/TNN)
if(NOT EXISTS ${TNN_DIR})
    message(FATAL_ERROR "${TNN_DIR} is not exists!")
endif()
include_directories(${TNN_DIR}/include)
link_directories(${TNN_DIR}/lib)

# 1. glob sources files
file(GLOB TNN_CORE_SRCS ${CMAKE_SOURCE_DIR}/lite/tnn/core/*.cpp)
file(GLOB TNN_CV_SRCS ${CMAKE_SOURCE_DIR}/lite/tnn/cv/*.cpp)
file(GLOB TNN_NLP_SRCS ${CMAKE_SOURCE_DIR}/lite/tnn/nlp/*.cpp)
file(GLOB TNN_ASR_SRCS ${CMAKE_SOURCE_DIR}/lite/tnn/asr/*.cpp)
# 2. glob headers files
file(GLOB TNN_CORE_HEAD ${CMAKE_SOURCE_DIR}/lite/tnn/core/*.h)
file(GLOB TNN_CV_HEAD ${CMAKE_SOURCE_DIR}/lite/tnn/cv/*.h)
file(GLOB TNN_NLP_HEAD ${CMAKE_SOURCE_DIR}/lite/tnn/nlp/*.h)
file(GLOB TNN_ASR_HEAD ${CMAKE_SOURCE_DIR}/lite/tnn/asr/*.h)

set(TNN_SRCS
        ${TNN_CV_SRCS}
        ${TNN_NLP_SRCS}
        ${TNN_ASR_SRCS}
        ${TNN_CORE_SRCS})

# 3. copy
message("Installing Lite.AI.ToolKit Headers for TNN Backend ...")
# "INSTALL" can copy all files from the list to the specified path.
# "COPY" only copies one file to a specified path
file(INSTALL ${TNN_CORE_HEAD} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/lite/tnn/core)
file(INSTALL ${TNN_CV_HEAD} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/lite/tnn/cv)
file(INSTALL ${TNN_ASR_HEAD} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/lite/tnn/asr)
file(INSTALL ${TNN_NLP_HEAD} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/lite/tnn/nlp)
