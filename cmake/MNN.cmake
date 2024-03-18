set(MNN_Version "2.8.2" CACHE STRING "MNN version" FORCE)
set(MNN_DIR ${THIRD_PARTY_PATH}/MNN)
# download from github if MNN library is not exists
if (NOT EXISTS ${MNN_DIR})
    set(MNN_Filename "MNN-${MNN_Version}-linux-cpu-x86_64.tgz")
    set(MNN_URL https://github.com/DefTruth/lite.ai.toolkit/releases/download/v0.2.0-rc0/${MNN_Filename})
    message("[Lite.AI.Toolkit][I] Downloading MNN library: ${MNN_URL}")
    download_and_decompress(${MNN_URL} ${MNN_Filename} ${MNN_DIR}) 
else() 
    message("[Lite.AI.Toolkit][I] Found local MNN library: ${MNN_DIR}")
endif() 
if(NOT EXISTS ${MNN_DIR})
    message(FATAL_ERROR "[Lite.AI.Toolkit][E] ${MNN_DIR} is not exists!")
endif()

include_directories(${MNN_DIR}/include)
link_directories(${MNN_DIR}/lib)

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

set(MNN_SRCS ${MNN_CV_SRCS} ${MNN_NLP_SRCS} ${MNN_ASR_SRCS} ${MNN_CORE_SRCS})
# 3. copy
message("[Lite.AI.Toolkit][I] Installing Lite.AI.ToolKit Headers for MNN Backend ...")
# "INSTALL" can copy all files from the list to the specified path.
# "COPY" only copies one file to a specified path
file(INSTALL ${MNN_CORE_HEAD} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/lite/mnn/core)
file(INSTALL ${MNN_CV_HEAD} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/lite/mnn/cv)
file(INSTALL ${MNN_ASR_HEAD} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/lite/mnn/asr)
file(INSTALL ${MNN_NLP_HEAD} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/lite/mnn/nlp)
