set(OnnxRuntime_Version "1.17.1" CACHE STRING "OnnxRuntime version" FORCE)
set(OnnxRuntime_DIR ${THIRD_PARTY_PATH}/onnxruntime)
# download from github if MNN library is not exists
if (NOT EXISTS ${MNN_DIR})
    set(MNN_Filename "MNN-${MNN_Version}-linux-cpu-x86_64.tgz")
    set(MNN_URL https://github.com/DefTruth/lite.ai.toolkit/releases/download/v0.2.0-rc0/${MNN_Filename})
    message("[Lite.AI.Toolkit][I] Downloading MNN library: ${MNN_URL}")
    download_and_decompress(${MNN_URL} ${MNN_Filename} ${MNN_DIR}) 
else() 
    message("[Lite.AI.Toolkit][I] Found local MNN library: ${MNN_DIR}")
endif() 

if(NOT EXISTS ${OnnxRuntime_DIR})
    message(FATAL_ERROR "[Lite.AI.Toolkit][E] ${OnnxRuntime_DIR} is not exists!")
endif()
include_directories(${OnnxRuntime_DIR}/include)
link_directories(${OnnxRuntime_DIR}/lib)

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

set(ORT_SRCS ${ONNXRUNTIME_CV_SRCS} ${ONNXRUNTIME_NLP_SRCS} ${ONNXRUNTIME_ASR_SRCS} ${ONNXRUNTIME_CORE_SRCS})
# 3. copy
message("[Lite.AI.Toolkit][I] Installing Lite.AI.ToolKit Headers for ONNXRuntime Backend ...")
# "INSTALL" can copy all files from the list to the specified path.
# "COPY" only copies one file to a specified path
file(INSTALL ${ONNXRUNTIME_CORE_HEAD} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/lite/ort/core)
file(INSTALL ${ONNXRUNTIME_CV_HEAD} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/lite/ort/cv)
file(INSTALL ${ONNXRUNTIME_ASR_HEAD} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/lite/ort/asr)
file(INSTALL ${ONNXRUNTIME_NLP_HEAD} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/lite/ort/nlp)
