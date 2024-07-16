set(TensorRT_Version "10.1.0.27" CACHE STRING "TensorRT version" FORCE)
set(TensorRT_DIR ${THIRD_PARTY_PATH}/TensorRT-10.1.0.27)
#set(TensorRT_DIR /usr/local/TensorRT-8.6.1.6)
set(CUDA_DIR  ${THIRD_PARTY_PATH}/cuda)

message("_______________________")
message(${CUDA_DIR})


# download tensorrt need user operation if trt doesn't exist


if(NOT EXISTS ${TensorRT_DIR})
    message(FATAL_ERROR "[Lite.AI.Toolkit][E] ${TensorRT_DIR} is not exists!")
endif()

include_directories(${CUDA_DIR}/include)
link_directories(${CUDA_DIR}/lib64)

include_directories(${TensorRT_DIR}/include)
link_directories(${TensorRT_DIR}/lib)

# 1. glob sources files
file(GLOB TENSORRT_CORE_SRCS ${CMAKE_SOURCE_DIR}/lite/trt/core/*.cpp)
file(GLOB TENSORRT_CV_SRCS ${CMAKE_SOURCE_DIR}/lite/trt/cv/*.cpp)
file(GLOB TENSORRT_NLP_SRCS ${CMAKE_SOURCE_DIR}/lite/trt/nlp/*.cpp)
file(GLOB TENSORRT_ASR_SRCS ${CMAKE_SOURCE_DIR}/lite/trt/asr/*.cpp)
# 2. glob headers files
file(GLOB TENSORRT_CORE_HEAD ${CMAKE_SOURCE_DIR}/lite/trt/core/*.h)
file(GLOB TENSORRT_CV_HEAD ${CMAKE_SOURCE_DIR}/lite/trt/cv/*.h)
file(GLOB TENSORRT_NLP_HEAD ${CMAKE_SOURCE_DIR}/lite/trt/nlp/*.h)
file(GLOB TENSORRT_ASR_HEAD ${CMAKE_SOURCE_DIR}/lite/trt/asr/*.h)

set(TRT_SRCS ${TENSORRT_CV_SRCS} ${TENSORRT_NLP_SRCS} ${TENSORRT_ASR_SRCS} ${TENSORRT_CORE_SRCS})
# 3. copy
message("[Lite.AI.Toolkit][I] Installing Lite.AI.ToolKit Headers for TensorRT Backend ...")
# "INSTALL" can copy all files from the list to the specified path.
# "COPY" only copies one file to a specified path
file(INSTALL ${TENSORRT_CORE_HEAD} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/lite/trt/core)
file(INSTALL ${TENSORRT_CV_HEAD} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/lite/trt/cv)
file(INSTALL ${TENSORRT_ASR_HEAD} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/lite/trt/asr)
file(INSTALL ${TENSORRT_NLP_HEAD} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/lite/trt/nlp)
