##################################### Source Files of LiteHub #####################################
# 1. setup 3rd-party dependencies
message(">>>> Current Project is [litehub] Library in : ${CMAKE_CURRENT_SOURCE_DIR}")
include(${CMAKE_SOURCE_DIR}/setup_3rdparty.cmake)

configure_file (
        "${CMAKE_SOURCE_DIR}/lite/config.h.in"
        "${CMAKE_SOURCE_DIR}/lite/config.h"
)

# 2. glob headers files
file(GLOB LITE_HEAD ${CMAKE_SOURCE_DIR}/lite/*.h)

# 3. glob sources files
file(GLOB LITE_SRCS ${CMAKE_SOURCE_DIR}/lite/*.cpp)
set(LITE_DEPENDENCIES ${OpenCV_LIBS})

if (ENABLE_ONNXRUNTIME)
    include(litehub-onnxruntime.cmake)
    set(LITE_SRCS ${LITE_SRCS} ${ORT_SRCS})
    set(LITE_DEPENDENCIES ${LITE_DEPENDENCIES} onnxruntime)
endif()

if (ENABLE_MNN)
    include(litehub-mnn.cmake)
    set(LITE_SRCS ${LITE_SRCS} ${MNN_SRCS})
    set(LITE_DEPENDENCIES ${LITE_DEPENDENCIES} mnn)
endif()

if (ENABLE_NCNN)
    include(litehub-ncnn.cmake)
    set(LITE_SRCS ${LITE_SRCS} ${NCNN_SRCS})
    set(LITE_DEPENDENCIES ${LITE_DEPENDENCIES} ncnn)
endif()

if (ENABLE_TNN)
    include(litehub-tnn.cmake)
    set(LITE_SRCS ${LITE_SRCS} ${TNN_SRCS})
    set(LITE_DEPENDENCIES ${LITE_DEPENDENCIES} tnn)
endif()

# 4. shared library
add_library(litehub SHARED ${LITE_SRCS})
target_link_libraries(litehub ${LITE_DEPENDENCIES})

# 5. copy
if (LITEHUB_COPY_BUILD)
    message("Installing LiteHub Headers ...")
    # "INSTALL" can copy all files from the list to the specified path.
    # "COPY" only copies one file to a specified path
    file(INSTALL ${LITE_HEAD} DESTINATION ${CMAKE_SOURCE_DIR}/build/litehub/include/lite)
    set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/build/litehub/lib)
    set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/build/litehub/bin)
endif ()



















