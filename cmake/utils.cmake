function(download_and_decompress url filename decompress_dir)
  if(NOT EXISTS ${filename})
    message("Downloading file from ${url} to ${filename} ...")
    file(DOWNLOAD ${url} "${filename}.tmp" SHOW_PROGRESS)
    file(RENAME "${filename}.tmp" ${filename})
  endif()
  if(NOT EXISTS ${decompress_dir})
    file(MAKE_DIRECTORY ${decompress_dir})
  endif()
  message("Decompress file ${filename} ...")
  execute_process(COMMAND ${CMAKE_COMMAND} -E tar -xf ${filename} WORKING_DIRECTORY ${decompress_dir})
endfunction()

# config lite.ai shared lib.
function(add_lite_ai_toolkit_shared_library version soversion)
    configure_file(
            "${CMAKE_SOURCE_DIR}/lite/config.h.in"
            "${CMAKE_SOURCE_DIR}/lite/config.h"
    )
    file(GLOB LITE_SRCS ${CMAKE_SOURCE_DIR}/lite/*.cpp)
    set(LITE_DEPENDENCIES ${OpenCV_LIBS})

    if (ENABLE_ONNXRUNTIME)
        include(cmake/onnxruntime.cmake)
        set(LITE_SRCS ${LITE_SRCS} ${ORT_SRCS})
        set(LITE_DEPENDENCIES ${LITE_DEPENDENCIES} onnxruntime)
    endif ()

    if (ENABLE_MNN)
        include(cmake/MNN.cmake)
        set(LITE_SRCS ${LITE_SRCS} ${MNN_SRCS})
        set(LITE_DEPENDENCIES ${LITE_DEPENDENCIES} MNN)
    endif ()

    if (ENABLE_NCNN)
        include(cmake/ncnn.cmake)
        set(LITE_SRCS ${LITE_SRCS} ${NCNN_SRCS})
        set(LITE_DEPENDENCIES ${LITE_DEPENDENCIES} ncnn)
    endif ()

    if (ENABLE_TNN)
        include(cmake/TNN.cmake)
        set(LITE_SRCS ${LITE_SRCS} ${TNN_SRCS})
        set(LITE_DEPENDENCIES ${LITE_DEPENDENCIES} TNN)
    endif ()

    # 4. shared library
    add_library(lite.ai.toolkit SHARED ${LITE_SRCS})
    target_link_libraries(lite.ai.toolkit ${LITE_DEPENDENCIES})
    set_target_properties(lite.ai.toolkit PROPERTIES VERSION ${version} SOVERSION ${soversion})

    message("[Lite.AI.Toolkit][I] Added Shared Library: lite.ai.toolkit !")

endfunction()

function(add_lite_executable executable_name field)
    add_executable(${executable_name} ${field}/test_${executable_name}.cpp)
    target_link_libraries(${executable_name} lite.ai.toolkit)  # link lite.ai.toolkit
    message("[Lite.AI.Toolkit][I] Added Lite Executable: ${executable_name} !")
endfunction()

