# config lite.ai shared lib.
function(add_lite_ai_toolkit_shared_library version soversion)
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
        include(cmake/lite.ai.toolkit-onnxruntime.cmake)
        set(LITE_SRCS ${LITE_SRCS} ${ORT_SRCS})
        set(LITE_DEPENDENCIES ${LITE_DEPENDENCIES} onnxruntime)
    endif()

    if (ENABLE_MNN)
        include(cmake/lite.ai.toolkit-mnn.cmake)
        set(LITE_SRCS ${LITE_SRCS} ${MNN_SRCS})
        set(LITE_DEPENDENCIES ${LITE_DEPENDENCIES} MNN)
    endif()

    if (ENABLE_NCNN)
        include(cmake/lite.ai.toolkit-ncnn.cmake)
        set(LITE_SRCS ${LITE_SRCS} ${NCNN_SRCS})
        set(LITE_DEPENDENCIES ${LITE_DEPENDENCIES} ncnn)
    endif()

    if (ENABLE_TNN)
        include(cmake/lite.ai.toolkit-tnn.cmake)
        set(LITE_SRCS ${LITE_SRCS} ${TNN_SRCS})
        set(LITE_DEPENDENCIES ${LITE_DEPENDENCIES} TNN)
    endif()

    # 4. shared library
    add_library(lite.ai.toolkit SHARED ${LITE_SRCS})
    target_link_libraries(lite.ai.toolkit ${LITE_DEPENDENCIES})
    set_target_properties(lite.ai.toolkit PROPERTIES VERSION ${version} SOVERSION ${soversion})

    message("Installing Lite.AI.ToolKit Headers ...")
    file(INSTALL ${LITE_HEAD} DESTINATION ${CMAKE_SOURCE_DIR}/build/lite.ai.toolkit/include/lite)

    message(">>>> Added Shared Library: lite.ai.toolkit !")

endfunction()

# add custom command for lite.ai shared lib.
function(add_lite_ai_toolkit_engines_headers_command)
    add_custom_command(TARGET lite.ai.toolkit
            PRE_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory ${EXECUTABLE_OUTPUT_PATH}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${LIBRARY_OUTPUT_PATH}
            COMMAND ${CMAKE_COMMAND} -E echo "create ${LIBRARY_OUTPUT_PATH} done!"
            COMMAND ${CMAKE_COMMAND} -E echo "create ${EXECUTABLE_OUTPUT_PATH} done!"
            )

    # copy opencv2 headers
    if (INCLUDE_OPENCV)
        add_custom_command(TARGET lite.ai.toolkit
                PRE_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory ${BUILD_LITE_AI_DIR}/include/opencv2
                COMMAND ${CMAKE_COMMAND} -E echo "create ${BUILD_LITE_AI_DIR}/include/opencv2 done!"
                )
        add_custom_command(TARGET lite.ai.toolkit
                PRE_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_directory ${LITE_AI_ROOT_DIR}/opencv2 ${BUILD_LITE_AI_DIR}/include/opencv2
                COMMAND ${CMAKE_COMMAND} -E echo "copy opencv2 headers to ${BUILD_LITE_AI_DIR}/opencv2 done!"
                )
    endif()

    if (ENABLE_ONNXRUNTIME)
        add_custom_command(TARGET lite.ai.toolkit
                PRE_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory ${BUILD_LITE_AI_DIR}/include/onnxruntime
                COMMAND ${CMAKE_COMMAND} -E echo "create ${BUILD_LITE_AI_DIR}/include/onnxruntime done!"
                )
        # copy onnxruntime headers
        add_custom_command(TARGET lite.ai.toolkit
                PRE_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_directory ${LITE_AI_ROOT_DIR}/onnxruntime ${BUILD_LITE_AI_DIR}/include/onnxruntime
                COMMAND ${CMAKE_COMMAND} -E echo "copy onnxruntime headers to ${BUILD_LITE_AI_DIR}/include/onnxruntime done!"
                )

    endif()

    if (ENABLE_MNN)
        add_custom_command(TARGET lite.ai.toolkit
                PRE_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory ${BUILD_LITE_AI_DIR}/include/MNN
                COMMAND ${CMAKE_COMMAND} -E echo "create ${BUILD_LITE_AI_DIR}/include/MNN done!"
                )
        # copy MNN headers
        add_custom_command(TARGET lite.ai.toolkit
                PRE_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_directory ${LITE_AI_ROOT_DIR}/MNN ${BUILD_LITE_AI_DIR}/include/MNN
                COMMAND ${CMAKE_COMMAND} -E echo "copy MNN headers to ${BUILD_LITE_AI_DIR}/include/MNN done!"
                )

    endif()

    if (ENABLE_NCNN)
        add_custom_command(TARGET lite.ai.toolkit
                PRE_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory ${BUILD_LITE_AI_DIR}/include/ncnn
                COMMAND ${CMAKE_COMMAND} -E echo "create ${BUILD_LITE_AI_DIR}/include/ncnn done!"
                )
        # copy ncnn headers
        add_custom_command(TARGET lite.ai.toolkit
                PRE_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_directory ${LITE_AI_ROOT_DIR}/ncnn ${BUILD_LITE_AI_DIR}/include/ncnn
                COMMAND ${CMAKE_COMMAND} -E echo "copy NCNN headers to ${BUILD_LITE_AI_DIR}/ncnn done!"
                )
    endif()

    if (ENABLE_TNN)
        add_custom_command(TARGET lite.ai.toolkit
                PRE_BUILD
                COMMAND ${CMAKE_COMMAND} -E make_directory ${BUILD_LITE_AI_DIR}/include/tnn
                COMMAND ${CMAKE_COMMAND} -E echo "create ${BUILD_LITE_AI_DIR}/include/tnn done!"
                )
        # copy TNN headers
        add_custom_command(TARGET lite.ai.toolkit
                PRE_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_directory ${LITE_AI_ROOT_DIR}/tnn ${BUILD_LITE_AI_DIR}/include/tnn
                COMMAND ${CMAKE_COMMAND} -E echo "copy TNN headers to ${BUILD_LITE_AI_DIR}/include/tnn done!"
                )
    endif()

endfunction()

function(add_lite_ai_toolkit_engines_libs_command)
    # copy opencv libs
    if (INCLUDE_OPENCV)
        message("Installing OpenCV libs -> INCLUDE_OPENCV: ${INCLUDE_OPENCV} ...")
        file(GLOB ALL_OpenCV_LIBS ${LITE_AI_ROOT_DIR}/lib/*opencv*)
        file(INSTALL ${ALL_OpenCV_LIBS} DESTINATION ${LIBRARY_OUTPUT_PATH})
    endif()
    # copy onnxruntime libs
    if (ENABLE_ONNXRUNTIME)
        message("Installing ONNXRuntime libs -> ENABLE_ONNXRUNTIME: ${ENABLE_ONNXRUNTIME} ...")
        file(GLOB ALL_ONNXRUNTIME_LIBS ${LITE_AI_ROOT_DIR}/lib/*onnxruntime*)
        file(INSTALL ${ALL_ONNXRUNTIME_LIBS} DESTINATION ${LIBRARY_OUTPUT_PATH})
    endif()
    # copy MNN libs
    if (ENABLE_MNN)
        message("Installing MNN libs -> ENABLE_MNN: ${ENABLE_MNN} ...")
        file(GLOB ALL_MNN_LIBS ${LITE_AI_ROOT_DIR}/lib/*MNN*)
        file(INSTALL ${ALL_MNN_LIBS} DESTINATION ${LIBRARY_OUTPUT_PATH})
    endif()
    # copy NCNN libs
    if (ENABLE_NCNN)
        message("Installing NCNN libs -> ENABLE_NCNN: ${ENABLE_NCNN} ...")
        file(GLOB ALL_NCNN_LIBS ${LITE_AI_ROOT_DIR}/lib/*ncnn*)
        file(INSTALL ${ALL_NCNN_LIBS} DESTINATION ${LIBRARY_OUTPUT_PATH})
    endif()
    # copy TNN libs
    if (ENABLE_TNN)
        message("Installing TNN libs -> ENABLE_TNN: ${ENABLE_TNN} ...")
        file(GLOB ALL_TNN_LIBS ${LITE_AI_ROOT_DIR}/lib/*TNN*)
        file(INSTALL ${ALL_TNN_LIBS} DESTINATION ${LIBRARY_OUTPUT_PATH})
    endif()

    if (LITE_AI_BUILD_TEST)
        # copy opencv & lite.ai.toolkit & engines libs to bin directory
        add_custom_command(TARGET lite.ai.toolkit
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_directory ${LIBRARY_OUTPUT_PATH} ${EXECUTABLE_OUTPUT_PATH}
                COMMAND ${CMAKE_COMMAND} -E echo "copy all lite.ai.toolkit libs to ${EXECUTABLE_OUTPUT_PATH} done!"
                )
    endif()
endfunction()

function(add_lite_ai_toolkit_test_custom_command)
    if (LITE_AI_BUILD_TEST)
        # copy opencv & lite.ai.toolkit & engines libs to bin directory
        add_custom_command(TARGET lite.ai.toolkit
                POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_directory ${LIBRARY_OUTPUT_PATH} ${EXECUTABLE_OUTPUT_PATH}
                COMMAND ${CMAKE_COMMAND} -E echo "copy all lite.ai.toolkit libs to ${EXECUTABLE_OUTPUT_PATH} done!"
                )
    endif()
endfunction()

function(add_lite_executable executable_name field)
    add_executable(${executable_name} ${field}/test_${executable_name}.cpp)
    target_link_libraries(${executable_name} lite.ai.toolkit)  # link lite.ai.toolkit
    message(">>>> Added Lite Executable: ${executable_name} !")
endfunction ()
