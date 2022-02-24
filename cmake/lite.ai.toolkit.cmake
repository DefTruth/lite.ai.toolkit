add_lite_ai_toolkit_shared_library(${VERSION_STRING} ${SOVERSION_STRING})
add_lite_ai_toolkit_engines_headers_command()

# TODO: Windows需要之后兼容
if(${PLATFORM_NAME} MATCHES macos OR ${PLATFORM_NAME} MATCHES linux)
    add_lite_ai_toolkit_engines_libs_command()
endif()

if(${PLATFORM_NAME} MATCHES macos OR ${PLATFORM_NAME} MATCHES linux)
    add_lite_ai_toolkit_test_custom_command()
endif()

















