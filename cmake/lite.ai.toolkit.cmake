add_lite_ai_toolkit_shared_library(${VERSION_STRING} ${SOVERSION_STRING})
add_lite_ai_toolkit_engines_headers_command()
add_lite_ai_toolkit_engines_libs_command()
if(${PLATFORM_NAME} MATCHES macos OR ${PLATFORM_NAME} MATCHES linux)
    add_lite_ai_toolkit_test_custom_command()
endif()

















