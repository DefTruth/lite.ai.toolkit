function(add_lite_executable executable_name field)
    message(">>>> Adding [${executable_name}] in : ${CMAKE_CURRENT_SOURCE_DIR}")
    include(${CMAKE_SOURCE_DIR}/setup_3rdparty.cmake)

    # will be include into CMakeLists.txt at examples/lite
    add_executable(${executable_name} ${field}/test_${executable_name}.cpp)
    target_link_libraries(${executable_name} litehub)  # link liblitehub

    if (LITEHUB_COPY_BUILD)
        # "set" only valid in the current directory and subdirectory and does not broadcast
        # to parent and sibling directories
        # CMAKE_SOURCE_DIR means the root path of top CMakeLists.txt
        # CMAKE_CURRENT_SOURCE_DIR the current path of current CMakeLists.txt
        set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/build/litehub/bin)
        message("=================================================================================")
        message("output binary [app: ${executable_name}] to ${EXECUTABLE_OUTPUT_PATH}")
        message("=================================================================================")
    endif ()

endfunction ()