function(add_ort_executable executable_name field)
    if (NOT ${THIRDPARTY_SET_STATE})
        include(${CMAKE_SOURCE_DIR}/setup_3rdparty.cmake)
    endif()
    # will be include into CMakeLists.txt at examples/ort
    add_executable(${executable_name} ${field}/test_${executable_name}.cpp)
    target_link_libraries(${executable_name} litehub)  # link liblitehub
    set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/build/litehub/bin)
    message(">>>> Added executable: ${executable_name} !")
endfunction ()
