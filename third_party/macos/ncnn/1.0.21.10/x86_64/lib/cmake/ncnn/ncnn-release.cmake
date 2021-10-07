#----------------------------------------------------------------
# Generated CMake target import file for configuration "release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "ncnn" for configuration "release"
set_property(TARGET ncnn APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ncnn PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libncnn.1.0.21.10.07.dylib"
  IMPORTED_SONAME_RELEASE "libncnn.1.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS ncnn )
list(APPEND _IMPORT_CHECK_FILES_FOR_ncnn "${_IMPORT_PREFIX}/lib/libncnn.1.0.21.10.07.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
