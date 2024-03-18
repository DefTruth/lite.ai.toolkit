#    This file will define the following variables for find_package method:
#      - lite.ai.toolkit_LIBS                 : The list of libraries to link against.
#      - lite.ai.toolkit_INCLUDE_DIRS         : The lite.ai.toolkit include directories.
#      - lite.ai.toolkit_Found                : The status of lite.ai.toolkit

include(${CMAKE_CURRENT_LIST_DIR}/lite.ai.toolkit.cmake)
# setup lite.ai.toolkit cmake variables
set(lite.ai.toolkit_LIBS ${Lite_AI_LIBS})
set(lite.ai.toolkit_INCLUDE_DIRS ${Lite_AI_INCLUDE_DIRS})
set(lite.ai.toolkit_Found TRUE)
