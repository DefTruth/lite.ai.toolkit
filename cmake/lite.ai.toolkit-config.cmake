#    This file will define the following variables for find_package method:
#      - Lite_AI_LIBS                 : The list of libraries to link against.
#      - Lite_AI_INCLUDE_DIRS         : The FastDeploy include directories.
#      - Lite_AI_Found                : The status of FastDeploy

include(${CMAKE_CURRENT_LIST_DIR}/lite.ai.toolkit.cmake)
# setup FastDeploy cmake variables
set(Lite_AI_LIBS ${Lite_AI_LIBS})
set(Lite_AI_INCLUDE_DIRS ${Lite_AI_INCLUDE_DIRS})
set(Lite_AI_Found TRUE)