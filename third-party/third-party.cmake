
# cmake_minimum_required(VERSION 3.12.0 FATAL_ERROR)
# project(MyProject)
include(ExternalProject)

# Define variables
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(THIRD_PARTY_INSTALL_PATH ${CMAKE_CURRENT_SOURCE_DIR}/build_deps/_install)

ExternalProject_Add(
    jsoncpp
    GIT_REPOSITORY https://github.com/open-source-parsers/jsoncpp
    GIT_TAG 1.9.5
    CMAKE_ARGS 
    	-DBUILD_SHARED_LIBS=OFF
    	-DCMAKE_INSTALL_PREFIX=${THIRD_PARTY_INSTALL_PATH}
)

ExternalProject_Add(
    trantor
    GIT_REPOSITORY https://github.com/an-tao/trantor
    GIT_TAG v1.5.17
    CMAKE_ARGS 
    	-DBUILD_SHARED_LIBS=OFF
    	-DCMAKE_INSTALL_PREFIX=${THIRD_PARTY_INSTALL_PATH}
)

# This is tricky, can find a better way?
if(WIN32)
  set(prefix "")
  set(suffix ".lib")
elseif(APPLE)
  set(prefix "lib")
  set(suffix ".a")
else()
  set(prefix "lib")
  set(suffix ".a")
endif()

add_library(jsoncpplib STATIC IMPORTED)
  set_target_properties(jsoncpplib PROPERTIES
          IMPORTED_LOCATION ${THIRD_PARTY_INSTALL_PATH}/lib/${prefix}jsoncpp${suffix}
)

add_library(trantorlib STATIC IMPORTED)
set_target_properties(trantorlib PROPERTIES
        IMPORTED_LOCATION ${THIRD_PARTY_INSTALL_PATH}/lib/${prefix}trantor${suffix}
)