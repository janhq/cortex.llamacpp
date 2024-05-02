
cmake_minimum_required(VERSION 3.12.0 FATAL_ERROR)
project(MyProject)
include(ExternalProject)

if(UNIX AND NOT APPLE)
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC")
  add_compile_options(-fPIC)
endif()

# Define variables
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(THIRD_PARTY_INSTALL_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../build_deps/_install)

ExternalProject_Add(
    c-ares
    GIT_REPOSITORY https://github.com/c-ares/c-ares
    GIT_TAG cares-1_26_0
    CMAKE_ARGS
    	-DCARES_SHARED=OFF
	    -DCARES_STATIC=ON
        -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
    	-DCMAKE_INSTALL_PREFIX=${THIRD_PARTY_INSTALL_PATH}
)

ExternalProject_Add(
    trantor
    GIT_REPOSITORY https://github.com/an-tao/trantor
    GIT_TAG v1.5.17
    CMAKE_ARGS 
    	-DBUILD_SHARED_LIBS=OFF
      -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
      -DCMAKE_PREFIX_PATH=${THIRD_PARTY_INSTALL_PATH}
    	-DCMAKE_INSTALL_PREFIX=${THIRD_PARTY_INSTALL_PATH}
)

ExternalProject_Add(
    jsoncpp
    GIT_REPOSITORY https://github.com/open-source-parsers/jsoncpp
    GIT_TAG 1.9.5
    CMAKE_ARGS 
    	-DBUILD_SHARED_LIBS=OFF
      -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
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

add_dependencies(trantor c-ares)