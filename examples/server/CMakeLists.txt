cmake_minimum_required(VERSION 3.5)
project(server)

find_package(Threads REQUIRED)

if(UNIX AND NOT APPLE)
  set(LINKER_FLAGS -ldl)
endif()

include(CheckIncludeFileCXX)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(${PROJECT_NAME}
    server.cc
    dylib.h
    httplib.h
)

set(THIRD_PARTY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../build_deps/_install)
set(CORTEX_COMMON_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../base/)

find_library(JSONCPP
    NAMES jsoncpp
    HINTS "${THIRD_PARTY_PATH}/lib"
)

find_library(TRANTOR
    NAMES trantor
    HINTS "${THIRD_PARTY_PATH}/lib"
)

target_link_libraries(${PROJECT_NAME} PRIVATE ${JSONCPP} ${TRANTOR} ${LINKER_FLAGS}
                                              ${CMAKE_THREAD_LIBS_INIT})

target_include_directories(${PROJECT_NAME} PRIVATE 
                                    ${CORTEX_COMMON_PATH}
                                    ${THIRD_PARTY_PATH}/include)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)                                    