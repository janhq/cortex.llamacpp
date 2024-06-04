cmake_minimum_required(VERSION 3.5)
project(cortex.llamacpp)
SET(TARGET engine)

if(UNIX AND NOT APPLE)
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fPIC")
  add_compile_options(-fPIC)
endif()

set(THIRD_PARTY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/build_deps/_install)

if(UNIX AND NOT APPLE)
  add_compile_options(-fPIC)
endif()

include(CheckIncludeFileCXX)
# CPP version
# check_include_file_cxx(any HAS_ANY)
# check_include_file_cxx(string_view HAS_STRING_VIEW)
# check_include_file_cxx(coroutine HAS_COROUTINE)
# if(HAS_ANY
#   AND HAS_STRING_VIEW
#   AND HAS_COROUTINE)
#   set(CMAKE_CXX_STANDARD 20)
# elseif(HAS_ANY AND HAS_STRING_VIEW)
#   set(CMAKE_CXX_STANDARD 17)
# else()
set(CMAKE_CXX_STANDARD 14)
# endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

SET(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

add_subdirectory(llama.cpp/examples/llava)
add_subdirectory(llama.cpp)

add_library(${TARGET} SHARED 
    src/llama_engine.cc
    src/llama_server_context.cc
    src/llama_client_slot.cc
)

find_library(JSONCPP
    NAMES jsoncpp
    HINTS "${THIRD_PARTY_PATH}/lib"
)

find_library(TRANTOR
    NAMES trantor
    HINTS "${THIRD_PARTY_PATH}/lib"
)

target_link_libraries(${TARGET} PRIVATE common llama llava ${JSONCPP} ${TRANTOR}
                                              ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE 
            ${CMAKE_CURRENT_SOURCE_DIR}/base
            ${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp 
            ${THIRD_PARTY_PATH}/include)