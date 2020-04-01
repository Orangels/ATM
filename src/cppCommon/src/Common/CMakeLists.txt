cmake_minimum_required(VERSION 3.5.1)
project(Common)
set(CMAKE_INSTALL_PREFIX ../../)
#SET(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/../../build)
#execute_process(COMMAND mkdir ${CMAKE_INSTALL_PREFIX}build)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O0 -ggdb -Wall -fPIC") #-O0 -ggdb
#set(CMAKE_BUILD_TYPE "Debug")
find_package(PythonLibs)
set(CMAKE_CXX_STANDARD 11)
include_directories(${CMAKE_SOURCE_DIR}/../../include ${PYTHON_INCLUDE_DIRS})
file(GLOB sources *.cpp )
add_library(Common SHARED ${sources})
target_link_libraries(Common  ${PYTHON_LIBRARIES})

install(TARGETS Common
        LIBRARY DESTINATION build
        )

