cmake_minimum_required(VERSION 3.12)
project(CMAES VERSION 0.0.1 LANGUAGES CXX)
  
SET(CMAKE_CXX_STANDARD 17) 
SET(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CONAN_DISABLE_CHECK_COMPILER ON)
execute_process(COMMAND conan install ${CMAKE_SOURCE_DIR})
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake) 

conan_basic_setup() 

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2 -march=native")

elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
endif()
   
 
file(GLOB SOURCES src/*.cpp) 
 
add_library(cmaes STATIC ${SOURCES}) 
target_link_libraries(cmaes ${CONAN_LIBS})
  
target_include_directories(cmaes
PUBLIC
    # Interface paths are relative to the installation path, not the current directory.
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include> 
) 
 
target_compile_features(cmaes
PUBLIC
    cxx_std_17
)
 
enable_testing()

add_subdirectory(external/IOHexperimenter)

add_executable(integration tests/integration.cpp)

target_link_libraries(integration PRIVATE cmaes ioh)

add_executable(x "tests/x.cpp")
target_link_libraries(x PRIVATE cmaes ioh)

target_precompile_headers(integration 
  PRIVATE 
  external/IOHexperimenter/include/ioh.hpp
)

add_test(NAME integration  COMMAND $<TARGET_FILE:integration>)
