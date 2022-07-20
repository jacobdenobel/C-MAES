cmake_minimum_required(VERSION 3.12)
project(CMAES)
  
SET(CMAKE_CXX_STANDARD 17) 
SET(CMAKE_CXX_STANDARD_REQUIRED ON)


include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake) 
conan_basic_setup()


file(GLOB SOURCES src/*.cpp) 
message(${SOURCES})
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

target_precompile_headers(integration 
  PRIVATE 
  external/IOHexperimenter/include/ioh.hpp
  
)

add_test(NAME integration  COMMAND $<TARGET_FILE:integration>)
