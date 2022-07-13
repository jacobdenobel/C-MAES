cmake_minimum_required(VERSION 2.8.12)
project(C++MAES)

if(CMAKE_VERSION VERSION_LESS 3.0.0)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag(-std=c++17 COMPILER_SUPPORTS_CXX17)
    check_cxx_compiler_flag(-std=c++0x COMPILER_SUPPORTS_CXX0X)
    if(COMPILER_SUPPORTS_CXX17)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
    elseif(COMPILER_SUPPORTS_CXX0X)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    endif()
else()
    SET(CMAKE_CXX_STANDARD 17)
    SET(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

file(GLOB HEADERS include/*.hpp)
file(GLOB SOURCES src/*.cpp)

add_library(c++maes ${SOURCES})
target_link_libraries(c++maes ${CONAN_LIBS})

target_include_directories(c++maes
INTERFACE
    # Interface paths are relative to the installation path, not the current directory.
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include> 
)

target_compile_features(c++maes
INTERFACE
    cxx_std_17
)


add_executable(main main.cpp)
target_link_libraries(main PRIVATE c++maes)
