cmake_minimum_required(VERSION 3.16)
project(numpy_port LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(npcore
    src/allocator.cpp
    src/broadcast.cpp
    src/kernels.cpp
    src/linalg.cpp
    src/ndarray.cpp
    src/reductions.cpp
)

target_include_directories(npcore PUBLIC include)

target_compile_options(npcore PRIVATE -Wall -Wextra -Wpedantic)

add_library(npwasm
    wasm/wasm_api.cpp
)

set_target_properties(npwasm PROPERTIES OUTPUT_NAME "numpy_port")

target_link_libraries(npwasm PRIVATE npcore)
target_include_directories(npwasm PUBLIC include)

add_executable(np_example examples/main.cpp)
target_link_libraries(np_example PRIVATE npcore)

include(CTest)
if(BUILD_TESTING)
    add_executable(np_tests tests/test_main.cpp)
    target_link_libraries(np_tests PRIVATE npcore)
    add_test(NAME np_tests COMMAND np_tests)
endif()
