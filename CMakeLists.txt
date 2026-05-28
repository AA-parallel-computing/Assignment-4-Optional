cmake_minimum_required(VERSION 3.10)

project(MatrixMultiplication)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_Alignof=alignof")
endif()


add_executable(matmul main.cpp)


if(OpenMP_CXX_FOUND)
    target_link_libraries(matmul PUBLIC OpenMP::OpenMP_CXX)
endif()