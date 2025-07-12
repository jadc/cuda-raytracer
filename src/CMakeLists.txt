cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(raytracer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_executable(raytracer
    main.cu
    render.cu
)

# Enable all warnings, and elevate warnings to errors
target_compile_options(raytracer PUBLIC
             "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options -Werror,-Wall>"
             "$<$<COMPILE_LANGUAGE:CXX>:-Werror;-Wall>")
