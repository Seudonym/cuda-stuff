cmake_minimum_required(VERSION 4.0)
project(buddhabrot LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SRC_FILES
	main.cpp
	include/io/image.cpp
	include/io/config.cpp
    include/io/cmd.cpp
)

file(GLOB KERNEL_SOURCES "include/kernels/*.cu")
set(KERNEL_SOURCES ${KERNEL_SOURCES} "include/complex.cu")

include_directories(
	include
	include/io
	include/kernels
)

add_executable(buddhabrot ${SRC_FILES} ${KERNEL_SOURCES})

set_target_properties(buddhabrot PROPERTIES
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_ARCHITECTURES 86
)

# Optionally, link libraries if needed
# target_link_libraries(buddhabrot ...)

