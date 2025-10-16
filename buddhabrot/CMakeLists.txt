cmake_minimum_required(VERSION 4.0)
project(buddhabrot LANGUAGES CXX CUDA)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
#set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G -lineinfo")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

set(SRC_FILES
	main.cpp
	include/io/image.cpp
	include/io/config.cpp
    include/io/cmd.cpp
)

file(GLOB KERNEL_SOURCES "include/kernels/*.cu")

include_directories(
	include
	include/io
	include/kernels
)

add_executable(buddhabrot ${SRC_FILES} ${KERNEL_SOURCES})
# set_target_properties(buddhabrot PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
#     CUDA_RESOLVE_DEVICE_SYMBOLS ON    # This fixes the "hidden symbol" error
#     CUDA_ARCHITECTURES 86
# )

# Optionally, link libraries if needed
# target_link_libraries(buddhabrot ...)

