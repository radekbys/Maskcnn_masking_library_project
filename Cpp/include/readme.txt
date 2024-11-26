Library requires installation and inclusion of OpenCV and OnnxRuntime
they are dynamically linked and consider dependencies

___________________________________________________________________________________________

add to CMakelists.txt:

find_package(OpenCV REQUIRED)
find_package(onnxruntime REQUIRED)
set(onnxruntime_INCLUDE_DIRS "/usr/local/include/onnxruntime")
set(onnxruntime_LIBRARIES "/usr/local/lib/libonnxruntime.so")

With the library included are three header files
-Masking.hpp
-OrtOperations.hpp
-ImageOperations.hpp

___________________________________________________________________________________________

Example CMakeLists.txt:

# Minimum CMake version
cmake_minimum_required(VERSION 3.10)

# Project name
project(TestLinking VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)
find_package(onnxruntime REQUIRED)
set(onnxruntime_INCLUDE_DIRS "/usr/local/include/onnxruntime")
set(onnxruntime_LIBRARIES "/usr/local/lib/libonnxruntime.so")

# Specify the include directory for the header files
include_directories(${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/ ${onnxruntime_INCLUDE_DIRS})

# Define the executable
add_executable(test_link main.cpp)

# Link the shared library to the executable
target_link_libraries(test_link PRIVATE ${CMAKE_SOURCE_DIR}/include/libmasking.so ${OpenCV_LIBS} ${onnxruntime_LIBRARIES})

# Optional: Set the output directory for the executable
set_target_properties(test_link PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)