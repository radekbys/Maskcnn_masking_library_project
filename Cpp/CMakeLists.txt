# Specify the minimum CMake version required
cmake_minimum_required(VERSION 3.10)

project(MyProject VERSION 1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)
find_package(onnxruntime REQUIRED)

set(onnxruntime_INCLUDE_DIRS "/usr/local/include/onnxruntime")
set(onnxruntime_LIBRARIES "/usr/local/lib/libonnxruntime.so")

include_directories(${onnxruntime_INCLUDE_DIRS})
include_directories(src/class_headers)

# Local header files
SET(TARGET_H
    src/class_headers/ImageOperations.hpp
    src/class_headers/OrtOperations.hpp
    src/class_headers/Masking.hpp
)

# Local source files
SET(TARGET_SRC
    src/class_sources/ImageOperations.cpp
    src/class_sources/OrtOperations.cpp
    src/class_sources/Masking.cpp
)

add_executable(executable src/main.cpp ${TARGET_SRC})
target_link_libraries(executable PRIVATE ${OpenCV_LIBS} ${onnxruntime_LIBRARIES})
add_library(masking SHARED src/class_sources/Masking.cpp src/class_sources/ImageOperations.cpp src/class_sources/OrtOperations.cpp)
target_link_libraries(masking PRIVATE ${OpenCV_LIBS} ${onnxruntime_LIBRARIES})


set_target_properties(executable PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/bin/${CMAKE_SYSTEM_NAME}${OSBitness}/debug
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/bin/${CMAKE_SYSTEM_NAME}${OSBitness}/release
)

set_target_properties(masking PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR}/bin/${CMAKE_SYSTEM_NAME}${OSBitness}/debug
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR}/bin/${CMAKE_SYSTEM_NAME}${OSBitness}/release
)
