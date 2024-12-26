cmake_minimum_required(VERSION 3.8)

project(DPU)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(pybind11 REQUIRED)

add_library(DPU MODULE
	DPU.cpp
    #ResNet50.cpp
    Object.cpp
    YOLOv8.cpp
	Wrapper.cpp
)

target_include_directories(DPU PUBLIC
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(DPU PRIVATE
    ${OpenCV_LIBRARIES}
    pybind11::module
)