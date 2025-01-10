cmake_minimum_required(VERSION 3.8)

project(yolov6)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(pybind11 REQUIRED)

add_library(yolov6 STATIC
	DPU.cpp
    Object.cpp
    YOLOv8.cpp
)

target_include_directories(yolov6 PUBLIC
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(yolov6 PRIVATE
    ${OpenCV_LIBRARIES}
)