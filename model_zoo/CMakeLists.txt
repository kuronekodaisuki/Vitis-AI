cmake_minimum_required(VERSION 3.8)

project(yolov8)


find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 17)

include_directories(
    "../xnnpp/include"
	"include"
	)

add_library(yolov8 STATIC
    common/common.cpp
	DPU.cpp
  	Object.cpp 
  	YOLOv8.cpp
)

target_include_directories(yolov8 PUBLIC
	${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(yolov8
	${OpenCV_LIBRARIES}
	xnnpp
	dpu_task
)
