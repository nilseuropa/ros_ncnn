cmake_minimum_required(VERSION 2.8.3)
project(ros_ncnn)

add_compile_options(-std=c++11 -DUSE_AVX_INSTRUCTIONS=ON)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  roslib
  cv_bridge
  image_transport
  image_geometry
  sensor_msgs
)

catkin_package(
    CATKIN_DEPENDS
    roscpp
    roslib
    cv_bridge
    image_transport
    image_geometry
    sensor_msgs
)

include_directories(
  "include"
  ${catkin_INCLUDE_DIRS}
)

find_package(ncnn REQUIRED)
if (${ncnn_FOUND})
	message("-- NCNN found.")
  include_directories(${ncnn_INCLUDE})
endif()

find_package(OpenCV QUIET COMPONENTS
  core highgui imgproc imgcodecs
)
include_directories(
  ${OpenCV_INCLUDE_DIRS}
)

find_package(Vulkan) # REQUIRES ncnn to be built with vulkan
if (${VULKAN_FOUND})
  message("-- Vulkan found.")
  # YOLACT
  add_executable(yolact_gpu_node src/yolact_gpu_node.cpp src/ncnn_yolact.cpp)
  target_link_libraries(yolact_gpu_node ${catkin_LIBRARIES} ncnn ${OpenCV_LIBS})
  # RETINAFACE
  add_executable(retinaface_gpu_node src/retinaface_gpu_node.cpp src/ncnn_retinaface.cpp)
  target_link_libraries(retinaface_gpu_node ${catkin_LIBRARIES} ncnn ${OpenCV_LIBS})
endif()

# YOLACT
add_executable(yolact_cpu_node src/yolact_cpu_node.cpp src/ncnn_yolact.cpp)
target_link_libraries(yolact_cpu_node ${catkin_LIBRARIES} ncnn ${OpenCV_LIBS})
# RETINAFACE
add_executable(retinaface_cpu_node src/retinaface_cpu_node.cpp src/ncnn_retinaface.cpp)
target_link_libraries(retinaface_cpu_node ${catkin_LIBRARIES} ncnn ${OpenCV_LIBS})
