cmake_minimum_required(VERSION 2.8.3)
project(ros_ncnn)

add_compile_options(-std=c++11 -DUSE_AVX_INSTRUCTIONS=ON)

option(GPU_SUPPORT "Enable GPU support (Vulkan)" OFF)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  roslib
  cv_bridge
  image_transport
  image_geometry
  sensor_msgs
  message_generation
)

add_message_files(
  FILES
  Vector2D.msg
  Rectangle.msg
  FaceObject.msg
  Object.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
    CATKIN_DEPENDS
    roscpp
    roslib
    cv_bridge
    image_transport
    image_geometry
    sensor_msgs
    message_generation
)

include_directories(
  "include"
  ${catkin_INCLUDE_DIRS}
)

find_package(ncnn REQUIRED)
if (${ncnn_FOUND})
  message("-- NCNN found.")
  message("-- NCNN_VULKAN flag is ${NCNN_VULKAN}")
  if (${NCNN_VULKAN})
    message("-- AUTO-ENABLING GPU_SUPPORT")
    set(GPU_SUPPORT ON)
  endif()
  include_directories(${ncnn_INCLUDE})
endif()

find_package(OpenCV REQUIRED COMPONENTS
  core highgui imgproc imgcodecs
)
include_directories(
  ${OpenCV_INCLUDE_DIRS}
)

if (GPU_SUPPORT)
  message("-- GPU support is ENABLED")
  find_package(Vulkan) # REQUIRES ncnn to be built with vulkan
  if (${VULKAN_FOUND})
    message("-- Vulkan found.")
  else()
    message("-- ERROR: AUTO-DISABLING GPU_SUPPORT, because Vulkan was not found")
    set(GPU_SUPPORT OFF)
  endif()
else()
  message("-- GPU support is DISABLED")
endif()

# NCNN config header exports GPU_SUPPORT definition towards source files
configure_file("include/ros_ncnn/ncnn_config.h.in" "ros_ncnn/ncnn_config.h")
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# RETINAFACE
add_executable(retinaface_node src/retinaface_node.cpp src/ncnn_retinaface.cpp)
target_link_libraries(retinaface_node ${catkin_LIBRARIES} ncnn ${OpenCV_LIBS})

# YOLACT
add_executable(yolact_node src/yolact_node.cpp src/ncnn_yolact.cpp)
target_link_libraries(yolact_node ${catkin_LIBRARIES} ncnn ${OpenCV_LIBS})

# YOLO
add_executable(yolo_node src/yolo_node.cpp src/ncnn_yolo.cpp)
target_link_libraries(yolo_node ${catkin_LIBRARIES} ncnn ${OpenCV_LIBS})

# POSENET
add_executable(posenet_node src/posenet_node.cpp src/ncnn_posenet.cpp)
target_link_libraries(posenet_node ${catkin_LIBRARIES} ncnn ${OpenCV_LIBS})

# FASTER RCNN
add_executable(faster_rcnn_node src/faster_rcnn_node.cpp src/ncnn_fast_rcnn.cpp)
target_link_libraries(faster_rcnn_node ${catkin_LIBRARIES} ncnn ${OpenCV_LIBS})
