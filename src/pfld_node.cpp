#include <ros/package.h>
#include <ros/ros.h>
#include <ros/console.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>

#include "ros_ncnn/ncnn_config.h"
#ifdef GPU_SUPPORT
  #include "gpu.h"
  #include "ros_ncnn/gpu_support.h"
#endif

/////////////////////////////////
#include "ros_ncnn/ncnn_pfld.h"
#include "ros_ncnn/FaceObject.h"
ncnnPFLD engine;
/////////////////////////////////

cv_bridge::CvImagePtr cv_ptr;

void boundingBoxUpdate(const ros_ncnn::FaceObject& msg)
{
  cv::Rect roi;
  roi.x = msg.boundingbox.position.x;
  roi.y = msg.boundingbox.position.y;
  roi.width  = msg.boundingbox.size.x;
  roi.height = msg.boundingbox.size.y;

  engine.detect(cv_ptr->image, roi);
  engine.draw(cv_ptr->image, roi);
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  }
  catch (cv_bridge::Exception& e) {
    ROS_ERROR("CV bridge exception: %s", e.what());
    return;
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "hopenet_node");
  ros::NodeHandle nhLocal("~");
  ros::NodeHandle n;
  std::string node_name = ros::this_node::getName();
  int gpu_device;
  nhLocal.param("gpu_device", gpu_device, 0);
  int num_threads;
  nhLocal.param("num_threads", num_threads, 0);
  engine.set_threads(num_threads);

#ifndef GPU_SUPPORT
  ROS_WARN_STREAM(node_name << " running on CPU");
#endif
#ifdef GPU_SUPPORT
  ROS_INFO_STREAM(node_name << " with GPU_SUPPORT, selected gpu_device: " << gpu_device);
  g_vkdev = ncnn::get_gpu_device(selectGPU(gpu_device));
  g_blob_vkallocator = new ncnn::VkBlobAllocator(g_vkdev);
  g_staging_vkallocator = new ncnn::VkStagingAllocator(g_vkdev);
  engine.neuralnet.opt.use_vulkan_compute = true;
  engine.neuralnet.set_vulkan_device(g_vkdev);
#endif

  const std::string package_name = "ros_ncnn";
  std::string path = ros::package::getPath(package_name)+("/assets/models/");
  ROS_INFO("Assets path: %s", path.c_str());
  engine.neuralnet.load_param((path+"pfld-sim.param").c_str());
  engine.neuralnet.load_model((path+"pfld-sim.bin").c_str());

  image_transport::ImageTransport it(n);
  image_transport::Subscriber video = it.subscribe("/head_camera/image_raw", 1, imageCallback);
  ros::Subscriber facebox_sub = n.subscribe("/ultraface_node/faces", 10, boundingBoxUpdate );

#ifdef GPU_SUPPORT
  ncnn::create_gpu_instance();
#endif

  while (ros::ok()) {
    ros::spinOnce();
  }

#ifdef GPU_SUPPORT
  ncnn::destroy_gpu_instance();
#endif

  return 0;
}
