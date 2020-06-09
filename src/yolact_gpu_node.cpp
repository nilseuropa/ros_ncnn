#include <ros/package.h>
#include <ros/ros.h>
#include <ros/console.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>

#include "gpu.h"
#include "ros_ncnn/nncn_yolact.h"

class GlobalGpuInstance
{
public:
    GlobalGpuInstance() { ncnn::create_gpu_instance(); }
    ~GlobalGpuInstance() { ncnn::destroy_gpu_instance(); }
};
// Initialize Vulkan runtime before main() // !!!
GlobalGpuInstance g_global_gpu_instance;

static ncnn::VulkanDevice* g_vkdev = 0;
static ncnn::VkAllocator* g_blob_vkallocator = 0;
static ncnn::VkAllocator* g_staging_vkallocator = 0;

ncnnYolact yolact;
bool display_output;
cv_bridge::CvImagePtr cv_ptr;
std::vector<Object> objects;
ros::Time last_time;

void imageCallback(const sensor_msgs::ImageConstPtr& msg, int n_threads)
{
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    yolact.detect_yolact(cv_ptr->image, objects, n_threads);

    if (display_output) {
      ros::Time current_time = ros::Time::now();
      yolact.draw_objects(cv_ptr->image, objects, (current_time-last_time).toSec());
      last_time = current_time;
    }
  }
  catch (cv_bridge::Exception& e) {
    ROS_ERROR("CV bridge exception: %s", e.what());
    return;
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "yolact_node");
  ros::NodeHandle nhLocal("~");
  ros::NodeHandle n;

  int gpu_device;
  nhLocal.param("gpu_device", gpu_device, 0);
  g_vkdev = ncnn::get_gpu_device(gpu_device);
  g_blob_vkallocator = new ncnn::VkBlobAllocator(g_vkdev);
  g_staging_vkallocator = new ncnn::VkStagingAllocator(g_vkdev);
  yolact.net.opt.use_vulkan_compute = true;
  yolact.net.set_vulkan_device(g_vkdev);

  const std::string package_name = "ros_ncnn";
  std::string path = ros::package::getPath(package_name)+("/assets/models/");
  ROS_INFO("Assets path: %s", path.c_str());
  yolact.net.load_param((path+("yolact.param")).c_str());
  yolact.net.load_model((path+("yolact.bin")).c_str());

  nhLocal.param("display_output", display_output, true);
  if (display_output) cv::namedWindow("YOLACT",1);

  int num_threads;
  nhLocal.param("num_threads", num_threads, ncnn::get_cpu_count());
  image_transport::ImageTransport it(n);
  image_transport::Subscriber video = it.subscribe("/camera/image_raw", 1, boost::bind(&imageCallback, _1, num_threads));

  ncnn::create_gpu_instance();
  while (ros::ok()) {
    ros::spinOnce();
  }
  ncnn::destroy_gpu_instance();

  return 0;
}
