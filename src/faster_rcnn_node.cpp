#include <ros/package.h>
#include <ros/ros.h>
#include <ros/console.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>

#include "ros_ncnn/ncnn_config.h"
#ifdef GPU_SUPPORT
  #include "gpu.h"
#endif
#include "ros_ncnn/ncnn_fast_rcnn.h"

#ifdef GPU_SUPPORT
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
#endif

ncnnFastRcnn rcnn;
bool display_output;
cv_bridge::CvImagePtr cv_ptr;
std::vector<Object> objects;
ros::Time last_time;

void print_objects(const std::vector<Object>& objects){
    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];
        if (obj.prob > 0.15)
        {
          ROS_INFO("%d = %.5f at %.2f %.2f %.2f x %.2f", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);
        }
    }
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg, int n_threads)
{
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    rcnn.detect_faster_rcnn(cv_ptr->image, objects, n_threads);
    print_objects(objects);

    if (display_output) {
      ros::Time current_time = ros::Time::now();
      rcnn.draw_objects(cv_ptr->image, objects, (current_time-last_time).toSec());
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
  ros::init(argc, argv, "fast_rcnn_node");
  ros::NodeHandle nhLocal("~");
  ros::NodeHandle n;

  int gpu_device;
  nhLocal.param("gpu_device", gpu_device, 0);
#ifndef GPU_SUPPORT
  ROS_INFO("R-CNN node running on CPU");
#endif
#ifdef GPU_SUPPORT
  ROS_INFO("R-CNN node with GPU_SUPPORT, selected gpu_device: %d", gpu_device);
  // Check GPU info, override selection with 1st discrete device if the selected gpu_device is non-discrete
  int gpus = ncnn::get_gpu_count();
  ncnn::GpuInfo gpu_info;
  int first_discrete = -1;
  std::string gpu_type_s, selected_gpu_type_s = "UNKNOWN";
  bool selected_discrete = false;
  bool has_discrete = false;
  for (int g=0; g<gpus; g++) {
    gpu_info = ncnn::get_gpu_info(g);
    switch (gpu_info.type) {
      case 0: // discrete
        if (first_discrete < 0) first_discrete = g;
        if (gpu_device == g) selected_discrete = true;
        has_discrete = true;
        gpu_type_s = "DISCRETE";
        break;
      case 1: // integrated
        gpu_type_s = "INTEGRATED";
        break;
      case 2: // virtual
        gpu_type_s = "VIRTUAL";
        break;
      case 3: // cpu
        gpu_type_s = "CPU";
        break;
    }
    if (gpu_device == g) selected_gpu_type_s = gpu_type_s;
    ROS_INFO_STREAM((gpu_device == g ? "[X] " : "[ ] ") << g << " " << gpu_type_s);
    if (!selected_discrete && has_discrete) {
      ROS_INFO_STREAM("OVERRIDING selected gpu_device '" << gpu_device << "' " << selected_gpu_type_s << " with '" << first_discrete << "'");
      gpu_device = first_discrete;
    }
  }
  g_vkdev = ncnn::get_gpu_device(gpu_device);
  g_blob_vkallocator = new ncnn::VkBlobAllocator(g_vkdev);
  g_staging_vkallocator = new ncnn::VkStagingAllocator(g_vkdev);
  rcnn.net.opt.use_vulkan_compute = true;
  rcnn.net.set_vulkan_device(g_vkdev);
#endif

  const std::string package_name = "ros_ncnn";
  std::string path = ros::package::getPath(package_name)+("/assets/models/");
  ROS_INFO("Assets path: %s", path.c_str());
  rcnn.net.load_param((path+("ZF_faster_rcnn_final.param")).c_str());
  rcnn.net.load_model((path+("ZF_faster_rcnn_final.bin")).c_str());

  nhLocal.param("display_output", display_output, true);
  if (display_output) cv::namedWindow("FAST_RCNN",1);

  int num_threads;
  nhLocal.param("num_threads", num_threads, ncnn::get_cpu_count());
  image_transport::ImageTransport it(n);
  image_transport::Subscriber video = it.subscribe("/camera/image_raw", 1, boost::bind(&imageCallback, _1, num_threads));

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
