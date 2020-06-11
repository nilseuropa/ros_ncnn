// model is converted from
// https://github.com/deepinsight/insightface/tree/master/RetinaFace#retinaface-pretrained-models
// https://github.com/deepinsight/insightface/issues/669

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
#include "ros_ncnn/ncnn_retinaface.h"

class GlobalGpuInstance
{
public:
    GlobalGpuInstance() { ncnn::create_gpu_instance(); }
    ~GlobalGpuInstance() { ncnn::destroy_gpu_instance(); }
};

GlobalGpuInstance g_global_gpu_instance;

static ncnn::VulkanDevice* g_vkdev = 0;
static ncnn::VkAllocator* g_blob_vkallocator = 0;
static ncnn::VkAllocator* g_staging_vkallocator = 0;

ncnnRetinaface retinaface;
bool display_output;
cv_bridge::CvImagePtr cv_ptr;
std::vector<FaceObject> faceobjects;
ros::Time last_time;

void print_objects(const std::vector<FaceObject>& objects){
    for (size_t i = 0; i < objects.size(); i++)
    {
        const FaceObject& obj = objects[i];
        if (obj.prob > 0.5)
        {
          ROS_INFO("%.5f at %.2f %.2f %.2f x %.2f",
          obj.prob, obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);
        }
    }
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg, int n_threads)
{
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    retinaface.detect_retinaface(cv_ptr->image, faceobjects, n_threads);
    print_objects(faceobjects); // TODO: faceobject message publisher

    if (display_output) {
      ros::Time current_time = ros::Time::now();
      retinaface.draw_faceobjects(cv_ptr->image, faceobjects, (current_time-last_time).toSec());
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
  ros::init(argc, argv, "retinaface_node");
  ros::NodeHandle nhLocal("~");
  ros::NodeHandle n;

  int gpu_device;
  nhLocal.param("gpu_device", gpu_device, 0);
  g_vkdev = ncnn::get_gpu_device(gpu_device);
  g_blob_vkallocator = new ncnn::VkBlobAllocator(g_vkdev);
  g_staging_vkallocator = new ncnn::VkStagingAllocator(g_vkdev);
  retinaface.net.opt.use_vulkan_compute = true;
  retinaface.net.set_vulkan_device(g_vkdev);

  const std::string package_name = "ros_ncnn";
  std::string path = ros::package::getPath(package_name)+("/assets/models/");
  ROS_INFO("Assets path: %s", path.c_str());
  retinaface.net.load_param((path+("mnet.25-opt.param")).c_str());
  retinaface.net.load_model((path+("mnet.25-opt.bin")).c_str());

  nhLocal.param("display_output", display_output, true);
  if (display_output) cv::namedWindow("RETINAFACE",1);

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
