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

/////////////////////////////////////
#include "ros_ncnn/ncnn_retinaface.h"
#include "ros_ncnn/FaceObject.h"
ncnnRetinaface engine;
/////////////////////////////////////

ros_ncnn::FaceObject faceMsg;
ros::Publisher face_pub;
cv_bridge::CvImagePtr cv_ptr;
std::vector<FaceObject> faceobjects;
ros::Time last_time;
double prob_threshold;
bool display_output;
bool enable_gpu;

void imageCallback(const sensor_msgs::ImageConstPtr& msg, int n_threads)
{
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    engine.detect(cv_ptr->image, faceobjects, n_threads);
    ros::Time current_time = ros::Time::now();
    for (size_t i = 0; i < faceobjects.size(); i++)
    {
        const FaceObject& obj = faceobjects[i];
        if (obj.prob > prob_threshold)
        {
          ROS_INFO("%.5f at %.2f %.2f %.2f x %.2f",
          obj.prob, obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);
          faceMsg.header.seq++;
          faceMsg.header.stamp = current_time;
          faceMsg.probability = obj.prob;
          faceMsg.boundingbox.position.x = obj.rect.x;
          faceMsg.boundingbox.position.y = obj.rect.y;
          faceMsg.boundingbox.size.x = obj.rect.width;
          faceMsg.boundingbox.size.y = obj.rect.height;
          for (uint8_t i=0; i<5; i++){
          faceMsg.landmark[i].x = obj.landmark[i].x;
          faceMsg.landmark[i].y = obj.landmark[i].y;}
          face_pub.publish(faceMsg);
        }
    }
    if (display_output) {
      engine.draw(cv_ptr->image, faceobjects, (current_time-last_time).toSec());
    }
    last_time = current_time;
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
  std::string node_name = ros::this_node::getName();
  int gpu_device;
  nhLocal.param("gpu_device", gpu_device, 0);
  nhLocal.param("enable_gpu", enable_gpu, true); // for benchmarking reasons

  #ifndef GPU_SUPPORT
    ROS_WARN_STREAM(node_name << " running on CPU");
  #endif
  #ifdef GPU_SUPPORT
    ROS_INFO_STREAM(node_name << " with GPU_SUPPORT, selected gpu_device: " << gpu_device);
    g_vkdev = ncnn::get_gpu_device(selectGPU(gpu_device));
    g_blob_vkallocator = new ncnn::VkBlobAllocator(g_vkdev);
    g_staging_vkallocator = new ncnn::VkStagingAllocator(g_vkdev);
    engine.neuralnet.opt.use_vulkan_compute = enable_gpu;
    engine.neuralnet.set_vulkan_device(g_vkdev);
  #endif

  const std::string package_name = "ros_ncnn";
  std::string path = ros::package::getPath(package_name)+("/assets/models/");
  ROS_INFO("Assets path: %s", path.c_str());
  engine.neuralnet.load_param((path+("mnet.25-opt.param")).c_str());
  engine.neuralnet.load_model((path+("mnet.25-opt.bin")).c_str());

  int num_threads;
  nhLocal.param("num_threads", num_threads, ncnn::get_cpu_count());
  nhLocal.param("probability_threshold", prob_threshold, 0.5);
  nhLocal.param("display_output", display_output, true);

  face_pub = n.advertise<ros_ncnn::FaceObject>(node_name+"/faces", 10);

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
