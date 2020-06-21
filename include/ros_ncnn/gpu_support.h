#ifndef _NODE_GPU_SUPPORT_HEADER_
#define _NODE_GPU_SUPPORT_HEADER_

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

int selectGPU(int gpu_device){
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
      return gpu_device;
    if (!selected_discrete && has_discrete) {
      ROS_INFO_STREAM("OVERRIDING selected gpu_device '" << gpu_device << "' " << selected_gpu_type_s << " with '" << first_discrete << "'");
      gpu_device = first_discrete;
      return gpu_device;
    }
  }
  return -1;
};

#endif
