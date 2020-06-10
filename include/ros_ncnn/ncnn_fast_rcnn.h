#ifndef _ROS_NCNN_FAST_RCNN_HEADER_
#define _ROS_NCNN_FAST_RCNN_HEADER_

#include "platform.h"
#include "net.h"
#include "cpu.h"

#include "ros_ncnn/ncnn_utils.h"

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static const char* class_names[] = {"background",
    "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"};

class ncnnFastRcnn
{

public:

  ncnn::Net net;

  int detect_faster_rcnn(const cv::Mat& bgr, std::vector<Object>& objects, uint8_t n_threads);
  void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects, double dT);
};

#endif
