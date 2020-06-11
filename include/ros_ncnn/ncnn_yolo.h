#ifndef _ROS_NCNN_YOLO_HEADER_
#define _ROS_NCNN_YOLO_HEADER_

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

class ncnnYolo {

public:

  ncnn::Net neuralnet;

  int  detect(const cv::Mat& bgr, std::vector<Object>& objects, uint8_t n_threads);
  void draw(const cv::Mat& bgr, const std::vector<Object>& objects, double dT);

};

#endif
