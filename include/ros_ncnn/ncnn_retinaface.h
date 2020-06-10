#ifndef _ROS_NCNN_RETINAFACE_HEADER_
#define _ROS_NCNN_RETINAFACE_HEADER_

#include <stdio.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "platform.h"
#include "net.h"
#include "cpu.h"

// #include "ros_ncnn/ncnn_utils.h"

struct FaceObject
{
    cv::Rect_<float> rect;
    cv::Point2f landmark[5];
    float prob;
};

class ncnnRetinaface {

public:

  ncnn::Net net;

  int detect_retinaface(const cv::Mat& bgr, std::vector<FaceObject>& faceobjects, uint8_t n_threads);
  void draw_faceobjects(const cv::Mat& bgr, const std::vector<FaceObject>& faceobjects, double dT);

};

#endif
