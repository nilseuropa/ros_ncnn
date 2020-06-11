#ifndef _ROS_NCNN_POSENET_HEADER_
#define _ROS_NCNN_POSENET_HEADER_

#include "platform.h"
#include "net.h"
#include "cpu.h"

#include "ros_ncnn/ncnn_utils.h"

struct KeyPoint
{
    cv::Point2f p;
    float prob;
};

class ncnnPoseNet {

public:

  ncnn::Net neuralnet;

  int  detect(const cv::Mat& bgr, std::vector<KeyPoint>& keypoints, uint8_t n_threads);
  void draw(const cv::Mat& bgr, const std::vector<KeyPoint>& keypoints, double dT);

};

#endif
