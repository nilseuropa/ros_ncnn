#ifndef _ROS_NCNN_PFLD_HEADER_
#define _ROS_NCNN_PFLD_HEADER_

#include "platform.h"
#include "net.h"
#include "cpu.h"

#include "ros_ncnn/ncnn_utils.h"

class ncnnPFLD {

private:
  const int num_landmarks = 106 * 2;
  float landmarks[212];
  uint8_t num_threads = 8;
  const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};

public:

  ncnn::Net neuralnet;

  int    detect(const cv::Mat& bgr, cv::Rect roi);
  void   draw(const cv::Mat& bgr, cv::Rect roi);
  void   set_threads(uint8_t num);
};

#endif
