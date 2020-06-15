#ifndef _ROS_NCNN_HOPENET_HEADER_
#define _ROS_NCNN_HOPENET_HEADER_

#include "platform.h"
#include "net.h"
#include "cpu.h"

#include "ros_ncnn/ncnn_utils.h"

struct HeadPose
{
    float roll;
    float pitch;
    float yaw;
};

class ncnnHopeNet {

private:
  float   idx_tensor[66];
  cv::Mat faceGrayResized;

public:

  ncnn::Net neuralnet;

  void   initialize();
  void   softmax(float* z, size_t el);
  double getAngle(float* prediction, size_t len);
  int    detect(const cv::Mat& bgr, cv::Rect roi, HeadPose& euler_angles);
  void   draw();

};

#endif
