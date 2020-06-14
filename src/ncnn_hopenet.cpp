#include "ros_ncnn/ncnn_utils.h"
#include "ros_ncnn/ncnn_hopenet.h"

#include <iostream>
#include <cmath>
#include <cerrno>
#include <cstring>
#include <cfenv>
#include <iomanip>

#define NEAR_0 1e-10
#define ODIM   66

void ncnnHopeNet::initialize()
{
  for (uint i=1; i<67; i++) idx_tensor[i-1] = i;
};

void ncnnHopeNet::softmax(float* z, size_t el) {
 double zmax = -INFINITY;
 double zsum = 0;
 for (size_t i = 0; i < el; i++) if (z[i] > zmax) zmax=z[i];
 for (size_t i=0; i<el; i++) z[i] = std::exp(z[i]-zmax);
 zsum = std::accumulate(z, z+el, 0.0);
 for (size_t i=0; i<el; i++) z[i] = (z[i]/zsum)+NEAR_0;
}

double ncnnHopeNet::getAngle(float* prediction, size_t len)
{
  double expectation[len];
  for (uint i=0; i<len; i++) expectation[i]=idx_tensor[i]*prediction[i];
  double pitch = std::accumulate(expectation, expectation+len, 0.0) * 3 - 99;
  return pitch;
}

int ncnnHopeNet::detect(const cv::Mat& bgr, cv::Rect roi, HeadPose& head_angles, uint8_t n_threads)
{
  cv::Mat faceImg = bgr(roi);
  cv::Mat faceGray;
  cv::cvtColor(faceImg, faceGray, CV_BGR2GRAY);
  cv::Size input_geometry = cv::Size(48,48);
  cv::resize(faceGray, faceGrayResized, input_geometry);

  ncnn::Mat in = ncnn::Mat::from_pixels(faceGrayResized.data, ncnn::Mat::PIXEL_GRAY, 48, 48);
  ncnn::Extractor ex = neuralnet.create_extractor();
  ex.input("data", in);
  ex.set_num_threads(n_threads);

  ncnn::Mat output;
  ex.extract("hybridsequential0_multitask0_dense0_fwd", output);
  float* pred_pitch = output.range(0, ODIM);
  float* pred_roll  = output.range(ODIM, ODIM*2);
  float* pred_yaw   = output.range(ODIM*2, ODIM*3);

  softmax(pred_pitch, ODIM);
  softmax(pred_roll,  ODIM);
  softmax(pred_yaw,   ODIM);

  head_angles.pitch = getAngle(pred_pitch, ODIM);
  head_angles.roll  = getAngle(pred_roll,  ODIM);
  head_angles.yaw   = getAngle(pred_yaw,   ODIM);
  return 0;
};

void ncnnHopeNet::draw(){
  cv::imshow("HOPENET", faceGrayResized);
  cv::waitKey(1);
};
