#include "ros_ncnn/ncnn_utils.h"
#include "ros_ncnn/ncnn_posenet.h"

int ncnnPoseNet::detect(const cv::Mat& bgr, std::vector<KeyPoint>& keypoints, uint8_t n_threads)
{

  int w = bgr.cols;
  int h = bgr.rows;

  ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, w, h, 192, 256);

  const float mean_vals[3] = {0.485f*255.f, 0.456f*255.f, 0.406f*255.f};
  const float norm_vals[3] = {1/0.229f/255.f, 1/0.224f/255.f, 1/0.225f/255.f};
  in.substract_mean_normalize(mean_vals, norm_vals);

  ncnn::Extractor ex = neuralnet.create_extractor();

  ex.input("data", in);
  ex.set_num_threads(n_threads);

  ncnn::Mat out;
  ex.extract("conv3_fwd", out);

  // resolve point from heatmap
  keypoints.clear();
  for (int p = 0; p < out.c; p++)
  {
      const ncnn::Mat m = out.channel(p);

      float max_prob = 0.f;
      int max_x = 0;
      int max_y = 0;
      for (int y = 0; y < out.h; y++)
      {
          const float* ptr = m.row(y);
          for (int x = 0; x < out.w; x++)
          {
              float prob = ptr[x];
              if (prob > max_prob)
              {
                  max_prob = prob;
                  max_x = x;
                  max_y = y;
              }
          }
      }

      KeyPoint keypoint;
      keypoint.p = cv::Point2f(max_x * w / (float)out.w, max_y * h / (float)out.h);
      keypoint.prob = max_prob;

      keypoints.push_back(keypoint);
  }

  return 0;
};


void ncnnPoseNet::draw(const cv::Mat& bgr, const std::vector<KeyPoint>& keypoints, double dT)
{
  cv::Mat image = bgr.clone();

  // draw bone
  static const int joint_pairs[16][2] = {
      {0, 1}, {1, 3}, {0, 2}, {2, 4},
      {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10},
      {5, 11}, {6, 12}, {11, 12},
      {11, 13}, {12, 14}, {13, 15}, {14, 16}
  };

  for (int i = 0; i < 16; i++)
  {
      const KeyPoint& p1 = keypoints[ joint_pairs[i][0] ];
      const KeyPoint& p2 = keypoints[ joint_pairs[i][1] ];

      if (p1.prob < 0.2f || p2.prob < 0.2f)
          continue;

      cv::line(image, p1.p, p2.p, cv::Scalar(255, 0, 0), 2);
  }

  // draw joint
  for (size_t i = 0; i < keypoints.size(); i++)
  {
      const KeyPoint& keypoint = keypoints[i];

      if (keypoint.prob < 0.2f)
          continue;

      cv::circle(image, keypoint.p, 3, cv::Scalar(0, 255, 0), -1);
  }

  cv::putText(image, std::to_string(1/dT)+" Hz", cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
  cv::imshow("PoseNet", image);
  cv::waitKey(1);
};
