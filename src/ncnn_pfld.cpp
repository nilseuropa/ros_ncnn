#include "ros_ncnn/ncnn_utils.h"
#include "ros_ncnn/ncnn_pfld.h"

void ncnnPFLD::set_threads(uint8_t num){
 num_threads = num;
}

int ncnnPFLD::detect(const cv::Mat& bgr, cv::Rect roi)
{
  cv::Mat face_rect = bgr(roi);
  cv::resize(face_rect, face_rect, cv::Size(112, 112));

  ncnn::Mat out;
  ncnn::Mat in = ncnn::Mat::from_pixels(face_rect.data, ncnn::Mat::PIXEL_BGR2RGB, 112, 112);

  in.substract_mean_normalize(0, norm_vals);
  ncnn::Extractor ex = neuralnet.create_extractor();

  ex.set_num_threads(num_threads);
  ex.input("input_1", in);
  ex.extract("415", out);

  for (int j = 0; j < out.w; j++) { landmarks[j] = out[j]; }

  return 0;
};

void ncnnPFLD::draw(const cv::Mat& bgr, cv::Rect roi){

  cv::Mat image = bgr.clone();
  // int capture_width = display_image.cols;
  // int capture_height = display_image.rows;

  for(int i = 0; i < num_landmarks / 2; i++){
      cv::circle(image, cv::Point(landmarks[i * 2] * roi.width + roi.x, landmarks[i * 2 + 1] * roi.height + roi.y),
                 2,cv::Scalar(0, 0, 255), -1);
  }

  cv::imshow("PFDL", image);
  cv::waitKey(1);
};
