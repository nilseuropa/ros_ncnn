#include "ros_ncnn/ncnn_utils.h"
#include "ros_ncnn/ncnn_yolov5.h"

inline float fast_exp(float x)
{
    union {uint32_t i;float f;} v{};
    v.i=(1<<23)*(1.4426950409*x+126.93490512f);
    return v.f;
}

inline float sigmoid(float x){
    return 1.0f / (1.0f + fast_exp(-x));
}

std::vector<Object>
ncnnYoloV5::decode_infer(ncnn::Mat &data, int stride, const cv::Size &frame_size, int net_size, int num_classes,const std::vector<cv::Size>& anchors, float threshold) {
    std::vector<Object> result;
    int grid_size = int(sqrt(data.h));
    float *mat_data[data.c];
    for(int i=0;i<data.c;i++){
        mat_data[i] = data.channel(i);
    }
    float cx,cy,w,h;
    for(int shift_y=0;shift_y<grid_size;shift_y++){
        for(int shift_x=0;shift_x<grid_size;shift_x++){
            int loc = shift_x+shift_y*grid_size;
            for(int i=0;i<3;i++){
                float *record = mat_data[i];
                float *cls_ptr = record + 5;
                for(int cls = 0; cls<num_classes;cls++){
                    float score = sigmoid(cls_ptr[cls]) * sigmoid(record[4]);
                    if(score>threshold){
                        cx = (sigmoid(record[1]) * 2.f - 0.5f + (float)shift_x) * (float) stride;
                        cy = (sigmoid(record[2]) * 2.f - 0.5f + (float)shift_y) * (float) stride;
                        w = pow(sigmoid(record[2]) * 2.f,2)*anchors[i].width;
                        h = pow(sigmoid(record[3]) * 2.f,2)*anchors[i].height;
                        Object object;
                        object.rect.width  = w;
                        object.rect.height = h;
                        object.rect.x = std::max(0,std::min(frame_size.width,int((cx / 2.f) * (float)frame_size.width / (float)net_size))) - w/2;
                        object.rect.y = std::max(0,std::min(frame_size.height,int((cy / 2.f) * (float)frame_size.height / (float)net_size))) -h/2;
                        object.prob = score;
                        object.label = cls;
                        result.push_back(object);
                    }
                }
            }
            for(auto& ptr:mat_data){
                ptr+=(num_classes + 5);
            }
        }
    }
    return result;
}

void ncnnYoloV5::nms(std::vector<Object> &input_boxes, float NMS_THRESH) {
    std::sort(input_boxes.begin(), input_boxes.end(), [](Object a, Object b){return a.prob > b.prob;});
    std::vector<float>vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        vArea[i] = input_boxes.at(i).rect.width * input_boxes.at(i).rect.height;
    }
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        for (int j = i + 1; j < int(input_boxes.size());)
        {
            float xx1 = std::max(input_boxes[i].rect.x, input_boxes[j].rect.x);
            float yy1 = std::max(input_boxes[i].rect.y, input_boxes[j].rect.y);
            float xx2 = std::min(input_boxes[i].rect.x+input_boxes[i].rect.width, input_boxes[j].rect.x+input_boxes[j].rect.width);
            float yy2 = std::min(input_boxes[i].rect.y+input_boxes[i].rect.height, input_boxes[j].rect.y+input_boxes[j].rect.height);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float h = std::max(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH)
            {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else
            {
                j++;
            }
        }
    }
}

int ncnnYoloV5::detect(const cv::Mat& bgr, std::vector<Object>& objects, uint8_t n_threads)
{

  double threshold = 0.5;
  double nms_threshold = 0.05;
  const int target_size = 320;

  ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, target_size, target_size);
  float norm[3] = {1/255.f,1/255.f,1/255.f};
  float mean[3] = {0,0,0};
  in.substract_mean_normalize(mean,norm);

  ncnn::Extractor ex = neuralnet.create_extractor();
  ex.set_num_threads(n_threads);
  ex.input(0, in);

  objects.clear();
  for(const auto& layer: layers){
      ncnn::Mat blob;
      ex.extract(layer.name.c_str(),blob);
      auto boxes = decode_infer(blob,layer.stride,{bgr.cols,bgr.rows},target_size,num_class,layer.anchors,threshold);
      objects.insert(objects.begin(),boxes.begin(),boxes.end());
  }
  nms(objects,nms_threshold);
  return 0;
};

void ncnnYoloV5::draw(const cv::Mat& bgr, const std::vector<Object>& objects, double dT)
{
    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y),
                                      cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::putText(image, std::to_string(1/dT)+" Hz", cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
    cv::imshow("YOLO v5", image);
    cv::waitKey(1);
}
