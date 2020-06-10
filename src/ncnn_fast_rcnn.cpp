#include "ros_ncnn/ncnn_utils.h"
#include "ros_ncnn/ncnn_fast_rcnn.h"

int ncnnFastRcnn::detect_faster_rcnn(const cv::Mat& bgr, std::vector<Object>& objects, uint8_t n_threads)
{

    const int target_size = 600;// __C.TEST.SCALES

    const int max_per_image = 100;
    const float confidence_thresh = 0.05f;

    const float nms_threshold = 0.3f;// __C.TEST.NMS

    // scale to target detect size
    int w = bgr.cols;
    int h = bgr.rows;
    float scale = 1.f;
    if (w < h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, w, h);

    const float mean_vals[3] = { 102.9801f, 115.9465f, 122.7717f };
    in.substract_mean_normalize(mean_vals, 0);

    ncnn::Mat im_info(3);
    im_info[0] = h;
    im_info[1] = w;
    im_info[2] = scale;

    // step1, extract feature and all rois
    ncnn::Extractor ex1 = net.create_extractor();
    ex1.set_num_threads(n_threads);

    ex1.input("data", in);
    ex1.input("im_info", im_info);

    ncnn::Mat conv5_relu5;// feature
    ncnn::Mat rois;// all rois
    ex1.extract("conv5_relu5", conv5_relu5);
    ex1.extract("rois", rois);

    // step2, extract bbox and score for each roi
    std::vector< std::vector<Object> > class_candidates;
    for (int i = 0; i < rois.c; i++)
    {
        ncnn::Extractor ex2 = net.create_extractor();

        ncnn::Mat roi = rois.channel(i);// get single roi
        ex2.input("conv5_relu5", conv5_relu5);
        ex2.input("rois", roi);

        ncnn::Mat bbox_pred;
        ncnn::Mat cls_prob;
        ex2.extract("bbox_pred", bbox_pred);
        ex2.extract("cls_prob", cls_prob);

        int num_class = cls_prob.w;
        class_candidates.resize(num_class);

        // find class id with highest score
        int label = 0;
        float score = 0.f;
        for (int i=0; i<num_class; i++)
        {
            float class_score = cls_prob[i];
            if (class_score > score)
            {
                label = i;
                score = class_score;
            }
        }

        // ignore background or low score
        if (label == 0 || score <= confidence_thresh)
            continue;

//         fprintf(stderr, "%d = %f\n", label, score);

        // unscale to image size
        float x1 = roi[0] / scale;
        float y1 = roi[1] / scale;
        float x2 = roi[2] / scale;
        float y2 = roi[3] / scale;

        float pb_w = x2 - x1 + 1;
        float pb_h = y2 - y1 + 1;

        // apply bbox regression
        float dx = bbox_pred[label * 4];
        float dy = bbox_pred[label * 4 + 1];
        float dw = bbox_pred[label * 4 + 2];
        float dh = bbox_pred[label * 4 + 3];

        float cx = x1 + pb_w * 0.5f;
        float cy = y1 + pb_h * 0.5f;

        float obj_cx = cx + pb_w * dx;
        float obj_cy = cy + pb_h * dy;

        float obj_w = pb_w * exp(dw);
        float obj_h = pb_h * exp(dh);

        float obj_x1 = obj_cx - obj_w * 0.5f;
        float obj_y1 = obj_cy - obj_h * 0.5f;
        float obj_x2 = obj_cx + obj_w * 0.5f;
        float obj_y2 = obj_cy + obj_h * 0.5f;

        // clip
        obj_x1 = std::max(std::min(obj_x1, (float)(bgr.cols - 1)), 0.f);
        obj_y1 = std::max(std::min(obj_y1, (float)(bgr.rows - 1)), 0.f);
        obj_x2 = std::max(std::min(obj_x2, (float)(bgr.cols - 1)), 0.f);
        obj_y2 = std::max(std::min(obj_y2, (float)(bgr.rows - 1)), 0.f);

        // append object
        Object obj;
        obj.rect = cv::Rect_<float>(obj_x1, obj_y1, obj_x2-obj_x1+1, obj_y2-obj_y1+1);
        obj.label = label;
        obj.prob = score;

        class_candidates[label].push_back(obj);
    }

    // post process
    objects.clear();
    for (int i = 0; i < (int)class_candidates.size(); i++)
    {
        std::vector<Object>& candidates = class_candidates[i];

        qsort_descent_inplace(candidates);

        std::vector<int> picked;
        nms_sorted_bboxes(candidates, picked, nms_threshold);

        for (int j = 0; j < (int)picked.size(); j++)
        {
            int z = picked[j];
            objects.push_back(candidates[z]);
        }
    }

    qsort_descent_inplace(objects);

    if (max_per_image > 0 && max_per_image < objects.size())
    {
        objects.resize(max_per_image);
    }

    return 0;
}

void ncnnFastRcnn::draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects, double dT)
{
    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

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
    cv::imshow("FAST_RCNN", image);
    cv::waitKey(1);
}
