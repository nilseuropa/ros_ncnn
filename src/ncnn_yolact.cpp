#include "ros_ncnn/ncnn_utils.h"
#include "ros_ncnn/ncnn_yolact.h"

int ncnnYolact::detect_yolact(const cv::Mat& bgr, std::vector<Object>& objects, uint8_t n_threads)
{

    const int target_size = 550;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, target_size, target_size);

    const float mean_vals[3] = {123.68f, 116.78f, 103.94f};
    const float norm_vals[3] = {1.0/58.40f, 1.0/57.12f, 1.0/57.38f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = net.create_extractor();
    ex.set_num_threads(n_threads);

    ex.input("input.1", in);

    ncnn::Mat maskmaps;
    ncnn::Mat location;
    ncnn::Mat mask;
    ncnn::Mat confidence;

    ex.extract("619", maskmaps);// 138x138 x 32

    ex.extract("816", location);// 4 x 19248
    ex.extract("818", mask);// maskdim 32 x 19248
    ex.extract("820", confidence);// 81 x 19248

    int num_class = confidence.w;
    int num_priors = confidence.h;

    // make priorbox
    ncnn::Mat priorbox(4, num_priors);
    {
        const int conv_ws[5] = {69, 35, 18, 9, 5};
        const int conv_hs[5] = {69, 35, 18, 9, 5};

        const float aspect_ratios[3] = {1.f, 0.5f, 2.f};
        const float scales[5] = {24.f, 48.f, 96.f, 192.f, 384.f};

        float* pb = priorbox;

        for (int p = 0; p < 5; p++)
        {
            int conv_w = conv_ws[p];
            int conv_h = conv_hs[p];

            float scale = scales[p];

            for (int i = 0; i < conv_h; i++)
            {
                for (int j = 0; j < conv_w; j++)
                {
                    // +0.5 because priors are in center-size notation
                    float cx = (j + 0.5f) / conv_w;
                    float cy = (i + 0.5f) / conv_h;

                    for (int k = 0; k < 3; k++)
                    {
                        float ar = aspect_ratios[k];

                        ar = sqrt(ar);

                        float w = scale * ar / 550;
                        float h = scale / ar / 550;

                        // This is for backward compatability with a bug where I made everything square by accident
                        // cfg.backbone.use_square_anchors:
                        h = w;

                        pb[0] = cx;
                        pb[1] = cy;
                        pb[2] = w;
                        pb[3] = h;

                        pb += 4;
                    }
                }
            }
        }
    }

    const float confidence_thresh = 0.05f;
    const float nms_threshold = 0.5f;
    const int keep_top_k = 200;

    std::vector< std::vector<Object> > class_candidates;
    class_candidates.resize(num_class);

    for (int i = 0; i < num_priors; i++)
    {
        const float* conf = confidence.row(i);
        const float* loc = location.row(i);
        const float* pb = priorbox.row(i);
        const float* maskdata = mask.row(i);

        // find class id with highest score
        // start from 1 to skip background
        int label = 0;
        float score = 0.f;
        for (int j=1; j<num_class; j++)
        {
            float class_score = conf[j];
            if (class_score > score)
            {
                label = j;
                score = class_score;
            }
        }

        // ignore background or low score
        if (label == 0 || score <= confidence_thresh)
            continue;

        // CENTER_SIZE
        float var[4] = {0.1f, 0.1f, 0.2f, 0.2f};

        float pb_cx = pb[0];
        float pb_cy = pb[1];
        float pb_w = pb[2];
        float pb_h = pb[3];

        float bbox_cx = var[0] * loc[0] * pb_w + pb_cx;
        float bbox_cy = var[1] * loc[1] * pb_h + pb_cy;
        float bbox_w = (float)(exp(var[2] * loc[2]) * pb_w);
        float bbox_h = (float)(exp(var[3] * loc[3]) * pb_h);

        float obj_x1 = bbox_cx - bbox_w * 0.5f;
        float obj_y1 = bbox_cy - bbox_h * 0.5f;
        float obj_x2 = bbox_cx + bbox_w * 0.5f;
        float obj_y2 = bbox_cy + bbox_h * 0.5f;

        // clip
        obj_x1 = std::max(std::min(obj_x1 * bgr.cols, (float)(bgr.cols - 1)), 0.f);
        obj_y1 = std::max(std::min(obj_y1 * bgr.rows, (float)(bgr.rows - 1)), 0.f);
        obj_x2 = std::max(std::min(obj_x2 * bgr.cols, (float)(bgr.cols - 1)), 0.f);
        obj_y2 = std::max(std::min(obj_y2 * bgr.rows, (float)(bgr.rows - 1)), 0.f);

        // append object
        Object obj;
        obj.rect = cv::Rect_<float>(obj_x1, obj_y1, obj_x2-obj_x1+1, obj_y2-obj_y1+1);
        obj.label = label;
        obj.prob = score;
        obj.maskdata = std::vector<float>(maskdata, maskdata + mask.w);

        class_candidates[label].push_back(obj);
    }

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

    // keep_top_k
    if (keep_top_k < (int)objects.size())
    {
        objects.resize(keep_top_k);
    }

    // generate mask
    for (int i=0; i<objects.size(); i++)
    {
        Object& obj = objects[i];

        cv::Mat mask(maskmaps.h, maskmaps.w, CV_32FC1);
        {
            mask = cv::Scalar(0.f);

            for (int p=0; p<maskmaps.c; p++)
            {
                const float* maskmap = maskmaps.channel(p);
                float coeff = obj.maskdata[p];
                float* mp = (float*)mask.data;

                // mask += m * coeff
                for (int j=0; j<maskmaps.w * maskmaps.h; j++)
                {
                    mp[j] += maskmap[j] * coeff;
                }
            }
        }

        cv::Mat mask2;
        cv::resize(mask, mask2, cv::Size(img_w, img_h));

        // crop obj box and binarize
        obj.mask = cv::Mat(img_h, img_w, CV_8UC1);
        {
            obj.mask = cv::Scalar(0);

            for (int y=0; y<img_h; y++)
            {
                if (y < obj.rect.y || y > obj.rect.y + obj.rect.height)
                    continue;

                const float* mp2 = mask2.ptr<const float>(y);
                uchar* bmp = obj.mask.ptr<uchar>(y);

                for (int x=0; x<img_w; x++)
                {
                    if (x < obj.rect.x || x > obj.rect.x + obj.rect.width)
                        continue;

                    bmp[x] = mp2[x] > 0.5f ? 255 : 0;
                }
            }
        }
    }

    return 0;
}

void ncnnYolact::draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects, double dT)
{

    static const unsigned char colors[19][3] = {
        {244,  67,  54},
        {233,  30,  99},
        {156,  39, 176},
        {103,  58, 183},
        { 63,  81, 181},
        { 33, 150, 243},
        {  3, 169, 244},
        {  0, 188, 212},
        {  0, 150, 136},
        { 76, 175,  80},
        {139, 195,  74},
        {205, 220,  57},
        {255, 235,  59},
        {255, 193,   7},
        {255, 152,   0},
        {255,  87,  34},
        {121,  85,  72},
        {158, 158, 158},
        { 96, 125, 139}
    };

    cv::Mat image = bgr.clone();

    int color_index = 0;

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        if (obj.prob < 0.15)
            continue;

        const unsigned char* color = colors[color_index++];

        cv::rectangle(image, obj.rect, cv::Scalar(color[0], color[1], color[2]));

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

        // draw mask
        for (int y=0; y<image.rows; y++)
        {
            const uchar* mp = obj.mask.ptr(y);
            uchar* p = image.ptr(y);
            for (int x=0; x<image.cols; x++)
            {
                if (mp[x] == 255)
                {
                    p[0] = cv::saturate_cast<uchar>(p[0] * 0.5 + color[0] * 0.5);
                    p[1] = cv::saturate_cast<uchar>(p[1] * 0.5 + color[1] * 0.5);
                    p[2] = cv::saturate_cast<uchar>(p[2] * 0.5 + color[2] * 0.5);
                }
                p += 3;
            }
        }
    }

    cv::putText(image, std::to_string(1/dT)+" Hz", cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
    cv::imshow("YOLACT", image);
    cv::waitKey(1);
}
