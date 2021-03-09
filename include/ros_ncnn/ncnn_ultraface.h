//  Created by vealocia on 2019/10/17.
//  Copyright © 2019 vealocia. All rights reserved.
//  Modifications by nilseuropa
//

#ifndef _ROS_NCNN_ULTRAFACE_HEADER_
#define _ROS_NCNN_ULTRAFACE_HEADER_

#include "platform.h"
#include "net.h"
#include "cpu.h"
#include "ros_ncnn/ncnn_utils.h"

#pragma once
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#define num_featuremap 4
#define hard_nms 1
#define blending_nms 2 /* mix nms was been proposaled in paper blaze face, aims to minimize the temporal jitter*/

typedef struct FaceInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    // float width;
    // float height;
    float score;
    // float *landmarks;
} FaceInfo;

class ncnnUltraFace {
public:

    void init(int input_width, int input_length, float probability_threshold, float iou_threshold);
    int detect(const cv::Mat& bgr, std::vector<FaceInfo>& face_list, uint8_t n_threads);
    void draw(const cv::Mat& bgr, const std::vector<FaceInfo>& face_info, double dT);

    ncnn::Net neuralnet;

private:
    void generateBBox(std::vector<FaceInfo> &bbox_collection, ncnn::Mat scores, ncnn::Mat boxes, float score_threshold, int num_anchors);
    void nms(std::vector<FaceInfo> &input, std::vector<FaceInfo> &output, int type = blending_nms);

private:

    int num_thread;
    int image_w;
    int image_h;

    int in_w;
    int in_h;
    int num_anchors;

    float score_threshold;
    float iou_threshold;

    const float mean_vals[3] = {127, 127, 127};
    const float norm_vals[3] = {1.0 / 128, 1.0 / 128, 1.0 / 128};

    const float center_variance = 0.1;
    const float size_variance = 0.2;
    const std::vector<std::vector<float>> min_boxes = {
            {10.0f,  16.0f,  24.0f},
            {32.0f,  48.0f},
            {64.0f,  96.0f},
            {128.0f, 192.0f, 256.0f}};
    const std::vector<float> strides = {8.0, 16.0, 32.0, 64.0};
    std::vector<std::vector<float>> featuremap_size;
    std::vector<std::vector<float>> shrinkage_size;
    std::vector<int> w_h_list;

    std::vector<std::vector<float>> priors = {};
};

#endif
