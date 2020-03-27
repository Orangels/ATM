//
// Created by 王智慧 on 2020/3/21.
//

#ifndef ATM_HOP_DET_H
#define ATM_HOP_DET_H

#include <vector>
#include <opencv2/opencv.hpp>
#include "structures/structs.h"
#include "detection.h"

using namespace std;

class HopDet {
public:
    HopDet();
    ~HopDet();
    void inference(cv::Mat image, SSD_Detection* ssd_detection);

    vector<float> hop_boxes;
    vector<Box> hat_boxes, glass_boxes, mask_boxes;
};

#endif //ATM_HOP_DET_H
