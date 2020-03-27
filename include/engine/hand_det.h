//
// Created by 王智慧 on 2020/3/21.
//

#ifndef ATM_HAND_DET_H
#define ATM_HAND_DET_H

#include <vector>
#include <opencv2/opencv.hpp>
#include "structures/structs.h"
#include "detection.h"

using namespace std;

class HandDet {
public:
    HandDet();
    ~HandDet();
    void inference(cv::Mat image, SSD_Detection* ssd_detection);

    vector<float> _hand_boxes;
    vector<Box> hand_boxes;
};

#endif //ATM_HAND_DET_H
