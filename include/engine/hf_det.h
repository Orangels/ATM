//
// Created by 王智慧 on 2020/3/18.
//

#ifndef ATM_HF_DET_H
#define ATM_HF_DET_H

#include <vector>
#include <opencv2/opencv.hpp>
#include "structures/structs.h"
#include "detection.h"

using namespace std;

class HFDet {
public:
    HFDet();
    ~HFDet();
    void inference(cv::Mat image, SSD_Detection* ssd_detection);
    
    vector<float> hf_boxes;
    vector<Box> face_boxes, head_boxes;
};

#endif //ATM_HF_DET_H
