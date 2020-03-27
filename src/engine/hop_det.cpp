//
// Created by 王智慧 on 2020/3/21.
//
#include "engine/hop_det.h"
#include "config.h"

HopDet::HopDet() = default;

HopDet::~HopDet() = default;

void HopDet::inference(cv::Mat image, SSD_Detection* ssd_detection){
    hop_boxes.clear();
    hat_boxes.clear();
    glass_boxes.clear();
    mask_boxes.clear();
    ssd_detection->detect_hop(image, hop_boxes);
    auto p = hop_boxes.data();
    for (int i = 0; i < (hop_boxes.size() / 6); ++i)
    {
        int label = *p++ ;
        float conf = *p++;
        Box box;
        box.x1 = *p++;
        box.y1 = *p++;
        box.x2 = *p++;
        box.y2 = *p++;
        if (label== 1)  //hat
        {
            hat_boxes.push_back(box);
        }
        if (label == 2)  //glass_box
        {
            glass_boxes.push_back(box);
        }
        if (label == 3)  //mask
        {
            mask_boxes.push_back(box);
        }
    }
}

