//
// Created by 王智慧 on 2020/3/18.
//
#include "engine/hf_det.h"
#include "config.h"

HFDet::HFDet() = default;

HFDet::~HFDet() = default;

void HFDet::inference(cv::Mat image, SSD_Detection* ssd_detection){
    hf_boxes.clear();
    head_boxes.clear();
    face_boxes.clear();
    ssd_detection->detect_hf(image, hf_boxes);
    auto p = hf_boxes.data();
    for (int i = 0; i < (hf_boxes.size() / 6); ++i)
    {
        int label = *p++ ;
        float conf = *p++;
        Box box;
        box.x1 = *p++;
        box.y1 = *p++;
        box.x2 = *p++;
        box.y2 = *p++;
        if (label== 1)  //head
        {
            head_boxes.push_back(box);
        }
        if (label == 2)  //face
        {
            face_boxes.push_back(box);
        }
    }
}