//
// Created by 王智慧 on 2020/3/21.
//
#include "engine/hand_det.h"
#include "config.h"

HandDet::HandDet()  = default;

HandDet::~HandDet() = default;

void HandDet::inference(cv::Mat image, SSD_Detection* ssd_detection){
    _hand_boxes.clear();
    hand_boxes.clear();
    ssd_detection->detect_hand(image, _hand_boxes);
    auto p = _hand_boxes.data();
    for (int i = 0; i < (_hand_boxes.size() / 6); ++i)
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
            hand_boxes.push_back(box);
        }
    }
}
