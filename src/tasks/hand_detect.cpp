//
// Created by 王智慧 on 2020/3/18.
//

#include "tasks/hand_detect.h"
#include "utils/misc.h"
#include "config.h"

HandDetect::HandDetect() {
    Cconfig labels = Cconfig("../cfg/process.ini");
    pre_area.x1 = stoi(labels["HAND_PRE_AREA_x1"]);
    pre_area.y1 = stoi(labels["HAND_PRE_AREA_y1"]);
    pre_area.x2 = stoi(labels["HAND_PRE_AREA_x2"]);
    pre_area.y2 = stoi(labels["HAND_PRE_AREA_y2"]);
    area_threshold = stoi(labels["HAND_AREA_THRESHOLD"]);
    hand_frame = stoi(labels["HAND_FRAME"]);
    hand_wake = stoi(labels["HAND_WAKE"]);
    hand_flag = false;
    hand_sleep = 20;
}

HandDetect::~HandDetect() = default;

void HandDetect::update(vector<Box> boxes){
    hand_boxes = invade(pre_area, boxes, area_threshold);
    flag_2 = hand_boxes.size() >= 2;
    keep_hand.push_back(flag_2);
    if (keep_hand.size() == hand_frame){
        flag = is_wake(keep_hand, hand_wake);
        keep_hand.pop_front();
    }
    if (flag){
        hand_sleep = 20;
        hand_flag = flag;
    }else{
        if (hand_sleep > 0){
            hand_sleep--;
        }else{
            hand_flag = flag;
        }
    }
}