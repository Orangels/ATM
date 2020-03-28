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
    time_threshold = stoi(labels["HAND_TIME_THRESHOLD"]);
    area_threshold = stoi(labels["HAND_AREA_THRESHOLD"]);
    hand_flag = false;
}

HandDetect::~HandDetect() = default;

void HandDetect::update(vector<Box> boxes){
    hand_boxes = invade(pre_area, boxes, area_threshold);
    hand_flag = hand_boxes.size() >= 2;
}