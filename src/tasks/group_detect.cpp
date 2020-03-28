//
// Created by 王智慧 on 2020/3/18.
//

#include "tasks/group_detect.h"
#include "utils/misc.h"
#include "config.h"

GroupDetect::GroupDetect() {
    Cconfig labels = Cconfig("../cfg/process.ini");
    group_max_num = stoi(labels["GROUP_MAX_NUM"]);
    group_area_1 = stoi(labels["GROUP_AREA_1"]);
    group_area_2 = stoi(labels["GROUP_AREA_2"]);
    group_min_head_area = stoi(labels["GROUP_MIN_HEAD_AREA"]);
    group_max_head_area = stoi(labels["GROUP_MAX_HEAD_AREA"]);
    group_frame_last = stoi(labels["GROUP_FRAME_LAST"]);
    group_pass_frame = stoi(labels["GROUP_PASS_FRAME"]);
    group_noise_frame = stoi(labels["GROUP_NOISE_FRAME"]);
    group_flag = false;
}

GroupDetect::~GroupDetect() = default;

void GroupDetect::update(vector<Box> boxes){
    head_boxes = get_box_by_area(boxes, group_min_head_area, group_max_head_area);
    if (!head_boxes.empty()){
        group_heads = group_point(head_boxes, group_max_num, group_area_1, group_area_2);
    }
    group_flag = !group_heads.empty();
}