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
    group_frame = stoi(labels["GROUP_FRAME"]);
    group_wake = stoi(labels["GROUP_WAKE"]);
    group_flag = flag = false;
    group_sleep = 0;
}

GroupDetect::~GroupDetect() = default;

void GroupDetect::update(vector<Box> boxes){
    head_boxes = get_box_by_area(boxes, group_min_head_area, group_max_head_area);
    if (!head_boxes.empty()){
        group_heads = group_point(head_boxes, group_max_num, group_area_1, group_area_2);
    }
    keep_group.push_back(!group_heads.empty());
    if (keep_group.size() == group_frame){
        flag = is_wake(keep_group, group_wake);
        keep_group.pop_front();
    }
    if (flag){
        group_sleep = 20;
        group_flag = flag;
    }else{
        if (group_sleep > 0){
            group_sleep--;
        }else{
            group_flag = flag;
        }
    }
}