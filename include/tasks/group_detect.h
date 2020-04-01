//
// Created by 王智慧 on 2020/3/18.
//

#ifndef ATM_GROUP_DETECT_H
#define ATM_GROUP_DETECT_H

#include <iostream>
#include "structures/structs.h"
#include "structures/image.h"
#include "structures/instance_group.h"

using namespace std;

class GroupDetect {
public:
    GroupDetect();
    ~GroupDetect();

    void update(vector<Box> boxes);

    vector<Box> head_boxes;
    vector<vector<Box>> group_heads;
    int group_max_num, group_area_1, group_area_2, group_min_head_area,
        group_max_head_area, group_frame, group_wake, group_sleep;
    bool group_flag, flag;
    deque<int> keep_group;
};

#endif //ATM_GROUP_DETECT_H