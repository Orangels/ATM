//
// Created by 王智慧 on 2020/3/18.
//

#ifndef ATM_HAND_DETECT_H
#define ATM_HAND_DETECT_H

#include <iostream>
#include "structures/structs.h"
#include "structures/image.h"
#include "structures/instance_group.h"

using namespace std;

class HandDetect {
public:
    HandDetect();
    ~HandDetect();

    void update(vector<Box> boxes);

    vector<Box> hand_boxes;
    Box pre_area;
    int area_threshold, hand_frame, hand_wake, hand_sleep;
    bool hand_flag, flag, flag_2;
    deque<int> keep_hand;
};

#endif //ATM_HAND_DETECT_H
