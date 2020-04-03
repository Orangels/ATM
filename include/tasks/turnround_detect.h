//
// Created by 王智慧 on 2020/3/18.
//

#ifndef ATM_TURNROUND_DETECT_H
#define ATM_TURNROUND_DETECT_H

#include <iostream>
#include <sys/time.h>
#include "structures/structs.h"
#include "structures/image.h"
#include "structures/instance_group.h"

using namespace std;

class TurnroundDetect {
public:
    TurnroundDetect();
    ~TurnroundDetect();

    void update(InstanceGroup instance_group);
    bool turnround_flag, lock;
    int t, turnround_sleep, turnround_frame_1, turnround_frame_2, turnround_angle_1, turnround_angle_2;
    struct timeval start_time, now_time;
    deque<int> keep_turnround_1, keep_turnround_2;
};

#endif //ATM_TURNROUND_DETECT_H