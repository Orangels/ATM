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
#include "config.h"


using namespace std;

class TurnroundDetect {
public:
    TurnroundDetect();
    ~TurnroundDetect();

    void update(InstanceGroup instance_group);
    bool turnround_flag, lock;
    int t, turnround_sleep, turnround_frame, turnround_angle;
    struct timeval start_time, now_time;
    deque<int> keep_turnround;
};

#endif //ATM_TURNROUND_DETECT_H