//
// Created by 王智慧 on 2020/3/18.
//

#ifndef ATM_HOP_DETECT_H
#define ATM_HOP_DETECT_H

#include <iostream>
#include "structures/structs.h"
#include "structures/image.h"
#include "structures/instance_group.h"

using namespace std;

class HopDetect {
public:
    HopDetect();
    ~HopDetect();

    void update(InstanceGroup instance_group);
    bool hop_flag, flag, flag_2;
    int hop_frame, hop_wake, hop_sleep;
    deque<int> keep_hop;
};

#endif //ATM_HOP_DETECT_H
