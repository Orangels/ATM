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

    void update(InstanceGroup instance_group);
    bool hand_flag;
};

#endif //ATM_HAND_DETECT_H
