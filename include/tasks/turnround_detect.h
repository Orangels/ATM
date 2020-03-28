//
// Created by 王智慧 on 2020/3/18.
//

#ifndef ATM_TURNROUND_DETECT_H
#define ATM_TURNROUND_DETECT_H

#include <iostream>
#include "structures/structs.h"
#include "structures/image.h"
#include "structures/instance_group.h"

using namespace std;

class TurnroundDetect {
public:
    TurnroundDetect();
    ~TurnroundDetect();

    void update(InstanceGroup instance_group);
    bool turnround_flag;
};

#endif //ATM_TURNROUND_DETECT_H