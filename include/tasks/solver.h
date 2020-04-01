//
// Created by 王智慧 on 2020/3/18.
//

#ifndef ATM_SOLVER_H
#define ATM_SOLVER_H

#include "structures/structs.h"
#include "structures/image.h"
#include "structures/instance_group.h"
#include "tasks/group_detect.h"
#include "tasks/hop_detect.h"
#include "tasks/entry_detect.h"
#include "tasks/hand_detect.h"
#include "tasks/turnround_detect.h"

using namespace std;

class Solver {
public:
    Solver();
    ~Solver();

    void update(Image image_class, InstanceGroup instance_group);

    GroupDetect group_detect;
    HandDetect hand_detect;
    HopDetect hop_detect;
    TurnroundDetect turnround_detect;
    vector<vector<Box>> group_heads;
    bool group_flag, hand_flag, hop_flag, tround_flag, entry_flag;
};

#endif //ATM_SOLVER_H
