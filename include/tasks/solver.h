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
    int hop_frame_last, hop_noise_frame, hop_pass_frame, tround_times, tround_frame_last, tround_pass_frame;
    float tround_sense, group_sense;
    int tround_area_1, tround_area_2, group_max_num, group_area_1, group_area_2, group_min_head_area,
        group_max_head_area, group_frame_last, group_pass_frame, group_noise_frame;
    bool group_flag, hand_flag, hop_flag, tround_flag, entry_flag;
};


#endif //ATM_SOLVER_H
