//
// Created by 王智慧 on 2020/3/18.
//

#include "tasks/solver.h"
#include "config.h"

Solver::Solver() {
    group_flag = hand_flag = hop_flag = tround_flag = entry_flag = false;
}

Solver::~Solver() = default;

void Solver::update(Image image_class, InstanceGroup instance_group){
    group_detect.update(image_class.oir_head_boxes);
    group_heads = group_detect.group_heads;
    group_flag = group_detect.group_flag;
    hand_detect.update(image_class.hand_boxes);
    hand_flag = hand_detect.hand_flag;
    hop_detect.update(instance_group);
    hop_flag = hop_detect.hop_flag;
}
