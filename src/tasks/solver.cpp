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
     group_flag = group_detect.group_flag;

}
