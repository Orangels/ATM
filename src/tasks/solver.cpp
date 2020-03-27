//
// Created by 王智慧 on 2020/3/18.
//

#include "tasks/solver.h"
#include "config.h"

Solver::Solver() {
//    wake_state = invade = group_flag = hand_flag = false;
//    Cconfig labels = Cconfig("../cfg/process.ini");
//    hop_frame_last = stoi(labels["HOP_FRAME_LAST"]);
//    hop_noise_frame = stoi(labels["HOP_NOISE_FRAME"]);
//    hop_pass_frame = stoi(labels["HOP_PASS_FRAME"]);
//    tround_times = stoi(labels["TROUND_TIMES"]);
//    tround_frame_last = stoi(labels["TROUND_FRAME_LAST"]);
//    tround_pass_frame = stoi(labels["TROUND_PASS_FRAME"]);
//    tround_sense = stof(labels["TROUND_SENSE"]);
//    tround_area_1 = stoi(labels["TROUND_AREA_1"]);
//    tround_area_2 = stoi(labels["TROUND_AREA_2"]);
//    group_max_num = stoi(labels["GROUP_MAX_NUM"]);
//    group_sense = stof(labels["GROUP_SENSE"]);
//    group_area_1 = stoi(labels["GROUP_AREA_1"]);
//    group_area_2 = stoi(labels["GROUP_AREA_2"]);
//    group_min_head_area = stoi(labels["GROUP_MIN_HEAD_AREA"]);
//    group_max_head_area = stoi(labels["GROUP_MAX_HEAD_AREA"]);
//    group_frame_last = stoi(labels["GROUP_FRAME_LAST"]);
//    group_pass_frame = stoi(labels["GROUP_PASS_FRAME"]);
//    group_noise_frame = stoi(labels["GROUP_NOISE_FRAME"]);
}

Solver::~Solver() = default;

void Solver::update(Image image_class, InstanceGroup instance_group){
     group_detect.update(image_class.oir_head_boxes);
}
