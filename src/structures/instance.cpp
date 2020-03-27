//
// Created by 王智慧 on 2020/3/21.
//

#include "structures/instance.h"
#include "utils/misc.h"
#include "config.h"

Instance::Instance() = default;

Instance::Instance(int _track_id, Box box, Angle angle) {
    track_id = _track_id;
    head_box = box;
    pos_angle = angle;
//    wake_state = invade = group_flag = hand_flag = false;
//    Cconfig labels = Cconfig("../cfg/process.ini");
//    sleep_wake_params_1 = stoi(labels["SLEEP_WAKE_PARAMS_1"]);
//    sleep_wake_params_2 = stoi(labels["SLEEP_WAKE_PARAMS_2"]);
//    pre_size_lift = stoi(labels["PRE_SIZE_LIFT"]);
//    pre_size_right = stoi(labels["PRE_SIZE_RIGHT"]);
//    face_th = stoi(labels["FACE_THRESHOLD"]);
//    head_th = stoi(labels["HEAD_THRESHOLD"]);
}

Instance::~Instance() = default;