//
// Created by 王智慧 on 2020/3/18.
//

#include "structures/image.h"
#include "utils/misc.h"
#include "config.h"

Image::Image() {
    wake_state = invade = group_flag = hand_flag = false;
    Cconfig labels = Cconfig("../cfg/process.ini");
    sleep_wake_params_1 = stoi(labels["SLEEP_WAKE_PARAMS_1"]);
    sleep_wake_params_2 = stoi(labels["SLEEP_WAKE_PARAMS_2"]);
    pre_size_lift = stoi(labels["PRE_SIZE_LIFT"]);
    pre_size_right = stoi(labels["PRE_SIZE_RIGHT"]);
    face_th = stoi(labels["FACE_THRESHOLD"]);
    head_th = stoi(labels["HEAD_THRESHOLD"]);
}

Image::~Image() = default;

void Image::update_hf(vector<Box> head, vector<Box> face) {
    oir_head_boxes = head;
    head_boxes = select_meet_head(head, pre_size_lift, pre_size_right, head_th);
    face_boxes = select_meet_face(face, pre_size_lift, pre_size_right, face_th);
    keep_head.push_back(head_boxes.size());
    if (keep_head.size() == sleep_wake_params_1){
        wake_state = is_wake(keep_head, sleep_wake_params_2);
        keep_head.pop_front();
    }
}
