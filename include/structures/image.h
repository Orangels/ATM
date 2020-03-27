//
// Created by 王智慧 on 2020/3/18.
//

#ifndef ATM_IMAGE_H
#define ATM_IMAGE_H

#include <vector>
#include <deque>
#include "structures/structs.h"

using namespace std;

class Image {
public:
    Image();
    ~Image();

    void update_hf(vector<Box> head, vector<Box> face);

    int frame_id, sleep_wake_params_1, sleep_wake_params_2, pre_size_lift, pre_size_right, face_th, head_th;
    bool wake_state, invade, group_flag, hand_flag;
    vector<Box> oir_head_boxes, head_boxes, face_boxes, hand_boxes, hop_boxes;
    deque<int> keep_head;
};

#endif //ATM_IMAGE_H
