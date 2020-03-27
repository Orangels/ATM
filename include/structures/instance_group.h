//
// Created by 王智慧 on 2020/3/21.
//

#ifndef ATM_INSTANCE_GROUP_H
#define ATM_INSTANCE_GROUP_H

#include <vector>
#include <deque>
#include <unordered_map>
#include "structures/structs.h"
#include "structures/instance.h"

using namespace std;

class InstanceGroup {
public:
    InstanceGroup();
    ~InstanceGroup();

    void update(int frame_id, vector<int> track_id, vector<Box> head_boxes,
            vector<vector<float>> face_angle, vector<int> delete_id);
    void check_state();

    unordered_map<int, Instance> instances;
    vector<int> track_ids, delete_id;

//    void update_hf(vector<Box> head, vector<Box> face);
//
//    int frame_id, sleep_wake_params_1, sleep_wake_params_2, pre_size_lift, pre_size_right, face_th, head_th;
//    bool wake_state, invade, group_flag, hand_flag;
//    vector<Box> oir_head_boxes, head_boxes, face_boxes, hand_boxes, hop_boxes;
//    deque<int> keep_head;
};

#endif //ATM_INSTANCE_GROUP_H
