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

    void clear();
    void update(int frame_id, vector<int> track_id, vector<Box> head_boxes,
            vector<Box> face_boxes, vector<int> delete_id);
    void add_hop_box(vector<Box> hat_boxes, vector<Box> glass_boxes, vector<Box> mask_boxes);
    void get_face_box(vector<vector<float>> &face_boxes_input);
    void update_face_angle(vector<vector<float>> face_angle);
    void update_track(int frame_id, vector<int> delete_id);
    void check_state();

    unordered_map<int, Instance> instances;
    vector<int> track_ids, track_ids_with_face, track_delete_id;
};

#endif //ATM_INSTANCE_GROUP_H
