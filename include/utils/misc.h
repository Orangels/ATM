//
// Created by 王智慧 on 2020/3/20.
//

#ifndef ATM_MISC_H
#define ATM_MISC_H

#include <vector>
#include <array>
#include <deque>
#include <cassert>
#include "structures/structs.h"

using namespace std;

vector<Box> select_meet_face(vector<Box> face_boxes,
        int pre_size_lift, int pre_size_right, int face_th);

vector<Box> select_meet_head(vector<Box> head_boxes,
        int pre_size_lift, int pre_size_right, int head_th);

bool is_wake(deque<int> keep_head, int sleep_wake_params_2);

vector<vector<float>> box2vector(vector<Box> boxes);

vector<Box> get_box_by_area(vector<Box> boxes, int min_area, int max_area);

#endif //ATM_MISC_H
