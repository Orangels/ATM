//
// Created by 王智慧 on 2020/3/20.
//

#include "utils/misc.h"
#include "config.h"

vector<Box> select_meet_face(vector<Box> face_boxes,
        int pre_size_lift, int pre_size_right, int face_th){
    float area, c_x;
    vector<Box> meet_face_boxes;

    if (!face_boxes.empty()) {
        for (auto &face_box : face_boxes){
            area = (face_box.x2 - face_box.x1) * (face_box.y2 - face_box.y1);
            if (area > face_th){
                c_x = 0.5 * (face_box.x1 + face_box.x2);
                if (c_x > pre_size_lift and c_x < pre_size_right){
                    meet_face_boxes.push_back(face_box);
                }
            }
        }
    }
    return meet_face_boxes;
}

vector<Box> select_meet_head(vector<Box> head_boxes,
        int pre_size_lift, int pre_size_right, int head_th){
    float area, c_x;
    vector<Box> meet_head_boxes;

    if (!head_boxes.empty()) {
        for (auto &head_box : head_boxes) {
            area = (head_box.x2 - head_box.x1) * (head_box.y2 - head_box.y1);
            if (area > head_th) {
                c_x = 0.5 * (head_box.x1 + head_box.x2);
                if (c_x > pre_size_lift and c_x < pre_size_right) {
                    meet_head_boxes.push_back(head_box);
                    if (meet_head_boxes.size() == 3) {
                        break;
                    }
                }
            }
        }
    }
    return meet_head_boxes;
}

bool is_wake(deque<int> keep_head, int sleep_wake_params_2){
    int num = 0;
    assert(keep_head.size() == 6);
    for (auto &keep : keep_head) {
        if (keep > 0){
            num++;
        }
    }
    return  (num >= sleep_wake_params_2);
}

vector<vector<float>> box2vector(vector<Box> boxes) {
    vector<vector<float>> results;
    vector<float> result;
    for (auto box : boxes){
        result.clear();
        result.push_back(box.x1);
        result.push_back(box.y1);
        result.push_back(box.x2);
        result.push_back(box.y2);
        results.push_back(result);
    }
    return results;
}

vector<Box> get_box_by_area(vector<Box> boxes, int min_area, int max_area){
    vector<Box> output;
    for (auto box : boxes){
        int s = (box.x2 - box.x1) * (box.y2 - box.y1);
        if (s >= min_area and s <= max_area){
            output.push_back(box);
        }
    }
    return output;
}