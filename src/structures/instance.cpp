//
// Created by 王智慧 on 2020/3/21.
//

#include "structures/instance.h"
#include "utils/misc.h"
#include "config.h"

Instance::Instance() = default;

Instance::Instance(int _track_id, Box head, Box face) {
    track_id = _track_id;
    head_box = head;
    face_id = -1;
    if (face.x2 > 0){
        face_box.push_back(face);
    }
}

Instance::~Instance() = default;

void Instance::update(Box head, Box face){
    head_box = head;
    face_box.clear();
    if (face.x2 > 0){
        face_box.push_back(face);
    }
}

void Instance::update_hop(Box hat, Box glass, Box mask){
    hat_box.clear();
    glass_box.clear();
    mask_box.clear();
    if (hat.x2 > 0){
        hat_box.push_back(hat);
    }
    if (glass.x2 > 0){
        glass_box.push_back(glass);
    }
    if (mask.x2 > 0){
        mask_box.push_back(mask);
    }
//    pos_angle = angle;
}