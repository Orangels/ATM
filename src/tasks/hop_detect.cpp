//
// Created by 王智慧 on 2020/3/18.
//

#include "tasks/hop_detect.h"
#include "utils/misc.h"

HopDetect::HopDetect() {
    hop_flag = false;
}

HopDetect::~HopDetect() = default;

void HopDetect::update(InstanceGroup instance_group){
    hop_flag = false;
    for (auto id : instance_group.track_ids){
        if (hop_flag) {
            break;
        }
        if (instance_group.instances[id].hat_box.size() > 0
        or instance_group.instances[id].glass_box.size() > 0
        or instance_group.instances[id].mask_box.size() > 0){
            hop_flag = true;
        }
    }
}