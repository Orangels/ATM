//
// Created by 王智慧 on 2020/3/18.
//

#include "tasks/hop_detect.h"
#include "utils/misc.h"
#include "config.h"

HopDetect::HopDetect() {
    Cconfig labels = Cconfig("../cfg/process.ini");
    hop_frame = stoi(labels["HOP_FRAME"]);
    hop_wake = stoi(labels["HOP_WAKE"]);
    hop_flag = false;
    hop_sleep = 20;
}

HopDetect::~HopDetect() = default;

void HopDetect::update(InstanceGroup instance_group){
    flag_2 = false;
    for (auto id : instance_group.track_ids){
        if (flag_2) {
            break;
        }
        if (instance_group.instances[id].hat_box.size() > 0
        or instance_group.instances[id].glass_box.size() > 0
        or instance_group.instances[id].mask_box.size() > 0){
            flag_2 = true;
        }
    }
    keep_hop.push_back(flag_2);
    if (keep_hop.size() == hop_frame){
        flag = is_wake(keep_hop, hop_wake);
        keep_hop.pop_front();
    }
    if (flag){
        hop_sleep = 20;
        hop_flag = flag;
    }else{
        if (hop_sleep > 0){
            hop_sleep--;
        }else{
            hop_flag = flag;
        }
    }

}