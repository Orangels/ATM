//
// Created by 王智慧 on 2020/3/18.
//

#include "tasks/entry_detect.h"
#include "utils/misc.h"
#include "config.h"

EntryDetect::EntryDetect() {
    Cconfig labels = Cconfig("../cfg/process.ini");
    entry_times = stoi(labels["ENTRY_TIMES"]);
    entry_flag = false;
}

EntryDetect::~EntryDetect() = default;

void EntryDetect::update(InstanceGroup instance_group){
    entry_flag = false;
    entry_face.clear();
    Box box;
    for (auto id : instance_group.track_ids){
        if (instance_group.instances[id].frequency >= entry_times){
            entry_flag = true;
            if (instance_group.instances[id].face_box.size() > 0){
                box = instance_group.instances[id].face_box[0];
                entry_face.push_back(box);
            }
        }
    }
}