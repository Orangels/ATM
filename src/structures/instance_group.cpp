//
// Created by 王智慧 on 2020/3/21.
//

#include "structures/instance_group.h"
#include "utils/misc.h"
#include "config.h"

InstanceGroup::InstanceGroup() {
//    wake_state = invade = group_flag = hand_flag = false;
//    Cconfig labels = Cconfig("../cfg/process.ini");
//    sleep_wake_params_1 = stoi(labels["SLEEP_WAKE_PARAMS_1"]);
//    sleep_wake_params_2 = stoi(labels["SLEEP_WAKE_PARAMS_2"]);
//    pre_size_lift = stoi(labels["PRE_SIZE_LIFT"]);
//    pre_size_right = stoi(labels["PRE_SIZE_RIGHT"]);
//    face_th = stoi(labels["FACE_THRESHOLD"]);
//    head_th = stoi(labels["HEAD_THRESHOLD"]);
}

InstanceGroup::~InstanceGroup() = default;

void InstanceGroup::update(int frame_id, vector<int> track_id, vector<Box> head_boxes,
        vector<vector<float>> face_angle, vector<int> _delete_id){
    delete_id = _delete_id;
    for (auto &id : _delete_id){
        instances.erase(id);
    }
    std::cout << "face angle size -- " << face_angle.size() << std::endl;
    std::cout << "head box size -- " << head_boxes.size() << std::endl;
    for (int i = 0; i < head_boxes.size(); i++){
        Angle angle;
        angle.Y = face_angle[i][0];
        angle.P = face_angle[i][1];
        angle.R = face_angle[i][2];
        if (instances.find(track_id[i]) == instances.end()){
            instances[track_id[i]].head_box = head_boxes[i];
            instances[track_id[i]].pos_angle = angle;
        } else{
            Instance instance(track_id[i], head_boxes[i], angle);
            instances[track_id[i]] = instance;
            instances.insert(pair<int, Instance>(track_id[i], instance));
        }
    }
}

void InstanceGroup::check_state(){
    cout<<"ins: "<<instances.size();
    for(auto &ins : instances){
        cout<<" "<<ins.first;
    }
    cout<<endl;
}