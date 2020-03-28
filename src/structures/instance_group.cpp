//
// Created by 王智慧 on 2020/3/21.
//

#include "structures/instance_group.h"
#include "utils/misc.h"
#include "config.h"

InstanceGroup::InstanceGroup() {
}

InstanceGroup::~InstanceGroup() = default;

void InstanceGroup::clear(){
    track_ids.clear();
    track_ids_with_face.clear();
}

void InstanceGroup::update(int frame_id, vector<int> track_id, vector<Box> head_boxes,
        vector<Box> face_boxes, vector<int> delete_id){
    for (auto &id : delete_id){
        instances.erase(id);
    }
    for (int i = 0; i < head_boxes.size(); i++){
        track_ids.push_back(track_id[i]);
        Box face_box;
        face_box = get_inside(head_boxes[i], face_boxes);
        if (instances.find(track_id[i]) == instances.end()){
            instances[track_id[i]].update(head_boxes[i], face_box);
        } else{
            Instance instance(track_id[i], head_boxes[i], face_box);
            instances[track_id[i]] = instance;
        }
    }
}

void InstanceGroup::add_hop_box(vector<Box> hat_boxes, vector<Box> glass_boxes, vector<Box> mask_boxes){
    for (auto &id : track_ids){
        Box hat_box, glass_box, mask_box;
        hat_box = get_inside(instances[id].head_box, hat_boxes);
        glass_box = get_inside(instances[id].head_box, glass_boxes);
        mask_box = get_inside(instances[id].head_box, mask_boxes);
        instances[id].update_hop(hat_box, glass_box, mask_box);
    }
}

void InstanceGroup::get_face_box(vector<vector<float>> &face_boxes_input){
    vector<float> result;
    for(auto &track_id : track_ids){
        if(!instances[track_id].face_box.empty()){
            result.clear();
            result.push_back(instances[track_id].face_box[0].x1);
            result.push_back(instances[track_id].face_box[0].y1);
            result.push_back(instances[track_id].face_box[0].x2);
            result.push_back(instances[track_id].face_box[0].y2);
            face_boxes_input.push_back(result);
            track_ids_with_face.push_back(track_id);
        }
    }
}

void InstanceGroup::update_face_angle(vector<vector<float>> face_angle){
    for (int i = 0; i < track_ids_with_face.size(); i++){
        Angle angle;
        angle.Y = face_angle[i][0];
        angle.P = face_angle[i][1];
        angle.R = face_angle[i][2];
        instances[track_ids_with_face[i]].pos_angle = angle;
    }
}

void InstanceGroup::update_track(int frame_id, vector<int> delete_id){
    for (auto &id : delete_id){
        instances.erase(id);
    }
}

void InstanceGroup::check_state(){
    cout<<"ins: "<<instances.size();
    for(auto &ins : instances){
        cout<<" "<<ins.first;
    }
    cout<<endl;
}