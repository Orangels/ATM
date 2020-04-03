//
// Created by 王智慧 on 2020/3/18.
//

#include "tasks/turnround_detect.h"
#include "utils/misc.h"
#include "config.h"

TurnroundDetect::TurnroundDetect() {
    Cconfig labels = Cconfig("../cfg/process.ini");
    turnround_frame_1 = stoi(labels["TROUND_FRAME_1"]);
    turnround_frame_2 = stoi(labels["TROUND_FRAME_2"]);
    turnround_angle_1 = stoi(labels["TROUND_ANGLE_1"]);
    turnround_angle_2 = stoi(labels["TROUND_ANGLE_2"]);
    turnround_flag = lock = false;
    turnround_sleep = 0;
    t = 0;
    gettimeofday(&start_time, NULL);
}

TurnroundDetect::~TurnroundDetect() = default;

void TurnroundDetect::update(InstanceGroup instance_group){
    gettimeofday(&now_time, NULL);
    if (now_time.tv_sec - start_time.tv_sec > 120){
        t=0;
        lock = false;
    }
    int angle_y = 0, area = 0, s_1 = 0, s_2 = 0;
    Box box;
    for (auto id : instance_group.track_ids){
        if (instance_group.instances[id].face_box.size() > 0){
            box = instance_group.instances[id].face_box[0];
            if ((box.x2 - box.x1) * (box.y2 - box.y1) > area){
                area = (box.x2 - box.x1) * (box.y2 - box.y1);
                angle_y = instance_group.instances[id].pos_angle.Y;
            }
        }
    }
    if (angle_y != 0){
        keep_turnround_1.push_back(angle_y);
        if (keep_turnround_1.size() == turnround_frame_1){
            for (auto keep : keep_turnround_1){
                s_1 = s_1 + keep;
            }
            keep_turnround_1.pop_front();
        }
        s_1 = s_1 / turnround_frame_1;

        keep_turnround_2.push_back(angle_y);
        if (keep_turnround_2.size() == turnround_frame_2){
            for (auto keep : keep_turnround_2){
                s_2 = s_2 + keep;
            }
            keep_turnround_2.pop_front();
        }
        s_2 = s_2 / turnround_frame_2;

        if (s_1 < -turnround_angle_1 or s_1 > turnround_angle_1){
            if (!lock) {
                if (t == 0){gettimeofday(&start_time, NULL);}
                t++;
                lock = true;
            }
        } else if (s_2 > -turnround_angle_2 and s_2 < turnround_angle_2){
            lock = false;
        }
    }
    if (t == 3){
        turnround_sleep = 20;
        turnround_flag = true;
        t = 0;
    }else{
        if (turnround_sleep > 0){
            turnround_sleep--;
        }else{
            turnround_flag = false;
        }
    }
}