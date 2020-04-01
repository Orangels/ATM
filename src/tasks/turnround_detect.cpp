//
// Created by 王智慧 on 2020/3/18.
//

#include "tasks/turnround_detect.h"
#include "utils/misc.h"

TurnroundDetect::TurnroundDetect() {
    turnround_flag = lock = false;
    turnround_sleep = 20;
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
    int angle_y = 0, area = 0, s = 0;
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
        keep_turnround.push_back(angle_y);
        if (keep_turnround.size() == 15){
            for (auto keep : keep_turnround){
                s = s + keep;
            }
            keep_turnround.pop_front();
        }
        s = s / 15;
        if (s < -40 or s > 40){
            if (!lock) {
                if (t == 0){gettimeofday(&start_time, NULL);}
                t++;
                lock = true;
            }
        } else if (s > -40 and s < 40){
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