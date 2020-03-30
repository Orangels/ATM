//
// Created by 王智慧 on 2020/3/26.
//

#include "utils/vis.h"

cv::Mat vis(cv::Mat &img, int frame_id, vector<int> track_id, Image image_class, InstanceGroup instance_group, Solver function_solver){
    Box head_box, face_box, hat_box, glass_box, mask_box;
    cv::Point p1, p2, p3, p4, p5;
    Angle angle;
    if (function_solver.group_flag){
        p1.x = 0;
        p1.y = 60;
        p2.x = 100;
        p2.y = p1.y + 30;
        cv::rectangle(img, p1, p2, cv::Scalar(0, 0, 255), -1, 1, 0);
    } else {
        p1.x = 0;
        p1.y = 60;
        p2.x = 100;
        p2.y = p1.y + 30;
        cv::rectangle(img, p1, p2, cv::Scalar(33, 94, 2), -1, 1, 0);
    }
    if (function_solver.hand_flag){
        p1.x = 0;
        p1.y = 120;
        p2.x = 100;
        p2.y = p1.y + 30;
        cv::rectangle(img, p1, p2, cv::Scalar(0, 0, 255), -1, 1, 0);
    } else {
        p1.x = 0;
        p1.y = 120;
        p2.x = 100;
        p2.y = p1.y + 30;
        cv::rectangle(img, p1, p2, cv::Scalar(33, 94, 2), -1, 1, 0);
    }
    if (function_solver.hop_flag){
        p1.x = 0;
        p1.y = 180;
        p2.x = 100;
        p2.y = p1.y + 30;
        cv::rectangle(img, p1, p2, cv::Scalar(0, 0, 255), -1, 1, 0);
    } else {
        p1.x = 0;
        p1.y = 180;
        p2.x = 100;
        p2.y = p1.y + 30;
        cv::rectangle(img, p1, p2, cv::Scalar(33, 94, 2), -1, 1, 0);
    }
    if (function_solver.tround_flag){
        p1.x = 0;
        p1.y = 240;
        p2.x = 100;
        p2.y = p1.y + 30;
        cv::rectangle(img, p1, p2, cv::Scalar(0, 0, 255), -1, 1, 0);
    } else {
        p1.x = 0;
        p1.y = 240;
        p2.x = 100;
        p2.y = p1.y + 30;
        cv::rectangle(img, p1, p2, cv::Scalar(33, 94, 2), -1, 1, 0);
    }
    if (function_solver.entry_flag){
        p1.x = 0;
        p1.y = 300;
        p2.x = 100;
        p2.y = p1.y + 30;
        cv::rectangle(img, p1, p2, cv::Scalar(0, 0, 255), -1, 1, 0);
    } else {
        p1.x = 0;
        p1.y = 300;
        p2.x = 100;
        p2.y = p1.y + 30;
        cv::rectangle(img, p1, p2, cv::Scalar(33, 94, 2), -1, 1, 0);
    }
    cv::putText(img, "Group Detect", cv::Point(10, 75), cv::FONT_HERSHEY_TRIPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
    cv::putText(img, " Hand Warn", cv::Point(10, 135), cv::FONT_HERSHEY_TRIPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
    cv::putText(img, " HOP  Detect", cv::Point(10, 195), cv::FONT_HERSHEY_TRIPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
    cv::putText(img, " Turn Round", cv::Point(10, 255), cv::FONT_HERSHEY_TRIPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
    cv::putText(img, "Multi Entry", cv::Point(10, 315), cv::FONT_HERSHEY_TRIPLEX, 0.4, cv::Scalar(255, 255, 255), 1);

    if (image_class.wake_state) {
        cv::rectangle(img, cv::Point(150, 4), cv::Point(500, 355), cv::Scalar(0, 0, 255), 2, 1, 0);
        for (auto & id : instance_group.track_ids){
            head_box = instance_group.instances[id].head_box;
            p1.x = head_box.x1;
            p1.y = head_box.y1;
            p2.x = head_box.x2;
            p2.y = head_box.y2;
            cv::rectangle(img, p1, p2, cv::Scalar(128, 128, 64), 2, 1, 0);
            if (!instance_group.instances[id].face_box.empty()){
                face_box = instance_group.instances[id].face_box[0];
                angle = instance_group.instances[id].pos_angle;
                p1.x = face_box.x1;
                p1.y = face_box.y1;
                p2.x = face_box.x2;
                p2.y = face_box.y2;
                cv::rectangle(img, p1, p2, cv::Scalar(128, 128, 64), 2, 1, 0);
                p3.x = p1.x + 5;
                p3.y = p1.y + 26;
                p4.x = p1.x + 5;
                p4.y = p1.y + 52;
                p5.x = p1.x + 5;
                p5.y = p1.y + 78;
                cv::putText(img, "Y: " + to_string((int)angle.Y), p3, cv::FONT_HERSHEY_TRIPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
                cv::putText(img, "P: " + to_string((int)angle.P), p4, cv::FONT_HERSHEY_TRIPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
                cv::putText(img, "R: " + to_string((int)angle.R), p5, cv::FONT_HERSHEY_TRIPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
            }
            if (!instance_group.instances[id].hat_box.empty()){
                hat_box = instance_group.instances[id].hat_box[0];
                p1.x = hat_box.x1;
                p1.y = hat_box.y1;
                p2.x = hat_box.x2;
                p2.y = hat_box.y2;
                cv::rectangle(img, p1, p2, cv::Scalar(255, 0, 0), 2, 1, 0);
            }
            if (!instance_group.instances[id].glass_box.empty()){
                glass_box = instance_group.instances[id].glass_box[0];
                p1.x = glass_box.x1;
                p1.y = glass_box.y1;
                p2.x = glass_box.x2;
                p2.y = glass_box.y2;
                cv::rectangle(img, p1, p2, cv::Scalar(255, 0, 0), 2, 1, 0);
            }
            if (!instance_group.instances[id].mask_box.empty()){
                mask_box = instance_group.instances[id].mask_box[0];
                p1.x = mask_box.x1;
                p1.y = mask_box.y1;
                p2.x = mask_box.x2;
                p2.y = mask_box.y2;
                cv::rectangle(img, p1, p2, cv::Scalar(255, 0, 0), 2, 1, 0);
            }
        }
    } else {
        cv::rectangle(img, cv::Point(150, 4), cv::Point(500, 355), cv::Scalar(0, 255, 0), 2, 1, 0);
        for (auto & box : image_class.head_boxes){
            p1.x = box.x1;
            p1.y = box.y1;
            p2.x = box.x2;
            p2.y = box.y2;
            cv::rectangle(img, p1, p2, cv::Scalar(0, 255, 0), 2, 1, 0);
        }
    }
    for (auto & boxes : function_solver.group_heads){
        for(auto & box : boxes){
            p1.x = box.x1;
            p1.y = box.y1;
            p2.x = box.x2;
            p2.y = box.y2;
            cv::rectangle(img, p1, p2, cv::Scalar(0, 255, 255), 2, 1, 0);
        }
    }
    return img;
//    cv::imwrite("../data/results/" + to_string(frame_id) + ".jpg", img);
}