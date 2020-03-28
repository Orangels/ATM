//
// Created by 王智慧 on 2020/3/26.
//

#include "utils/vis.h"

void vis(cv::Mat &img, int frame_id, vector<int> track_id, InstanceGroup instance_group){
    Box head_box, face_box;
    cv::Point p1, p2, p3, p4, p5;
    Angle angle;
    for (auto & id : instance_group.track_ids){
        head_box = instance_group.instances[id].head_box;
        p1.x = head_box.x1;
        p1.y = head_box.y1;
        p2.x = head_box.x2;
        p2.y = head_box.y2;
        cv::rectangle(img, p1, p2, cv::Scalar(0, 0, 255), 3, 4, 0);
        if (!instance_group.instances[id].face_box.empty()){
            face_box = instance_group.instances[id].face_box[0];
            angle = instance_group.instances[id].pos_angle;
            p1.x = face_box.x1;
            p1.y = face_box.y1;
            p2.x = face_box.x2;
            p2.y = face_box.y2;
            cv::rectangle(img, p1, p2, cv::Scalar(255, 0, 0), 3, 4, 0);
            p3.x = p1.x + 5;
            p3.y = p1.y + 26;
            p4.x = p1.x + 5;
            p4.y = p1.y + 52;
            p5.x = p1.x + 5;
            p5.y = p1.y + 78;
            cv::putText(img, "Y: " + to_string(angle.Y), p3, cv::FONT_HERSHEY_TRIPLEX, 0.9, cv::Scalar(255, 255, 255), 2);
            cv::putText(img, "P: " + to_string(angle.P), p4, cv::FONT_HERSHEY_TRIPLEX, 0.9, cv::Scalar(255, 255, 255), 2);
            cv::putText(img, "R: " + to_string(angle.R), p5, cv::FONT_HERSHEY_TRIPLEX, 0.9, cv::Scalar(255, 255, 255), 2);
        }
    }
    cv::imwrite("../data/results/" + to_string(frame_id) + ".png", img);
}