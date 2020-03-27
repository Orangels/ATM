//
// Created by 王智慧 on 2020/3/26.
//

#include "utils/vis.h"

void vis(cv::Mat &img, int frame_id, vector<int> track_id, vector<Box> head_boxes,
         vector<vector<float>> face_angle){
    for (int i = 0; i < head_boxes.size(); i++) {
        cv::Point p1, p2, p3, p4, p5;
        p1.x = head_boxes[i].x1;
        p1.y = head_boxes[i].y1;
        p2.x = head_boxes[i].x2;
        p2.y = head_boxes[i].y2;
        p3.x = p1.x + 5;
        p3.y = p1.y + 26;
        p4.x = p1.x + 5;
        p4.y = p1.y + 52;
        p5.x = p1.x + 5;
        p5.y = p1.y + 78;
        cv::rectangle(img, p1, p2, cv::Scalar(0, 0, 255), 3, 4, 0);
        cv::putText(img, "Y: " + to_string(face_angle[i][0]), p3, cv::FONT_HERSHEY_TRIPLEX, 0.9, cv::Scalar(255, 255, 255), 2);
        cv::putText(img, "P: " + to_string(face_angle[i][1]), p4, cv::FONT_HERSHEY_TRIPLEX, 0.9, cv::Scalar(255, 255, 255), 2);
        cv::putText(img, "R: " + to_string(face_angle[i][2]), p5, cv::FONT_HERSHEY_TRIPLEX, 0.9, cv::Scalar(255, 255, 255), 2);
    }
    cv::imwrite("../data/results/" + to_string(frame_id) + ".png", img);
}