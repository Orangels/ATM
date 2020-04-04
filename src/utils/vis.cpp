//
// Created by 王智慧 on 2020/3/26.
//

#include "utils/vis.h"

using namespace cv;

void vis(Mat &img, int frame_id, vector<int> track_id, Image image_class, InstanceGroup instance_group, Solver function_solver){
    int face_id;
    Box head_box, face_box, hat_box, glass_box, mask_box;
    Point p1, p2, p3, p4, p5;
    Angle angle;
    if (function_solver.group_flag){
        rectangle(img, Point(0, 60), Point(100, 90), Scalar(0, 0, 255), -1, 1, 0);
    } else {
        rectangle(img, Point(0, 60), Point(100, 90), Scalar(33, 94, 2), -1, 1, 0);
    }
    if (function_solver.hand_flag){
        rectangle(img, Point(0, 120), Point(100, 150), Scalar(0, 0, 255), -1, 1, 0);
    } else {
        rectangle(img, Point(0, 120), Point(100, 150), Scalar(33, 94, 2), -1, 1, 0);
    }
    if (function_solver.hop_flag){
        rectangle(img, Point(0, 180), Point(100, 210), Scalar(0, 0, 255), -1, 1, 0);
    } else {
        rectangle(img, Point(0, 180), Point(100, 210), Scalar(33, 94, 2), -1, 1, 0);
    }
    if (function_solver.tround_flag){
        rectangle(img, Point(0, 240), Point(100, 270), Scalar(0, 0, 255), -1, 1, 0);
    } else {
        rectangle(img, Point(0, 240), Point(100, 270), Scalar(33, 94, 2), -1, 1, 0);
    }
    if (function_solver.entry_flag){
        rectangle(img, Point(0, 300), Point(100, 330), Scalar(0, 0, 255), -1, 1, 0);
    } else {
        rectangle(img, Point(0, 300), Point(100, 330), Scalar(33, 94, 2), -1, 1, 0);
    }
    putText(img, "Group Detect", Point(10, 75), FONT_HERSHEY_TRIPLEX, 0.4, Scalar(255, 255, 255), 1);
    putText(img, " Hand Warn", Point(10, 135), FONT_HERSHEY_TRIPLEX, 0.4, Scalar(255, 255, 255), 1);
    putText(img, " HOP  Detect", Point(10, 195), FONT_HERSHEY_TRIPLEX, 0.4, Scalar(255, 255, 255), 1);
    putText(img, " Turn Round", Point(10, 255), FONT_HERSHEY_TRIPLEX, 0.4, Scalar(255, 255, 255), 1);
    putText(img, "Multi Entry", Point(10, 315), FONT_HERSHEY_TRIPLEX, 0.4, Scalar(255, 255, 255), 1);

    if (image_class.wake_state) {
        rectangle(img, Point(150, 4), Point(500, 355), Scalar(0, 0, 255), 2, 1, 0);
        for (auto & id : instance_group.track_ids){
            head_box = instance_group.instances[id].head_box;
            p1.x = head_box.x1;
            p1.y = head_box.y1;
            p2.x = head_box.x2;
            p2.y = head_box.y2;
            rectangle(img, p1, p2, Scalar(128, 128, 64), 2, 1, 0);
            if (!instance_group.instances[id].face_box.empty()){
                face_box = instance_group.instances[id].face_box[0];
                angle = instance_group.instances[id].pos_angle;
                face_id = instance_group.instances[id].face_id;
                p1.x = face_box.x1;
                p1.y = face_box.y1;
                p2.x = face_box.x2;
                p2.y = face_box.y2;
                rectangle(img, p1, p2, Scalar(128, 128, 64), 2, 1, 0);
                p3.x = p1.x + 5;
                p3.y = p1.y + 26;
                p4.x = p1.x + 5;
                p4.y = p1.y + 52;
                p5.x = p1.x + 5;
                p5.y = p1.y + 78;
                putText(img, "Y: " + to_string((int)angle.Y), p3, FONT_HERSHEY_TRIPLEX, 0.4, Scalar(255, 255, 255), 1);
                putText(img, "P: " + to_string((int)angle.P), p4, FONT_HERSHEY_TRIPLEX, 0.4, Scalar(255, 255, 255), 1);
                putText(img, "R: " + to_string((int)angle.R), p5, FONT_HERSHEY_TRIPLEX, 0.4, Scalar(255, 255, 255), 1);
                putText(img, "ID: " + to_string(face_id), Point(p1.x + 15, p1.y), FONT_HERSHEY_TRIPLEX, 0.8, Scalar(255, 255, 255), 1);
            }
            if (!instance_group.instances[id].hat_box.empty()){
                hat_box = instance_group.instances[id].hat_box[0];
                p1.x = hat_box.x1;
                p1.y = hat_box.y1;
                p2.x = hat_box.x2;
                p2.y = hat_box.y2;
                rectangle(img, p1, p2, Scalar(128, 0, 192), 3, 1, 0);
            }
            if (!instance_group.instances[id].glass_box.empty()){
                glass_box = instance_group.instances[id].glass_box[0];
                p1.x = glass_box.x1;
                p1.y = glass_box.y1;
                p2.x = glass_box.x2;
                p2.y = glass_box.y2;
                rectangle(img, p1, p2, Scalar(128, 0, 192), 3, 1, 0);
            }
            if (!instance_group.instances[id].mask_box.empty()){
                mask_box = instance_group.instances[id].mask_box[0];
                p1.x = mask_box.x1;
                p1.y = mask_box.y1;
                p2.x = mask_box.x2;
                p2.y = mask_box.y2;
                rectangle(img, p1, p2, Scalar(128, 0, 192), 3, 1, 0);
            }
        }
    } else {
        rectangle(img, Point(150, 4), Point(500, 355), Scalar(0, 255, 0), 2, 1, 0);
        for (auto & box : image_class.head_boxes){
            p1.x = box.x1;
            p1.y = box.y1;
            p2.x = box.x2;
            p2.y = box.y2;
            rectangle(img, p1, p2, Scalar(0, 255, 0), 2, 1, 0);
        }
    }
    for (auto & boxes : function_solver.group_heads){
        for(auto & box : boxes){
            p1.x = box.x1;
            p1.y = box.y1;
            p2.x = box.x2;
            p2.y = box.y2;
            rectangle(img, p1, p2, Scalar(0, 255, 255), 2, 1, 0);
        }
    }
    imwrite("../data/results/" + to_string(frame_id) + ".jpg", img);
}

cv::Mat get_vis(Mat &img, int frame_id, vector<int> track_id, Image image_class, InstanceGroup instance_group, Solver function_solver){
    cv:Mat result_img = img.clone();
    int face_id;
    Box head_box, face_box, hat_box, glass_box, mask_box;
    Point p1, p2, p3, p4, p5;
    Angle angle;
    if (function_solver.group_flag){
        rectangle(result_img, Point(0, 60), Point(100, 90), Scalar(0, 0, 255), -1, 1, 0);
    } else {
        rectangle(result_img, Point(0, 60), Point(100, 90), Scalar(33, 94, 2), -1, 1, 0);
    }
    if (function_solver.hand_flag){
//        rectangle(result_img, Point(0, 120), Point(100, 150), Scalar(0, 0, 255), -1, 1, 0);
        rectangle(result_img, Point(0, 120), Point(100, 150), Scalar(33, 94, 2), -1, 1, 0);
    } else {
        rectangle(result_img, Point(0, 120), Point(100, 150), Scalar(33, 94, 2), -1, 1, 0);
    }
    if (function_solver.hop_flag){
        rectangle(result_img, Point(0, 180), Point(100, 210), Scalar(0, 0, 255), -1, 1, 0);
    } else {
        rectangle(result_img, Point(0, 180), Point(100, 210), Scalar(33, 94, 2), -1, 1, 0);
    }
    if (function_solver.tround_flag){
        rectangle(result_img, Point(0, 240), Point(100, 270), Scalar(0, 0, 255), -1, 1, 0);
    } else {
        rectangle(result_img, Point(0, 240), Point(100, 270), Scalar(33, 94, 2), -1, 1, 0);
    }
    if (function_solver.entry_flag){
        rectangle(result_img, Point(0, 300), Point(100, 330), Scalar(0, 0, 255), -1, 1, 0);
    } else {
        rectangle(result_img, Point(0, 300), Point(100, 330), Scalar(33, 94, 2), -1, 1, 0);
    }
    putText(result_img, "Group Detect", Point(10, 75), FONT_HERSHEY_TRIPLEX, 0.4, Scalar(255, 255, 255), 1);
    putText(result_img, " Hand Warn", Point(10, 135), FONT_HERSHEY_TRIPLEX, 0.4, Scalar(255, 255, 255), 1);
    putText(result_img, " HOP  Detect", Point(10, 195), FONT_HERSHEY_TRIPLEX, 0.4, Scalar(255, 255, 255), 1);
    putText(result_img, " Turn Round", Point(10, 255), FONT_HERSHEY_TRIPLEX, 0.4, Scalar(255, 255, 255), 1);
    putText(result_img, "Multi Entry", Point(10, 315), FONT_HERSHEY_TRIPLEX, 0.4, Scalar(255, 255, 255), 1);

    if (image_class.wake_state) {
        rectangle(result_img, Point(150, 4), Point(500, 355), Scalar(0, 0, 255), 2, 1, 0);
        for (auto & id : instance_group.track_ids){
            head_box = instance_group.instances[id].head_box;
            p1.x = head_box.x1;
            p1.y = head_box.y1;
            p2.x = head_box.x2;
            p2.y = head_box.y2;
            rectangle(result_img, p1, p2, Scalar(128, 128, 64), 2, 1, 0);
            if (!instance_group.instances[id].face_box.empty()){
                face_box = instance_group.instances[id].face_box[0];
                angle = instance_group.instances[id].pos_angle;
                face_id = instance_group.instances[id].face_id;
                p1.x = face_box.x1;
                p1.y = face_box.y1;
                p2.x = face_box.x2;
                p2.y = face_box.y2;
                rectangle(result_img, p1, p2, Scalar(128, 128, 64), 2, 1, 0);
                p3.x = p1.x + 5;
                p3.y = p1.y + 26;
                p4.x = p1.x + 5;
                p4.y = p1.y + 52;
                p5.x = p1.x + 5;
                p5.y = p1.y + 78;
                putText(result_img, "Y: " + to_string((int)angle.Y), p3, FONT_HERSHEY_TRIPLEX, 0.4, Scalar(255, 255, 255), 1);
                putText(result_img, "P: " + to_string((int)angle.P), p4, FONT_HERSHEY_TRIPLEX, 0.4, Scalar(255, 255, 255), 1);
                putText(result_img, "R: " + to_string((int)angle.R), p5, FONT_HERSHEY_TRIPLEX, 0.4, Scalar(255, 255, 255), 1);
                putText(result_img, "ID: " + to_string(face_id), Point(p1.x + 15, p1.y), FONT_HERSHEY_TRIPLEX, 0.8, Scalar(255, 255, 255), 1);
            }
            if (!instance_group.instances[id].hat_box.empty()){
                hat_box = instance_group.instances[id].hat_box[0];
                p1.x = hat_box.x1;
                p1.y = hat_box.y1;
                p2.x = hat_box.x2;
                p2.y = hat_box.y2;
                rectangle(result_img, p1, p2, Scalar(128, 0, 192), 3, 1, 0);
            }
            if (!instance_group.instances[id].glass_box.empty()){
                glass_box = instance_group.instances[id].glass_box[0];
                p1.x = glass_box.x1;
                p1.y = glass_box.y1;
                p2.x = glass_box.x2;
                p2.y = glass_box.y2;
                rectangle(result_img, p1, p2, Scalar(128, 0, 192), 3, 1, 0);
            }
            if (!instance_group.instances[id].mask_box.empty()){
                mask_box = instance_group.instances[id].mask_box[0];
                p1.x = mask_box.x1;
                p1.y = mask_box.y1;
                p2.x = mask_box.x2;
                p2.y = mask_box.y2;
                rectangle(result_img, p1, p2, Scalar(128, 0, 192), 3, 1, 0);
            }
        }
    } else {
        rectangle(result_img, Point(150, 4), Point(500, 355), Scalar(0, 255, 0), 2, 1, 0);
        for (auto & box : image_class.head_boxes){
            p1.x = box.x1;
            p1.y = box.y1;
            p2.x = box.x2;
            p2.y = box.y2;
            rectangle(result_img, p1, p2, Scalar(0, 255, 0), 2, 1, 0);
        }
    }
    for (auto & boxes : function_solver.group_heads){
        for(auto & box : boxes){
            p1.x = box.x1;
            p1.y = box.y1;
            p2.x = box.x2;
            p2.y = box.y2;
            rectangle(result_img, p1, p2, Scalar(0, 255, 255), 2, 1, 0);
        }
    }

    for (auto & box : function_solver.entry_face){
        p1.x = box.x1;
        p1.y = box.y1;
        p2.x = box.x2;
        p2.y = box.y2;
        rectangle(result_img, p1, p2, Scalar(128, 64, 0), 2, 1, 0);
    }

    return result_img;
//    imwrite("../data/results/" + to_string(frame_id) + ".jpg", img);

}