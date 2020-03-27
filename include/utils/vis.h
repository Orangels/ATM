//
// Created by 王智慧 on 2020/3/26.
//

#ifndef ATM_VIS_H
#define ATM_VIS_H

#include <vector>
#include <opencv2/opencv.hpp>
#include "structures/structs.h"

using namespace std;

void vis(cv::Mat &img, int frame_id, vector<int> track_id, vector<Box> head_boxes,
         vector<vector<float>> face_angle);

#endif //ATM_VIS_H