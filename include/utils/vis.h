//
// Created by 王智慧 on 2020/3/26.
//

#ifndef ATM_VIS_H
#define ATM_VIS_H

#include <vector>
#include <opencv2/opencv.hpp>
#include "structures/image.h"
#include "structures/instance_group.h"
#include "structures/instance.h"
#include "structures/structs.h"
#include "tasks/solver.h"

using namespace std;

cv::Mat vis(cv::Mat &img, int frame_id, vector<int> track_id, Image image_class, InstanceGroup instance_group, Solver function_solver);

#endif //ATM_VIS_H