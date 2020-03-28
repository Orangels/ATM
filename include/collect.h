//
// Created by 王智慧 on 2020/3/18.
//

#ifndef ATM_COLLECT_H
#define ATM_COLLECT_H

#include <iostream>
#include <sys/time.h>
#include "utils/image_deliver.h"
#include "engine/hf_det.h"
#include "engine/hop_det.h"
#include "engine/hand_det.h"
#include "structures/image.h"
#include "structures/instance_group.h"
#include "utils/track.h"
#include "tasks/solver.h"
#include "Common.h"
#include "detection.h"
#include "face_angle.h"
#include "face_reco.h"

#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include "rtmpHandler.h"

using namespace std;

class Collect {
public:
    Collect();
    ~Collect();
    void run();
    void test();
    void test2();

    ImageDeliver image_deliver;
    Track* head_tracker;
    HFDet hf_det;
    HopDet hop_det;
    HandDet hand_det;
    Image image_class;
    InstanceGroup instance_group;
    Solver function_solver;
    SSD_Detection* ssd_detection;
    Face_Angle* face_angle;
    Face_Reco* face_reco;

    double num_images;
    int frames_skip, sleep_frames_skip, wake_frames_skip, head_track_mistimes;

    //    ls add
    void multithreadTest();

    queue<cv::Mat> mQueue_front;
    queue<cv::Mat> mQueue_top;

    int mQueueArrLen = 2;
    int mQueueLen = 5;
    condition_variable con_front_not_full;
    condition_variable con_front_not_empty;

    condition_variable con_top_not_full;
    condition_variable con_top_not_empty;

    mutex myMutex_front;
    mutex myMutex_top;

    mutex rtmpMutex;
    mutex rtmpMutex_2;
    rtmpHandler ls_handler ;
    rtmpHandler ls_handler_2;
private:
    void ProduceImage(int mode);
    void ConsumeImage(int mode);
};

#endif //ATM_COLLECT_H
