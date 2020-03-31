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
#include <libgearman/gearman.h>

using namespace std;

enum WARING_TYPE
{
    GROUP = 0,
    HAND = 1,
    TROUND = 2,
    HOP = 3,
    ENTRY = 4
};

typedef struct
{
    bool group_flag;
    bool hand_flag;
    bool tround_flag;
    bool hop_flag;
    bool entry_flag;
} WATING_FLAG;

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
    int mQueueArrLen = 2;
    int mQueueLen = 5;

    queue<cv::Mat> mQueue_front;
    queue<cv::Mat> mQueue_top;
    queue<cv::Mat> mQueue_rtmp_front;
    queue<cv::Mat> mQueue_rtmp_top;
    queue<cv::Mat> mQueue_waring_front;
    queue<cv::Mat> mQueue_waring_top;


    condition_variable con_front_not_full;
    condition_variable con_front_not_empty;

    condition_variable con_top_not_full;
    condition_variable con_top_not_empty;

    condition_variable con_rtmp_front;
    condition_variable con_rtmp_top;

    condition_variable con_waring_front;
    condition_variable con_waring_top;

    mutex myMutex_front;
    mutex myMutex_top;
    mutex myMutex_rtmp_front;
    mutex myMutex_rtmp_top;
    mutex myMutex_waring_front;
    mutex myMutex_waring_top;

    mutex rtmpMutex_front;
    mutex rtmpMutex_top;

    rtmpHandler ls_handler ;
    rtmpHandler ls_handler_2;

    gearman_return_t gearRet;
    gearman_client_st* gearClient;

    cv::Mat rtmp_front_img;
    cv::Mat rtmp_top_img;

    WATING_FLAG warningFlag = {false,false,false,false,false};

    void hf_thread();
    void hop_thread();
    void hand_thread();
private:
    void ProduceImage(int mode);
    void ConsumeImage(int mode);
    void ConsumeRTMPImage(int mode);
    void ConsumeWaringImage(int mode);
};

#endif //ATM_COLLECT_H
