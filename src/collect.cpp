//
// Created by 王智慧 on 2020/3/18.
//

#include "collect.h"
#include "config.h"
#include "utils/misc.h"
#include "utils/vis.h"

#include "ls_vis.h"
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/prettywriter.h"
#include <unistd.h>

int64_t getCurrentTime()
{
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

bool updateWaringFlagHand(WATING_FLAG & warning, Solver function_solver);
bool updateWaringFlag(WATING_FLAG & warning, Solver function_solver);

Collect::Collect() {
    num_images = image_deliver.num_images;
    Cconfig labels = Cconfig("../cfg/process.ini");
    CConfiger* m_pconfiger = CConfiger::getOrCreateConfiger("../cfg/configer.ini");
    sleep_frames_skip = stoi(labels["SLEEP_FRAMES_SKIP"]);
    wake_frames_skip = stoi(labels["WAKE_FRAMES_SKIP"]);
    head_track_mistimes = stoi(labels["HEAD_TRACK_MISTIMES"]);
    head_tracker = new Track(head_track_mistimes, stoi(labels["IMAGE_W"]), stoi(labels["IMAGE_H"]));
    ssd_detection = new SSD_Detection;

    //    ls add
    mQueueLen = m_pconfiger->readValue<int>("mQueueLen");
    frames_skip = sleep_frames_skip;
    int fps = m_pconfiger->readValue<int>("fps");
    int out_w = m_pconfiger->readValue<int>("out_w");
    int out_h = m_pconfiger->readValue<int>("out_h");
    std::string rtmpPath = m_pconfiger->readValue("rtmpPath");
    cout << rtmpPath << endl;
    ls_handler = rtmpHandler("",rtmpPath,out_w,out_h,fps);


    int fps_2 = m_pconfiger->readValue<int>("fps_2");
    int out_w_2 = m_pconfiger->readValue<int>("out_w_2");
    int out_h_2 = m_pconfiger->readValue<int>("out_h_2");
    std::string rtmpPath_2 = m_pconfiger->readValue("rtmpPath_2");
    std::string img_root_path = m_pconfiger->readValue("waring_img_path");
    ls_handler_2 = rtmpHandler("",rtmpPath_2,out_w_2,out_h_2,fps_2);

    //    ls add gearman client init
    char* gearSvrHost=(char*)"127.0.0.1", *gearSvrPort=(char*)"4730";

    gearClient = gearman_client_create(NULL);
    gearman_client_set_options(gearClient, GEARMAN_CLIENT_FREE_TASKS);
    gearman_client_set_timeout(gearClient, 15000);

    gearRet = gearman_client_add_server(gearClient, gearSvrHost, atoi(gearSvrPort));
    if (gearman_failed(gearRet))
    {
        cout << "gearman client init failed" << endl;
    }
}

Collect::~Collect() = default;

void Collect::run() {
    struct timeval tp1;
    struct timeval tp2;
    struct timeval t_im, tra_1, hf_1, hf_2, hop_1, hand_1, hand_2, ang_1, ang_2, reco, vis_1, vis_2;
    for (int i=1000; i<num_images+1000; i++){
//        instance_group.check_state();
        instance_group.clear();
        image_class.frame_id = i;
        gettimeofday(&tp1, NULL);
        image_deliver.get_frame();
        gettimeofday(&t_im, NULL);

        if (image_class.wake_state){
            frames_skip = wake_frames_skip;
        } else{
            frames_skip = sleep_frames_skip;
        }

        if (i % frames_skip == 0){
            gettimeofday(&hf_1, NULL);
            hf_det.inference(image_deliver.front_img, ssd_detection);
            gettimeofday(&hf_2, NULL);
            image_class.update_hf(hf_det.head_boxes, hf_det.face_boxes);
//            if(!image_class.keep_head.empty()){
//                for (auto &keep : image_class.keep_head){
//                    cout<<keep<<" ";
//                }
//                cout<<endl;
//            }
            if (image_class.wake_state) {
                gettimeofday(&tra_1, NULL);
                head_tracker->run(image_class.head_boxes);
                gettimeofday(&hop_1, NULL);
                hop_det.inference(image_deliver.front_img, ssd_detection);
                gettimeofday(&hand_1, NULL);
                hand_det.inference(image_deliver.top_img, ssd_detection);
                gettimeofday(&hand_2, NULL);
                image_class.update_hand(hand_det.hand_boxes);
                instance_group.update(i, head_tracker->tracking_result, image_class.head_boxes,
                                      image_class.face_boxes, head_tracker->delete_tracking_id);
                instance_group.add_hop_box( hop_det.hat_boxes, hop_det.glass_boxes, hop_det.mask_boxes);
                vector<vector<float>> face_boxes, reco_boxes, face_angles, face_fea;
                instance_group.get_face_box(face_boxes);
                gettimeofday(&ang_1, NULL);
                ssd_detection->get_angles(face_boxes, face_angles);
                gettimeofday(&ang_2, NULL);
                instance_group.update_face_angle(face_angles, reco_boxes);
                ssd_detection->get_features(reco_boxes, face_fea);
                vector<int> face_ids;
                face_lib.get_identity(face_fea, face_ids);
                instance_group.update_face_id(face_ids);
                gettimeofday(&reco, NULL);
                cout<<" hf:  "<< 1000 * (hf_2.tv_sec-hf_1.tv_sec) + (hf_2.tv_usec-hf_1.tv_usec)/1000<<endl;
                cout<<" tracker:  "<< 1000 * (hop_1.tv_sec-tra_1.tv_sec) + (hop_1.tv_usec-tra_1.tv_usec)/1000<<endl;
                cout<<" hop:  "<< 1000 * (hand_1.tv_sec-hop_1.tv_sec) + (hand_1.tv_usec-hop_1.tv_usec)/1000<<endl;
                cout<<" hand:  "<< 1000 * (hand_2.tv_sec-hand_1.tv_sec) + (hand_2.tv_usec-hand_1.tv_usec)/1000<<endl;
                cout<<" angle:  "<< 1000 * (ang_2.tv_sec-ang_1.tv_sec) + (ang_2.tv_usec-ang_1.tv_usec)/1000<<endl;
                cout<<" reco:  "<< 1000 * (reco.tv_sec-ang_2.tv_sec) + (reco.tv_usec-ang_2.tv_usec)/1000<<endl;
            } else{
                head_tracker->run(image_class.head_boxes);
                instance_group.update_track(i, head_tracker->delete_tracking_id);
            }
            function_solver.update(image_class, instance_group);
            gettimeofday(&vis_1, NULL);
            vis(image_deliver.front_img, i, head_tracker->tracking_result, image_class, instance_group, function_solver);
            gettimeofday(&vis_2, NULL);
            cout<<" image:  "<< 1000 * (t_im.tv_sec-tp1.tv_sec) + (t_im.tv_usec-tp1.tv_usec)/1000<<endl;
            cout<<" vis:  "<< 1000 * (vis_2.tv_sec-vis_1.tv_sec) + (vis_2.tv_usec-vis_1.tv_usec)/1000<<endl;
        }
        gettimeofday(&tp2, NULL);
        cout<<i<<" :  "<< 1000 * (tp2.tv_sec-tp1.tv_sec) + (tp2.tv_usec-tp1.tv_usec)/1000<<endl;
    }
}

void Collect::ConsumeWaringImage(int mode){
    cv::Mat img;
    int num = 0;
    std::string img_root_path = CConfiger::getOrCreateConfiger()->readValue("waring_img_path");
    mutex *lock;
    queue<cv::Mat> *queue;
    condition_variable *con_v_wait;
    switch (mode) {
        case 0:
            lock = &myMutex_waring_front;
            queue = &mQueue_waring_front;
            con_v_wait = &con_waring_front;
            break;
        case 1:
            lock = &myMutex_waring_top;
            queue = &mQueue_waring_top;
            con_v_wait = &con_waring_top;
            break;
        default:
            lock = &myMutex_waring_front;
            queue = &mQueue_waring_front;
            con_v_wait = &con_waring_front;
            break;

    }
    while (true){
        std::unique_lock<std::mutex> guard(*lock);
        while(queue->empty()) {
            std::cout << "Consumer WARING " << mode << " -- " << num <<" is waiting for items...\n";
            con_v_wait->wait(guard);
        }
        int64_t start_read = getCurrentTime();
        img = queue->front();
        queue->pop();
        guard.unlock();

        rapidjson::StringBuffer buf;
        rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buf);

        if (mode == 0){
            enum WARING_TYPE waringType;
            waringType = GROUP;

            string timestamp = to_string(getCurrentTime());
            string img_full_path = img_root_path + timestamp + "_" + to_string(waringType) + ".jpg";
            cv::imwrite(img_full_path, img);
            string img_path = timestamp + "_" + to_string(waringType) + ".jpg";

            writer.StartObject();

            writer.Key("params");
            writer.StartArray();
            if (warningFlag.group_flag){
                writer.StartObject();
                writer.Key("path"); writer.String(img_path.c_str());
                writer.Key("mode");writer.Int(waringType);
                writer.EndObject();
            }

            if (warningFlag.tround_flag){
                waringType = TROUND;
                writer.StartObject();
                writer.Key("path"); writer.String(img_path.c_str());
                writer.Key("mode");writer.Int(waringType);
                writer.EndObject();
            }

            if (warningFlag.hop_flag){
                waringType = HOP;
                writer.StartObject();
                writer.Key("path"); writer.String(img_path.c_str());
                writer.Key("mode");writer.Int(waringType);
                writer.EndObject();
            }

            if (warningFlag.entry_flag){
                waringType = ENTRY;
                writer.StartObject();
                writer.Key("path"); writer.String(img_path.c_str());
                writer.Key("mode");writer.Int(waringType);
                writer.EndObject();
            }

            writer.EndArray();

            writer.EndObject();
        }
        else{
            enum WARING_TYPE waringType;
            waringType = HAND;

            string timestamp = to_string(getCurrentTime());
            string img_full_path = img_root_path + timestamp + "_" + to_string(waringType) + ".jpg";
            cv::imwrite(img_full_path, img);

            writer.StartObject();

            writer.Key("params");
            writer.StartArray();

            string img_path = timestamp + "_" + to_string(waringType) + ".jpg";
            writer.StartObject();
            writer.Key("path"); writer.String(img_path.c_str());
            writer.Key("mode");writer.Int(waringType);
            writer.EndObject();

            writer.EndArray();

            writer.EndObject();

        }

        const char* json_content = buf.GetString();
        string json_cs = json_content;
//        TODO lock and unlock
        gearRet = gearman_client_do_background(gearClient,
                                               "det_car",
                                               NULL,
                                               json_cs.c_str(),
                                               (size_t)strlen(json_cs.c_str()),
                                               NULL);
        if (gearRet == GEARMAN_SUCCESS)
        {
            fprintf(stdout, "Work success!\n");
        }
        else if (gearRet == GEARMAN_WORK_FAIL)
        {
            fprintf(stderr, "Work failed\n");
        }
        else if (gearRet == GEARMAN_TIMEOUT)
        {
            fprintf(stderr, "Work timeout\n");
        }
        else
        {
            fprintf(stderr, "%d,%s\n", gearman_client_errno(gearClient), gearman_client_error(gearClient));
        }
    }

}

void Collect::ConsumeRTMPImage(int mode){
    cv::Mat img;
    int num = 0;
    std::string img_root_path = CConfiger::getOrCreateConfiger()->readValue("waring_img_path");


    mutex *lock;
    queue<cv::Mat> *queue;
    condition_variable *con_v_wait;
    mutex* rtmpLock;
    rtmpHandler* rtmpHandler;

    switch (mode) {
        case 0:
            lock = &myMutex_rtmp_front;
            queue = &mQueue_rtmp_front;
            con_v_wait = &con_rtmp_front;
            rtmpLock = &rtmpMutex_front;
            rtmpHandler = &ls_handler;
            break;
        case 1:
            lock = &myMutex_rtmp_top;
            queue = &mQueue_rtmp_top;
            con_v_wait = &con_rtmp_top;
            rtmpLock = &rtmpMutex_top;
            rtmpHandler = &ls_handler_2;
            break;
        default:
            lock = &myMutex_rtmp_front;
            queue = &mQueue_rtmp_front;
            con_v_wait = &con_rtmp_front;
            rtmpLock = &rtmpMutex_front;
            rtmpHandler = &ls_handler;
            break;
    }

    while (true) {
        std::unique_lock<std::mutex> guard(*lock);
        while(queue->empty()) {
            std::cout << "Consumer RTMP " << mode << " -- " << num <<" is waiting for items...\n";
            con_v_wait->wait(guard);
        }
        int64_t start_read = getCurrentTime();
        img = queue->front().clone();
        queue->pop();
        guard.unlock();
        rtmpLock->lock();
        rtmpHandler->pushRTMP(img);
        rtmpLock->unlock();
        num++;
    }
}

void Collect::ProduceImage(int mode){
    Cconfig labels = Cconfig("../cfg/process.ini");
    string path_0 = labels["video_path_0"];
    string path_1 = labels["video_path_1"];
    cout << "path_0 " << path_0 << endl;
    cout << "path_1 " << path_1 << endl;
    cv::VideoCapture cam;
    cv::Mat frame;
    string path = "";
    mutex *lock;
    queue<cv::Mat> *queue;
    condition_variable *con_v_wait, *con_v_notification;

    switch (mode){
        case 0:
//            path = "rtspsrc location=rtsp://admin:sx123456@192.168.88.37:554/h264/ch2/sub/av_stream latency=0 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! videoconvert ! appsink";
//            path = "filesrc location=/srv/ATM_ls/ATM/data/front.mp4 ! qtdemux ! queue ! h264parse !  omxh264dec  ! nvvidconv ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! videoconvert  ! appsink";
            path = path_0;
            lock = &myMutex_front;
            queue = &mQueue_front;
            con_v_wait = &con_front_not_full;
            con_v_notification = &con_front_not_empty;
            break;
        case 1:
//            path = "rtspsrc location=rtsp://admin:sx123456@192.168.88.37:554/h264/ch2/sub/av_stream latency=0 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! videoconvert ! appsink";
//            path = "filesrc location=/srv/ATM_ls/ATM/data/top.mp4 ! qtdemux ! queue ! h264parse !  omxh264dec  ! nvvidconv ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! videoconvert  ! appsink";
            path = path_1;
            lock = &myMutex_top;
            queue = &mQueue_top;
            con_v_wait = &con_top_not_full;
            con_v_notification = &con_top_not_empty;

            break;
        default:
            path = "rtspsrc location=rtsp://admin:sx123456@192.168.88.37:554/h264/ch2/sub/av_stream latency=0 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! videoconvert ! appsink";
            lock = &myMutex_front;
            queue = &mQueue_front;
            con_v_wait = &con_front_not_full;
            con_v_notification = &con_front_not_empty;

            break;
    }

    cam.open(path);

    if (!cam.isOpened())
    {
        cout << "cam open failed!" << endl;
        return;
    }
    int num = 0;
    int64_t start_all = getCurrentTime();
    int clo_num = num_images;
//    for (int i=0; i<clo_num; i++) {

    for (int i=0; ; i++) {
        //        TODO 跳帧
        int64_t start = getCurrentTime();
        cam.read(frame);

        if (mode == 0){
            if (image_class.wake_state){
                frames_skip = wake_frames_skip;
            } else{
                frames_skip = sleep_frames_skip;
            }
        }
        if (i % frames_skip != 0 ){
            if (mode == 0){

                if (i == 0){
                    rtmp_front_img = frame.clone();
                }

                rtmpMutex_front.lock();
                ls_handler.pushRTMP(rtmp_front_img);
                rtmpMutex_front.unlock();
            }

            if (mode == 1){

                if (i == 0){
                    rtmp_top_img = frame.clone();
                }

                rtmpMutex_top.lock();
                ls_handler_2.pushRTMP(rtmp_top_img);
                rtmpMutex_top.unlock();
            }
            continue;
        }


        num ++;
        if (mode == 0){
            image_class.frame_id = i;
        }
        std::unique_lock<std::mutex> guard(*lock);
        while(queue->size() >=  mQueueLen) {
            std::cout << "Produce " << mode <<" is waiting for items...\n";
            con_v_wait->wait(guard);
        }

        queue->push(frame);
        con_v_notification->notify_all();
        guard.unlock();
        int64_t end = getCurrentTime();
        cout << "write every time  : " << mode << "  " << i << " -- " << (end - start)  << endl;
        cout << "write process time  : " << mode << "  " << num << " -- " << (end - start)  << endl;


    }
    int64_t end_all = getCurrentTime();
    cout << "write img over -- " << (end_all - start_all) / clo_num / 1.0<< endl;
//    cout << "timestamp -- " << getCurrentTime() << endl;
    cout << "start timestamp --- " << start_all << endl;
}

void Collect::ConsumeImage(int mode){
//    cv::Mat img;
//    cv::Mat rtmp_frame;
    int num = 1;
    int front_num = 0;
    float front_sum = 0.;

    mutex *lock;
    queue<cv::Mat> *queue;
    condition_variable *con_v_wait, *con_v_notification;

    switch (mode){
        case 0:
            lock = &myMutex_front;
            queue = &mQueue_front;
            con_v_wait = &con_front_not_empty;
            con_v_notification = &con_front_not_full;
            break;
        case 1:
            lock = &myMutex_top;
            queue = &mQueue_top;
            con_v_wait = &con_top_not_empty;
            con_v_notification = &con_top_not_full;
            break;
        default:
            lock = &myMutex_front;
            queue = &mQueue_front;
            con_v_wait = &con_front_not_empty;
            con_v_notification = &con_front_not_full;
            break;
    }

    while (true){
        std::unique_lock<std::mutex> guard(*lock);
        while(queue->empty()) {
            std::cout << "Consumer " << mode <<" is waiting for items...\n";
            con_v_wait->wait(guard);
        }
        int64_t start_read = getCurrentTime();
        cv::Mat img = queue->front();

        queue->pop();
        con_v_notification->notify_all();
        guard.unlock();
        num ++;

        if (mode == 0){

            instance_group.clear();

            int64_t start_front = getCurrentTime();
            std::thread hf_thread([&]()
                                  {
                                      int64_t start = getCurrentTime();
                                      hf_det.inference(img, ssd_detection);
                                      int64_t end = getCurrentTime();
                                      image_class.update_hf(hf_det.head_boxes, hf_det.face_boxes);
////                                      int64_t end = getCurrentTime();
                                      cout << "hf time  : " << num << " -- " << (end - start) << endl;
//                                      cout << "timestamp mode 0 -- " << getCurrentTime() << endl;
                                  });

            if (image_class.wake_state) {
                front_num++;
                std::thread track_points_thread([&](){
                    hf_thread.join();
                    head_tracker->run(image_class.head_boxes);
                    instance_group.update(num, head_tracker->tracking_result, image_class.head_boxes,
                                          image_class.face_boxes, head_tracker->delete_tracking_id);
//                    vector <vector<float>> boxes, face_a, face_b;
                    vector<vector<float>> face_boxes, reco_boxes, face_angles, face_fea;
                    instance_group.get_face_box(face_boxes);
                    int64_t kp_start = getCurrentTime();
                    ssd_detection->get_angles(face_boxes, face_angles);
                    int64_t kp_end = getCurrentTime();
                    cout << "kp time : "<< num << " --  "  << kp_end-kp_start << endl;
                    instance_group.update_face_angle(face_angles, reco_boxes);
                    ssd_detection->get_features(reco_boxes, face_fea);
                    vector<int> face_ids;
                    face_lib.get_identity(face_fea, face_ids);
                    instance_group.update_face_id(face_ids);
                });

                std::thread hop_thread([&](){
                    int64_t start = getCurrentTime();
                    hop_det.inference(img, ssd_detection);
                    int64_t end = getCurrentTime();
                    cout << "hop time  : " << num << " -- " << (end - start) << endl;
//                    cout << "timestamp mode 1 -- " << getCurrentTime() << endl;
                });
                track_points_thread.join();
                hop_thread.join();
                instance_group.add_hop_box( hop_det.hat_boxes, hop_det.glass_boxes, hop_det.mask_boxes);
                function_solver.update(image_class, instance_group);
//                post waring

                int64_t end_front = getCurrentTime();
                if (front_num > 30){
                    front_sum += (end_front - start_front);
                    cout << "front time  avg: "<< mode << " " << front_num << " -- " << front_sum / (front_num-30) << endl;
                }

                cout << "front time  : "<< mode << " " << num << " -- " << (end_front - start_front) << endl;
//                cout << "front time  avg: "<< mode << " " << front_num << " -- " << front_sum / num << endl;
//                cout << "front time  avg: "<< mode << " " << front_num << " -- " << front_sum / (front_num-10) << endl;

            } else {
                std::thread tracker_thread([&](){
                    hf_thread.join();
                    head_tracker->run(image_class.head_boxes);
                    instance_group.update_track(num, head_tracker->delete_tracking_id);
                });
                tracker_thread.join();
                function_solver.update(image_class, instance_group);
            }
            cout << "rtmp push start " << endl;
            cv::Mat rtmp_frame = get_vis(img, num, head_tracker->tracking_result, image_class, instance_group, function_solver).clone();
            rtmp_front_img = rtmp_frame.clone();
            cout << "rtmp push over " << endl;
            //                post waring
            bool waringTag = updateWaringFlag(warningFlag, function_solver);
            if (waringTag){
                mQueue_waring_front.push(rtmp_frame);
                con_waring_front.notify_all();
            }
//            ls_handler.pushRTMP(rtmp_frame);
            mQueue_rtmp_front.push(rtmp_frame);
            con_rtmp_front.notify_all();

        } else if (mode == 1){
            if (image_class.wake_state) {
                int64_t start_hand = getCurrentTime();
                std::thread hand_thread([&]()
                                        {
                                            int64_t start = getCurrentTime();
                                            hand_det.inference(img, ssd_detection);
                                            image_class.update_hand(hand_det.hand_boxes);
//                                            cv::Mat rtmp_frame = lsUtils::vis_Box(img, hand_det.hand_boxes);
                                            cv::Mat rtmp_frame = lsUtils::vis_Box(img.clone(), image_class.hand_boxes);
                                            rtmp_top_img = rtmp_frame.clone();
                                            mQueue_rtmp_top.push(rtmp_frame);
                                            con_rtmp_top.notify_all();
                                            int64_t end = getCurrentTime();
                                            cout << "hand time : " << num << " -- " << (end - start) << endl;
//                                            cout << "timestamp -- " << getCurrentTime() << endl;
                                        });
                hand_thread.join();

//                bool waringTag = updateWaringFlagHand(warningFlag, function_solver);
//                if (waringTag){
//                    cout << "hand warning !!" << endl;
//                    mQueue_waring_top.push(img);
//                    con_waring_top.notify_all();
//                }

                int64_t end_hand = getCurrentTime();
                cout << "hand time total: " << num << " -- " << (end_hand - start_hand) << endl;
            } else{
                rtmp_top_img = img.clone();
                cv::Mat rtmp_frame = lsUtils::vis_hand_det_box(rtmp_top_img);
//                mQueue_rtmp_top.push(img);
                mQueue_rtmp_top.push(rtmp_frame);
                con_rtmp_top.notify_all();
            }
        }
        int64_t end_read = getCurrentTime();
//        cout << "read timestamp -- " << end_read << endl;
        cout << "read time cost : " << mode << " " << num << " -- " << (end_read - start_read) << endl;
//        queue->pop();
//        con_v_notification->notify_all();
//        guard.unlock();
//        num ++;
//        cout << "timestamp -- " << getCurrentTime() << endl;
    }

}

void Collect::multithreadTest(){
    thread thread_write_image_front(&Collect::ProduceImage, this, 0);
    thread thread_write_image_top(&Collect::ProduceImage, this, 1);

    thread thread_read_image_front(&Collect::ConsumeImage, this, 0);
    thread thread_read_image_top(&Collect::ConsumeImage, this, 1);

    thread thread_RTMP_front(&Collect::ConsumeRTMPImage, this, 0);
    thread thread_RTMP_top(&Collect::ConsumeRTMPImage, this, 1);

    thread thread_waring_front(&Collect::ConsumeWaringImage, this, 0);
    thread thread_waring_top(&Collect::ConsumeWaringImage, this, 1);

    thread_write_image_front.join();
    thread_write_image_top.join();

    thread_read_image_front.join();
    thread_read_image_top.join();

    thread_RTMP_front.join();
    thread_RTMP_top.join();

    thread_waring_front.join();
    thread_waring_top.join();
}

bool updateWaringFlagHand(WATING_FLAG & warning, Solver function_solver){
    bool result = false;
    if (warning.hand_flag != function_solver.hand_flag && function_solver.hand_flag){
        result = true;
    }
    warning.hand_flag = function_solver.hand_flag;
    return result;
}

bool updateWaringFlag(WATING_FLAG & warning, Solver function_solver){
    bool result = false;

    if (warning.group_flag != function_solver.group_flag && function_solver.group_flag){
        result = true;
    }
    if (warning.tround_flag != function_solver.tround_flag && function_solver.tround_flag){
        result = true;
    }
    if (warning.hop_flag != function_solver.hop_flag && function_solver.hop_flag){
        result = true;
    }
    if (warning.entry_flag != function_solver.entry_flag && function_solver.entry_flag){
        result = true;
    }
    warning.group_flag = function_solver.group_flag;
    warning.tround_flag = function_solver.tround_flag;
    warning.hop_flag = function_solver.hop_flag;
    warning.entry_flag = function_solver.entry_flag;

    return result;


}

void Collect::hf_thread(int clo_num){

    for (int i=0; i<clo_num; i++){
        int64_t start_hf = getCurrentTime();
        hf_det.inference(image_deliver.front_img, ssd_detection);
        int64_t start_hop = getCurrentTime();
        cout << "hf -- " << start_hop - start_hf <<  endl;
    }
}

void Collect::hop_thread(int clo_num){

    for (int i=0; i<clo_num; i++){
        int64_t start_hf = getCurrentTime();
        hop_det.inference(image_deliver.front_img, ssd_detection);
        int64_t start_hop = getCurrentTime();
        cout << "hop -- " << start_hop - start_hf <<  endl;
    }
}

void Collect::hand_thread(int clo_num){

    for (int i=0; i<clo_num; i++){
        int64_t start_hf = getCurrentTime();
        hand_det.inference(image_deliver.top_img, ssd_detection);
        int64_t start_hop = getCurrentTime();
        cout << "hand -- " << start_hop - start_hf <<  endl;
    }
}

void Collect::test(){
    image_deliver.get_frame();
    int64_t start_all = getCurrentTime();
    int clo_num = 2000;
    for (int i=0; i<clo_num; i++){
        int64_t start_hf = getCurrentTime();
        hf_det.inference(image_deliver.front_img, ssd_detection);
        int64_t start_hop = getCurrentTime();
        hop_det.inference(image_deliver.front_img, ssd_detection);
        int64_t start_hand = getCurrentTime();
        hand_det.inference(image_deliver.top_img, ssd_detection);
        int64_t end_hand = getCurrentTime();

        cout << "hf -- " << start_hop - start_hf << " hop -- " << start_hand-start_hop << " hand -- " << end_hand-start_hand << endl;
    }
    int64_t end_all = getCurrentTime();
    cout << "total avg -- " << (end_all-start_all) * 1.0 / clo_num << endl;
}

void Collect::test2(){
    image_deliver.get_frame();
    int clo_num = 1000;
    int64_t start_all = getCurrentTime();
    thread th_hf(&Collect::hf_thread, this, clo_num);
    thread th_hop(&Collect::hop_thread, this, clo_num);
    thread th_hand(&Collect::hand_thread, this, clo_num);

    th_hf.join();
    th_hop.join();
    th_hand.join();
    int64_t end_all = getCurrentTime();
    cout << "total avg -- " << (end_all-start_all) * 1.0 / clo_num << endl;
}