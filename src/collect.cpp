//
// Created by 王智慧 on 2020/3/18.
//

#include "collect.h"
#include "config.h"
#include "utils/misc.h"
#include "utils/vis.h"
#include "ls_vis.h"

int64_t getCurrentTime()
{
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

Collect::Collect() {
    num_images = image_deliver.num_images;
    Cconfig labels = Cconfig("../cfg/process.ini");
    CConfiger* m_pconfiger = CConfiger::getOrCreateConfiger("../cfg/configer.ini");
    sleep_frames_skip = stoi(labels["SLEEP_FRAMES_SKIP"]);
    wake_frames_skip = stoi(labels["WAKE_FRAMES_SKIP"]);
    head_track_mistimes = stoi(labels["HEAD_TRACK_MISTIMES"]);
    head_tracker = new Track(head_track_mistimes, stoi(labels["IMAGE_W"]), stoi(labels["IMAGE_H"]));
    ssd_detection = new SSD_Detection;
    face_angle = new Face_Angle;
//    face_reco = new Face_Reco;

//    ls add
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
}

Collect::~Collect() = default;

void Collect::run() {
    struct timeval tp1;
    struct timeval tp2;
    for (int i=0; i<num_images; i++){
        instance_group.check_state();
        image_class.frame_id = i;
        gettimeofday(&tp1, NULL);
        image_deliver.get_frame();

        if (image_class.wake_state){
            frames_skip = wake_frames_skip;
        } else{
            frames_skip = sleep_frames_skip;
        }

        if (i % frames_skip == 0){
            hf_det.inference(image_deliver.front_img, ssd_detection);
            image_class.update_hf(hf_det.head_boxes, hf_det.face_boxes);
            hop_det.inference(image_deliver.front_img, ssd_detection);
            hand_det.inference(image_deliver.top_img, ssd_detection);
            if(!image_class.keep_head.empty()){
                for (auto &keep : image_class.keep_head){
                    cout<<keep<<" ";
                }
                cout<<endl;
            }

            if (image_class.wake_state and !image_class.head_boxes.empty()){
                head_tracker->run(image_class.head_boxes);
                vector<vector<float>> boxes, face_a, face_b;
                boxes = box2vector(image_class.head_boxes);
                face_angle->get_points(image_deliver.front_img, boxes, face_a);
                instance_group.update(i, head_tracker->tracking_result, image_class.head_boxes,
                                      face_a, head_tracker->delete_tracking_id);

//                vis(image_deliver.front_img, i, head_tracker->tracking_result, image_class.head_boxes,
//                        face_a);

//                face_reco->get_feature(image_deliver.front_img, boxes, face_b);
//                cout<<"track id: "<<endl;
//                for (auto &keep : head_tracker->tracking_result){
//                    cout<<keep<<" ";
//                }
//                cout<<endl;
//                cout<<"delete id: "<<endl;
//                for (auto &keep : head_tracker->delete_tracking_id){
//                    cout<<keep<<" ";
//                }
//                cout<<endl;
            }
            function_solver.update(image_class, instance_group);
        }
        gettimeofday(&tp2, NULL);
        cout<<i<<" :  "<< 1000 * (tp2.tv_sec-tp1.tv_sec) + (tp2.tv_usec-tp1.tv_usec)/1000<<endl;
    }
//        imshow("output1", image_deliver.front_img);
//        imshow("output2", image_deliver.top_img);
//        cv::waitKey(1);
}

void Collect::ProduceImage(int mode){

    cv::VideoCapture cam;
    cv::Mat frame;
    string path = "";
    mutex *lock;
    queue<cv::Mat> *queue;
    condition_variable *con_v_wait, *con_v_notification;

    switch (mode){
        case 0:
//            path = "rtspsrc location=rtsp://admin:sx123456@192.168.88.37:554/h264/ch2/sub/av_stream latency=0 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! videoconvert ! appsink";
            path = "filesrc location=/srv/ATM_ls/ATM/data/front.mp4 ! qtdemux ! queue ! h264parse !  omxh264dec  ! nvvidconv ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! videoconvert  ! appsink";
//            path = "../data/front.mp4";
            lock = &myMutex_front;
            queue = &mQueue_front;
            con_v_wait = &con_front_not_full;
            con_v_notification = &con_front_not_empty;
            break;
        case 1:
//            path = "rtspsrc location=rtsp://admin:sx123456@192.168.88.37:554/h264/ch2/sub/av_stream latency=0 ! rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! videoconvert ! appsink";
            path = "filesrc location=/srv/ATM_ls/ATM/data/top.mp4 ! qtdemux ! queue ! h264parse !  omxh264dec  ! nvvidconv ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! videoconvert  ! appsink";
//            path = "../data/front.mp4";
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
    for (int i=0; i<clo_num; i++) {
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

        if (i % frames_skip != 0){
            if (i % 2 == 1 && mode == 0){
                rtmpMutex.lock();
                ls_handler.pushRTMP(frame);
                rtmpMutex.unlock();
            }

            if (i % 2 == 1 && mode == 1){
                rtmpMutex_2.lock();
                ls_handler_2.pushRTMP(frame);
                rtmpMutex_2.unlock();
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
    cout << "timestamp -- " << getCurrentTime() << endl;
    cout << "start timestamp --- " << start_all << endl;
}

void Collect::ConsumeImage(int mode){
    cv::Mat img;
    int num = 0;

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
        img = queue->front();
        if (mode == 0){
            std::thread hf_thread([&]()
                                  {
                                      int64_t start = getCurrentTime();
                                      hf_det.inference(img, ssd_detection);
                                      int64_t end = getCurrentTime();
                                      image_class.update_hf(hf_det.head_boxes, hf_det.face_boxes);
//                                      int64_t end = getCurrentTime();
                                      if (image_class.wake_state and !image_class.head_boxes.empty()) {
                                          int64_t start_kp = getCurrentTime();
                                          head_tracker->run(image_class.head_boxes);
                                          vector <vector<float>> boxes, face_a, face_b;
                                          boxes = box2vector(image_class.head_boxes);
                                          face_angle->get_points(img, boxes, face_a);
                                          instance_group.update(num, head_tracker->tracking_result,
                                                                image_class.head_boxes,
                                                                face_a, head_tracker->delete_tracking_id);
                                          cv::Mat rtmp_frame = lsUtils::vis(img, num, head_tracker->tracking_result,
                                                                            image_class.head_boxes,face_a);
                                          rtmpMutex.lock();
                                          cout << "33333333333333" << endl;
                                          ls_handler.pushRTMP(rtmp_frame);
                                          cout << "44444444444444" << endl;
                                          rtmpMutex.unlock();

                                          int64_t end_kp = getCurrentTime();
                                          cout << "3d kp time : " << num << " -- " << end_kp-start_kp << endl;
                                      }
                                      function_solver.update(image_class, instance_group);
                                      cout << "hf time  : " << num << " -- " << (end - start) << endl;
                                      cout << "timestamp mode 0 -- " << getCurrentTime() << endl;
                                  });
            std::thread hop_thread([this, img, num]()
                                   {
                                       int64_t start = getCurrentTime();
                                       hop_det.inference(img, ssd_detection);
                                       int64_t end = getCurrentTime();
                                       cout << "hop time  : " << num << " -- " << (end - start) << endl;
                                       cout << "timestamp mode 1 -- " << getCurrentTime() << endl;
                                   });
            hf_thread.join();
            hop_thread.join();

        } else if (mode == 1){
//            std::thread hand_thread([this, img, num]()
            std::thread hand_thread([&]()
                                    {
                                        int64_t start = getCurrentTime();
                                        hand_det.inference(img, ssd_detection);
                                        int64_t end = getCurrentTime();
                                        cv::Mat rtmp_frame = lsUtils::vis_Box(img, hand_det.hand_boxes);
                                        rtmpMutex_2.lock();
                                        ls_handler_2.pushRTMP(rtmp_frame);
                                        rtmpMutex_2.unlock();
                                        cout << "hand time  : " << num << " -- " << (end - start) << endl;
                                        cout << "timestamp -- " << getCurrentTime() << endl;
                                    });
            hand_thread.join();
        }
        int64_t end_read = getCurrentTime();
        cout << "read timestamp -- " << end_read << endl;
        cout << "read time cost : " << mode << " " << num << " -- " << (end_read - start_read) << endl;
        queue->pop();
        con_v_notification->notify_all();
        guard.unlock();
        num ++;
//        cout << "timestamp -- " << getCurrentTime() << endl;
    }

}

void Collect::multithreadTest(){
    thread thread_write_image_front(&Collect::ProduceImage, this, 0);
    thread thread_write_image_top(&Collect::ProduceImage, this, 1);

    thread thread_read_image_front(&Collect::ConsumeImage, this, 0);
    thread thread_read_image_top(&Collect::ConsumeImage, this, 1);

    thread_write_image_front.join();
    thread_write_image_top.join();

    thread_read_image_front.join();
    thread_read_image_top.join();
}

void Collect::test() {
    cv::Mat image = cv::imread("/srv/ATM/sample_image/000000255800.jpg");

    image_deliver.get_frame();
    // clock_t start_all = clock();

    struct timeval tp;
    struct timeval tp1;
    int g_time_start;
    int g_time_end;
    gettimeofday(&tp,NULL);
    g_time_start = tp.tv_sec * 1000 + tp.tv_usec/1000;
    int clo_num = 300;
    for (int i=0; i<clo_num; i++){
        image_class.frame_id = i;
        image_deliver.get_frame();
        hf_det.inference(image_deliver.front_img, ssd_detection);
        hop_det.inference(image_deliver.front_img, ssd_detection);
        hand_det.inference(image_deliver.top_img, ssd_detection);

        // hf_det.inference(image, ssd_detection);
        // hop_det.inference(image, ssd_detection);
        // hand_det.inference(image, ssd_detection);
    }
    // clock_t end_all = clock();
    // cout << (end_all - start_all) / 1000. / clo_num << endl;
    gettimeofday(&tp1,NULL);
    g_time_end = tp1.tv_sec * 1000 + tp1.tv_usec/1000;

    cout << (g_time_end - g_time_start) / clo_num << endl;
    // cout << g_time_end - g_time_start << endl;
    cout << "read img over" << endl;
}

void Collect::test2() {
    cv::Mat image = cv::imread("/srv/ATM/sample_image/000000255800.jpg");

    image_deliver.get_frame();
    hf_det.inference(image_deliver.front_img, ssd_detection);
    image_class.update_hf(hf_det.head_boxes, hf_det.face_boxes);
    cout<<image_class.head_boxes.size()<<endl;

    struct timeval tp;
    struct timeval tp1;
    int g_time_start;
    int g_time_end;
    gettimeofday(&tp,NULL);
    g_time_start = tp.tv_sec * 1000 + tp.tv_usec/1000;
    int clo_num = 300;
    for (int i=0; i<clo_num; i++){
        vector<vector<float>> boxes, face_a, face_b;
        boxes = box2vector(image_class.head_boxes);
        face_angle->get_points(image_deliver.front_img, boxes, face_a);
//        face_reco->get_feature(image_deliver.front_img, boxes, face_b);
//        image_class.frame_id = i;
//        image_deliver.get_frame();
//        hf_det.inference(image_deliver.front_img, ssd_detection);
//        hop_det.inference(image_deliver.front_img, ssd_detection);
//        hand_det.inference(image_deliver.top_img, ssd_detection);
//
//        // hf_det.inference(image, ssd_detection);
//        // hop_det.inference(image, ssd_detection);
//        // hand_det.inference(image, ssd_detection);
    }
    gettimeofday(&tp1,NULL);
    g_time_end = tp1.tv_sec * 1000 + tp1.tv_usec/1000;

    cout << (g_time_end - g_time_start) / clo_num << endl;
    cout << "read img over" << endl;
}
