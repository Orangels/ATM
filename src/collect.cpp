//
// Created by 王智慧 on 2020/3/18.
//

#include "collect.h"
#include "config.h"
#include "utils/misc.h"
#include "utils/vis.h"

Collect::Collect() {
    num_images = image_deliver.num_images;
    Cconfig labels = Cconfig("../cfg/process.ini");
    CConfiger::getOrCreateConfiger("../cfg/configer.ini");
    sleep_frames_skip = stoi(labels["SLEEP_FRAMES_SKIP"]);
    wake_frames_skip = stoi(labels["WAKE_FRAMES_SKIP"]);
    head_track_mistimes = stoi(labels["HEAD_TRACK_MISTIMES"]);
    head_tracker = new Track(head_track_mistimes, stoi(labels["IMAGE_W"]), stoi(labels["IMAGE_H"]));
    ssd_detection = new SSD_Detection;
    face_angle = new Face_Angle;
    face_reco = new Face_Reco;
}

Collect::~Collect() = default;

void Collect::run() {
    struct timeval tp1;
    struct timeval tp2;
    for (int i=0; i<num_images; i++){
//        instance_group.check_state();
        instance_group.clear();
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
//            if(!image_class.keep_head.empty()){
//                for (auto &keep : image_class.keep_head){
//                    cout<<keep<<" ";
//                }
//                cout<<endl;
//            }
            if (image_class.wake_state) {
                head_tracker->run(image_class.head_boxes);
                hop_det.inference(image_deliver.front_img, ssd_detection);
                hand_det.inference(image_deliver.top_img, ssd_detection);
                image_class.update_hand(hand_det.hand_boxes);
                instance_group.update(i, head_tracker->tracking_result, image_class.head_boxes,
                                      image_class.face_boxes, head_tracker->delete_tracking_id);
                instance_group.add_hop_box( hop_det.hat_boxes, hop_det.glass_boxes, hop_det.mask_boxes);
                vector <vector<float>> boxes, face_a, face_b;
                instance_group.get_face_box(boxes);
                if (!boxes.empty()) {
                    face_angle->get_points(image_deliver.front_img, boxes, face_a);
                }
                instance_group.update_face_angle(face_a);

                vis(image_deliver.front_img, i, head_tracker->tracking_result, instance_group);
            } else{
                head_tracker->run(image_class.head_boxes);
                instance_group.update_track(i, head_tracker->delete_tracking_id);
//                face_reco->get_feature(image_deliver.front_img, boxes, face_b);
            }
            function_solver.update(image_class, instance_group);
        }
        gettimeofday(&tp2, NULL);
        cout<<i<<" :  "<< 1000 * (tp2.tv_sec-tp1.tv_sec) + (tp2.tv_usec-tp1.tv_usec)/1000<<endl;
    }
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
//        face_angle->get_points(image_deliver.front_img, boxes, face_a);
        face_reco->get_feature(image_deliver.front_img, boxes, face_b);
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
