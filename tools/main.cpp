//
// Created by 王智慧 on 2020/3/18.
//

#include <iostream>
#include "collect.h"

int main() {
//    std::vector<float> hf_boxs;
//    std::vector<float> hand_boxs;
//    std::vector<float> hop_boxs;   //mask
//
//    cv::Mat image = cv::imread("/srv/ATM/sample_image/000000255800.jpg");
//    CConfiger::getOrCreateConfiger("/srv/ATM_PRIV/ATM/cfg/configer.ini");
//    SSD_Detection ssd_detection;
//    ssd_detection.detect_hf(image, hf_boxs);
//    for (auto b : hf_boxs){
//        std::cout <<b<< std::endl;
//    }
//
//    ssd_detection.detect_hand(image, hand_boxs);
//    for (auto b : hand_boxs){
//        std::cout <<b<< std::endl;
//    }
//    ssd_detection.detect_hop(image, hop_boxs);
//    for (auto b : hop_boxs){
//        std::cout <<b<< std::endl;
//    }
//
//    std::cout << "TEST TREngine" << std::endl;

    Collect collect;
//    collect.multithreadTest();
//    collect.test();
//    collect.run();
    collect.test2();
    return 0;
}
