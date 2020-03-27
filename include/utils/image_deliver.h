//
// Created by 王智慧 on 2020/3/18.
//

#ifndef ATM_IMAGE_DELIVER_H
#define ATM_IMAGE_DELIVER_H

#include <opencv2/opencv.hpp>

using namespace std;

class ImageDeliver {
public:
    ImageDeliver();
    ~ImageDeliver();
    void get_frame();

    int img_w, img_h;
    double num_images, front_num_images, top_num_images;
    cv::Mat front_img, top_img;
    cv::VideoCapture front_capture, top_capture;
};

#endif //ATM_IMAGE_DELIVER_H
