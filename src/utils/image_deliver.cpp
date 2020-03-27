//
// Created by 王智慧 on 2020/3/18.
//

#include "utils/image_deliver.h"
#include "config.h"

using namespace std;

ImageDeliver::ImageDeliver() {
    Cconfig labels = Cconfig("../cfg/process.ini");
    front_capture.open(labels["VIDEO_PATH_FRONT"]);
    front_num_images = front_capture.get(7);
    top_capture.open(labels["VIDEO_PATH_TOP"]);
    top_num_images = top_capture.get(7);
    num_images = min(front_num_images, top_num_images);
    img_w = stoi(labels["IMAGE_W"]);
    img_h = stoi(labels["IMAGE_H"]);
}

ImageDeliver::~ImageDeliver() = default;

void ImageDeliver::get_frame() {
    cv::Mat f_img, t_img;
    front_capture.read(f_img);
    top_capture.read(t_img);
    resize(f_img, front_img, cv::Size(img_w, img_h), 0, 0, 1);
    resize(t_img, top_img, cv::Size(img_w, img_h), 0, 0, 1);
}