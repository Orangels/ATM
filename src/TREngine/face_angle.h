#pragma once
#include <vector>

//#include "rtmpHandler.h"
#include <stdio.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class CConfiger;
class CDataShared;
class CModelEngine;
class CSharedBuffer;
class CImagesFeatures;
class CFactoryDirectory;
class Face_Angle
{
public:
    Face_Angle();
    ~Face_Angle();
    void get_points(cv::Mat &image,std::vector<std::vector<float>>& rects, std::vector<std::vector<float>>& out_put);
private:
    CModelEngine *fa_m_pdetector;
    CConfiger* m_pconfiger;
    float * pointsOnHost;
};