#pragma once
#include <vector>
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
class Face_Reco
{
public:
    Face_Reco();
    ~Face_Reco();
    void get_feature(cv::Mat &image,std::vector<std::vector<float>>& rects, std::vector<std::vector<float>>& out_put);
private:
    CModelEngine *fr_m_pdetector, *fa_m_pdetector;
    CConfiger* m_pconfiger;
};