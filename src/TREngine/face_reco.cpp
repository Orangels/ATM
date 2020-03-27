#include <list>
#include <thread>
#include "face_reco.h"
#include "sharedBuffer.h"
#include "Common.h"
#include "StringFunction.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "dataStructure.h"
#include "modelEngine.h"
#include "cudaCommon.h"
#include "pythonCaller.h"
#include "FileFunction.h"
#include "imagesFeatures.h"
#include "instance.h"

Face_Reco ::Face_Reco()
{
    m_pconfiger = CConfiger::getOrCreateConfiger();
	CFactoryDirectory* m_pfactories = CFactoryDirectory::getOrCreateInstance(true);
	fa_m_pdetector = (CModelEngine*)m_pfactories->createProduct("CKeyPointsGenerator");
	fr_m_pdetector = (CModelEngine*)m_pfactories->createProduct("CFaceAttributeCalculator");
}

Face_Reco::~Face_Reco()
{

}
void Face_Reco::get_feature(cv::Mat &image,std::vector<std::vector<float>>& rects, std::vector<std::vector<float>>& feature)
{
    CImage simage;
    char* cpuBuffer = NULL;
    cudaMallocHost((void**)&cpuBuffer, image.cols*image.rows * 3);
    memcpy(cpuBuffer, image.data, image.cols*image.rows * 3);
    simage.initalize(0, 0, image.cols, image.rows, cpuBuffer);
    unsigned char* gpuImage = NULL;
    cudaMalloc((void**)&gpuImage, simage.getClodDataSize());
    simage.setClodData(gpuImage, true);
    simage.transferData(true, fa_m_pdetector->getViceCudaStream());

    int current_size = rects.size();
    std::vector<CFace> fs(current_size);
    for (int i = 0; i < fs.size(); ++i)
    {
        CFace& f = fs[i];
        f.m_pproducer = &simage;
        f.confidence = 0.99;
        f.trackId = i;
        float *pface = new float[sizeof(rects[i])];
        if (!rects[i].empty())
        {
            memcpy(pface, &rects[i][0], rects[i].size()*sizeof(float));
        }
        memcpy(f.xyMinMax, pface, 4 * sizeof(float));
    }

    std::vector<CDataShared*> srcp(4, &fs[0]);
    std::vector<CFaceKeyPoints> facePoints(fa_m_pdetector->getBatchSize());
    for (int i = 0; i < fs.size(); ++i) srcp[i] = &fs[i];

    fa_m_pdetector->preProcessV(srcp);
    exitIfCudaError(cudaStreamSynchronize(fa_m_pdetector->getViceCudaStream()));
    fa_m_pdetector->inference(4);
    exitIfCudaError(cudaStreamSynchronize(fa_m_pdetector->getCudaStream()));
    fa_m_pdetector->postProcessV(srcp, facePoints.data());

    std::vector<CFaceInfor> facesInfor(fr_m_pdetector->getBatchSize());//Batchsize ==4
    std::vector<CDataShared*> nextBatch(4);     //Batchsize ==4
    nextBatch.resize(4);
    for (int i = 0; i < 4; ++i)         //Batchsize ==4
        nextBatch[i] = &facePoints[i];
    fr_m_pdetector->preProcessV(nextBatch);
    fr_m_pdetector->inference(4);            //Batchsize ==4
    fr_m_pdetector->postProcessV(nextBatch, facesInfor.data());
    std::vector<std::vector<float>> fea;
    fr_m_pdetector->get_feature(fea);

    feature = fea;

    cudaFreeHost(cpuBuffer);
    cudaFree(gpuImage);
}