#include <list>
#include <thread>
#include "face_angle.h"
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

//ls add
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cublas_v2.h>

Face_Angle ::Face_Angle()
{
    m_pconfiger = CConfiger::getOrCreateConfiger();
	CFactoryDirectory* m_pfactories = CFactoryDirectory::getOrCreateInstance(true);
	fa_m_pdetector = (CModelEngine*)m_pfactories->createProduct("CKeyPointsGenerator");
	cudaMallocHost((void**)&pointsOnHost,219*sizeof(float));
}

Face_Angle::~Face_Angle()
{

}

void Face_Angle::get_points(cv::Mat &image,std::vector<std::vector<float>>& rects, std::vector<std::vector<float>>& out_put)
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
//    std::vector<CFace> fs(current_size);
    std::vector<CFace> fs(4);
    for (int i = 0; i < fs.size(); ++i)
    {
        CFace& f = fs[i];
        f.m_pproducer = &simage;
        f.confidence = 0.99;
        f.trackId = i;
//        float *pface = new float[sizeof(rects[i])];
        float *pface = new float[sizeof(rects[0])];
//        if (!rects[i].empty())
        if (!rects[0].empty())
        {
            memcpy(pface, &rects[0][0], rects[0].size()*sizeof(float));
        }
        memcpy(f.xyMinMax, pface, 4 * sizeof(float));
    }

    std::vector<CDataShared*> srcp(4, &fs[0]);
    std::vector<CFaceKeyPoints> facePoints(fa_m_pdetector->getBatchSize());
    for (int i = 0; i < fs.size(); ++i) srcp[i] = &fs[i];

    fa_m_pdetector->preProcessV(srcp);
//    exitIfCudaError(cudaStreamSynchronize(fa_m_pdetector->getViceCudaStream()));
//    fa_m_pdetector->inference(4);
//    exitIfCudaError(cudaStreamSynchronize(fa_m_pdetector->getCudaStream()));
//    fa_m_pdetector->postProcessV(srcp, facePoints.data());

//    float* p68keypointsCPU = (float*)facePoints.front().getClodData(true);
//    cudaStream_t cudaStream = fa_m_pdetector->getCudaStream();
    std::vector<std::vector<float>> outs;
//    for(int t= 0 ;t<current_size;t++){
//        cudaMemcpyAsync(pointsOnHost, p68keypointsCPU + t*219, sizeof(float)*219, cudaMemcpyDeviceToHost, cudaStream);
//    }
    out_put = outs;
    
    cudaFreeHost(cpuBuffer);
    cudaFree(gpuImage);
}