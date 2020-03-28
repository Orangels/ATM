#include <list>
#include <thread>
#include <math.h>
#include "detection.h"
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
#include "hand_ssdDetector.h"
#include "hop_ssdDetector.h"

//ls add
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cublas_v2.h>

SSD_Detection ::SSD_Detection()
{
    m_pconfiger = CConfiger::getOrCreateConfiger();
	CFactoryDirectory* m_pfactories = CFactoryDirectory::getOrCreateInstance(true);
	std::string hf_detectionType = m_pconfiger->readValue("hf_detectionType");
	std::string hand_detectionType = m_pconfiger->readValue("hand_detectionType");
	std::string hop_detectionType = m_pconfiger->readValue("hop_detectionType");

	hf_m_pdetector = (CModelEngine*)m_pfactories->createProduct(hf_detectionType);
	hand_m_pdetector = (CModelEngine*)m_pfactories->createProduct(hand_detectionType);
	hop_m_pdetector =(CModelEngine*)m_pfactories->createProduct(hop_detectionType);
	fa_m_pdetector = (CModelEngine*)m_pfactories->createProduct("CKeyPointsGenerator");

}

SSD_Detection::~SSD_Detection()
{

}

void __outputDetections(const CImageDetections& vd)
{
    std::cout << ((CImage*)vd.m_pproducer)->videoID << " detections : " << vd.detectionCount << std::endl;
    auto p = vd.detections.data();
    for (int i = 0; i < vd.detectionCount; ++i)
        std::cout << *p++ << "," << *p++ << " : " << *p++ << "," << *p++ << "," << *p++ << "," << *p++ << std::endl;
    std::cout << std::endl;
}

void SSD_Detection::detect_hf(cv::Mat &image, std::vector<float>& hf_boxs)
{

    int gpuIndex = m_pconfiger->readValue<int>("gpuIndex");
    int batchSize = m_pconfiger->readValue<int>("hf_detectionBatchSize");
    std::vector<CImageDetections> dstd(batchSize);
    auto viceStream = hf_m_pdetector->getViceCudaStream();
	cudaSetDevice(gpuIndex);
	CImage simage;
	char* cpuBuffer = NULL;
    cudaMallocHost((void**)&cpuBuffer, image.cols*image.rows * 3);
    memcpy(cpuBuffer, image.data, image.cols*image.rows * 3);
    simage.initalize(0, 0, image.cols, image.rows, cpuBuffer);
    unsigned char* gpuImage = NULL;
//    std::cout << "image bites: " << simage.getClodDataSize() << std::endl;
    cudaMalloc((void**)&gpuImage, simage.getClodDataSize());
    simage.setClodData(gpuImage, true);
    std::vector<CDataShared*> srcd(batchSize, &simage);
    simage.transferData(true, viceStream);

    hf_m_pdetector->preProcessV(srcd);
    cudaStreamSynchronize(viceStream);
	hf_m_pdetector->inference(batchSize);
	exitIfCudaError(cudaStreamSynchronize(viceStream));
	hf_m_pdetector->postProcessV(srcd, dstd.data());
    dstd[0].m_pproducer = (CImage*)srcd[0];
//    std::cout << "head face result" << std::endl;
//    __outputDetections(dstd[0]);
    hf_boxs = dstd[0].detections;

    cudaFreeHost(cpuBuffer);
    cudaFree(gpuImage);
}

void SSD_Detection::detect_hf_with_point(cv::Mat &image, std::vector<float>& hf_boxs)
{

    int gpuIndex = m_pconfiger->readValue<int>("gpuIndex");
    int batchSize = m_pconfiger->readValue<int>("hf_detectionBatchSize");
    std::vector<CImageDetections> dstd(batchSize);
    auto viceStream = hf_m_pdetector->getViceCudaStream();
	cudaSetDevice(gpuIndex);
	CImage simage;
	char* cpuBuffer = NULL;
    cudaMallocHost((void**)&cpuBuffer, image.cols*image.rows * 3);
    memcpy(cpuBuffer, image.data, image.cols*image.rows * 3);
    simage.initalize(0, 0, image.cols, image.rows, cpuBuffer);
    unsigned char* gpuImage = NULL;
//    std::cout << "image bites: " << simage.getClodDataSize() << std::endl;
    cudaMalloc((void**)&gpuImage, simage.getClodDataSize());
    simage.setClodData(gpuImage, true);
    std::vector<CDataShared*> srcd(batchSize, &simage);
    simage.transferData(true, viceStream);

    hf_m_pdetector->preProcessV(srcd);
    cudaStreamSynchronize(viceStream);
	hf_m_pdetector->inference(batchSize);
	exitIfCudaError(cudaStreamSynchronize(viceStream));
	hf_m_pdetector->postProcessV(srcd, dstd.data());
    dstd[0].m_pproducer = (CImage*)srcd[0];
//    __outputDetections(dstd[0]);
    hf_boxs = dstd[0].detections;

    //  do something to the boxes
    std::vector<float> boxs_left = hf_boxs;

    std::vector<float> angles;
    int num_faces = boxs_left.size() / 6;
    int num_iter = ceil(num_faces / 4.0);

    for(int i =0;i<num_iter;i++)
    {
        // 4face as a batch ,if num of face less than 4, batch = num
        int current_size = num_iter*4 >num_faces? num_faces%4: 4;
        std::vector<CFace> fs(current_size);
        for (int j = 0; j < fs.size(); ++j)
        {
            CFace& f = fs[j];
            f.m_pproducer = &simage;
            f.confidence = 0.99;
            f.trackId = i;
            float *pface = new float[6];
            memcpy(pface, &boxs_left[(i*4 +j)*6], 6 *sizeof(float));
            memcpy(f.xyMinMax, pface + 2, 4 * sizeof(float));
//            std::cout << f.xyMinMax[0] << ", " << f.xyMinMax[1] << ", " << f.xyMinMax[2] << ", " << f.xyMinMax[3] << ", " <<std::endl;
        }
        std::vector<CDataShared*> srcp(4, &fs[0]);
        std::vector<CFaceKeyPoints> facePoints(fa_m_pdetector->getBatchSize());
        for (int i = 0; i < fs.size(); ++i) srcp[i] = &fs[i];

        fa_m_pdetector->preProcessV(srcp);
        exitIfCudaError(cudaStreamSynchronize(fa_m_pdetector->getViceCudaStream()));
        fa_m_pdetector->inference(4);
        exitIfCudaError(cudaStreamSynchronize(fa_m_pdetector->getCudaStream()));
        fa_m_pdetector->postProcessV(srcp, facePoints.data());
        float* p68keypointsCPU = (float*)facePoints.front().getClodData(true);
        //    std::cout <<"p68keypointsCPU"<<p68keypointsCPU<<std::endl;

        cudaStream_t cudaStream = fa_m_pdetector->getCudaStream();
        std::vector<std::vector<float>> outs;
        for(int t= 0 ;t<current_size;t++){
            std::string vInfor = std::to_string(t) + "p73";
            std::vector<float> r(219);
            std::vector<float> out;
            cudaMemcpyAsync(r.data(), p68keypointsCPU + t*219, sizeof(float)*219, cudaMemcpyDeviceToHost, cudaStream);
            cudaStreamSynchronize(cudaStream);

//            std::cout <<" angle "<< r[216]<<" "<<r[217]<<" "<<r[218]<<std::endl;
            angles.push_back(r[216]);
            angles.push_back(r[217]);
            angles.push_back(r[218]);
        }
    }
    for(int t=0;t<angles.size();t++) std::cout<< angles[t]<<" ";

    cudaFreeHost(cpuBuffer);
    cudaFree(gpuImage);
}

void SSD_Detection::detect_hand(cv::Mat &image, std::vector<float>& hand_boxs)
{

    int gpuIndex = m_pconfiger->readValue<int>("gpuIndex");
    int batchSize = m_pconfiger->readValue<int>("hand_detectionBatchSize");
    std::vector<CImageDetections> dstd(batchSize);
    auto viceStream = hand_m_pdetector->getViceCudaStream();
	cudaSetDevice(gpuIndex);
	CImage simage;
	char* cpuBuffer = NULL;
    cudaMallocHost((void**)&cpuBuffer, image.cols*image.rows * 3);
    memcpy(cpuBuffer, image.data, image.cols*image.rows * 3);
    simage.initalize(0, 0, image.cols, image.rows, cpuBuffer);
    unsigned char* gpuImage = NULL;
//    std::cout << "image bites: " << simage.getClodDataSize() << std::endl;
    cudaMalloc((void**)&gpuImage, simage.getClodDataSize());
    simage.setClodData(gpuImage, true);
    std::vector<CDataShared*> srcd(batchSize, &simage);
    simage.transferData(true, viceStream);

    hand_m_pdetector->preProcessV(srcd);
    cudaStreamSynchronize(viceStream);
	hand_m_pdetector->inference(batchSize);
	exitIfCudaError(cudaStreamSynchronize(viceStream));
	hand_m_pdetector->postProcessV(srcd, dstd.data());

    dstd[0].m_pproducer = (CImage*)srcd[0];
//    std::cout << "hand result" << std::endl;
//    __outputDetections(dstd[j]);
    hand_boxs = dstd[0].detections;

    cudaFreeHost(cpuBuffer);
    cudaFree(gpuImage);
}

void SSD_Detection::detect_hop(cv::Mat &image, std::vector<float>& hop_boxs)
{

    int gpuIndex = m_pconfiger->readValue<int>("gpuIndex");
    int batchSize = m_pconfiger->readValue<int>("hop_detectionBatchSize");
    std::vector<CImageDetections> dstd(batchSize);
    auto viceStream = hop_m_pdetector->getViceCudaStream();
	cudaSetDevice(gpuIndex);
	CImage simage;
	char* cpuBuffer = NULL;
    cudaMallocHost((void**)&cpuBuffer, image.cols*image.rows * 3);
    memcpy(cpuBuffer, image.data, image.cols*image.rows * 3);
    simage.initalize(0, 0, image.cols, image.rows, cpuBuffer);
    unsigned char* gpuImage = NULL;
//    std::cout << "image bites: " << simage.getClodDataSize() << std::endl;
    cudaMalloc((void**)&gpuImage, simage.getClodDataSize());
    simage.setClodData(gpuImage, true);
    std::vector<CDataShared*> srcd(batchSize, &simage);
    simage.transferData(true, viceStream);

    hop_m_pdetector->preProcessV(srcd);
    cudaStreamSynchronize(viceStream);
	hop_m_pdetector->inference(batchSize);
	exitIfCudaError(cudaStreamSynchronize(viceStream));
	hop_m_pdetector->postProcessV(srcd, dstd.data());
    dstd[0].m_pproducer = (CImage*)srcd[0];
//    std::cout << "hop result" << std::endl,
//    __outputDetections(dstd[j]);
    hop_boxs = dstd[0].detections;

    cudaFreeHost(cpuBuffer);
    cudaFree(gpuImage);
}
