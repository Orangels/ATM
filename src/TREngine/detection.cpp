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
	cudaMalloc((void**)&mhf_gpuImage, 1920*1080*3);
	fr_m_pdetector = (CModelEngine*)m_pfactories->createProduct("CFaceRecognization");

}

SSD_Detection::~SSD_Detection()
{
    cudaFree(mhf_gpuImage);
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
    auto Stream = hf_m_pdetector->getCudaStream();
	cudaSetDevice(gpuIndex);


    mhf_image.initalize(0, 0, image.cols, image.rows, image.data);
    mhf_image.setClodData(mhf_gpuImage, true);
    std::vector<CDataShared*> srcd(batchSize, &mhf_image);
    mhf_image.transferData(true, viceStream);

    hf_m_pdetector->preProcessV(srcd);
    cudaStreamSynchronize(viceStream);
	hf_m_pdetector->inference(batchSize);
	exitIfCudaError(cudaStreamSynchronize(Stream));
	hf_m_pdetector->postProcessV(srcd, dstd.data());
	exitIfCudaError(cudaStreamSynchronize(Stream));
    dstd[0].m_pproducer = (CImage*)srcd[0];
//    std::cout << "head face result" << std::endl;
//    __outputDetections(dstd[0]);
    hf_boxs = dstd[0].detections;

}

void SSD_Detection::get_angles(std::vector<std::vector<float>>& rects, std::vector<std::vector<float>>& angles)
{
    if(rects.size()!=0)
    {
        auto Stream = fa_m_pdetector->getCudaStream();
        int num_faces = rects.size();
        int num_iter = ceil(num_faces / 4.0);
        for(int i =0;i<num_iter;i++)
        {
            // 4face as a batch ,if num of face less than 4, batch = num
            int current_size = num_iter*4 >num_faces? num_faces%4: 4;
            std::vector<CFace> fs(current_size);
            for (int j = 0; j < fs.size(); ++j)
            {
                CFace& f = fs[j];
                f.m_pproducer = &mhf_image;
                f.confidence = 0.99;
                f.trackId = i;
                float *pface = new float[4];
                memcpy(pface, &rects[j][0], rects[j].size() *sizeof(float));
                memcpy(f.xyMinMax, pface , 4 * sizeof(float));
//                std::cout << f.xyMinMax[0] << ", " << f.xyMinMax[1] << ", " << f.xyMinMax[2] << ", " << f.xyMinMax[3] << ", " <<std::endl;
//                std::cout << *pface << ", " << *(pface+1) << ", " << *(pface+2) << ", " << *(pface+3) << ", "<<std::endl;
            }
            std::vector<CDataShared*> srcp(4, &fs[0]);
            std::vector<CFaceKeyPoints> facePoints(fa_m_pdetector->getBatchSize());
            for (int i = 0; i < fs.size(); ++i) srcp[i] = &fs[i];

            fa_m_pdetector->preProcessV(srcp);

            cudaStreamSynchronize(fa_m_pdetector->getViceCudaStream());


            fa_m_pdetector->inference(4);

            exitIfCudaError(cudaStreamSynchronize(Stream));
            fa_m_pdetector->postProcessV(srcp, facePoints.data());
            exitIfCudaError(cudaStreamSynchronize(Stream));
            float *points = fa_m_pdetector->resultOnHost;
            for(int t= 0 ;t<current_size;t++){
                std::vector<float> out;
                out.push_back(*(points+t*219+216));
                out.push_back(*(points+t*219+217));
                out.push_back(*(points+t*219+218));
//                std::cout<<out[0]<< " "<<out[1]<<" "<<out[2]<<std::endl;
                angles.push_back(out);
            }
        }
    }
}

void SSD_Detection::get_features(std::vector<std::vector<float>>& rects, std::vector<std::vector<float>>& features)
{
    if(rects.size()==0) return;
    auto Stream = fa_m_pdetector->getCudaStream();
    int num_faces = rects.size();
    int num_iter = ceil(num_faces / 4.0);
    for(int i =0;i<num_iter;i++)
    {
        // 4face as a batch ,if num of face less than 4, batch = num
        int current_size = num_iter*4 >num_faces? num_faces%4: 4;
        std::vector<CFace> fs(current_size);
        for (int j = 0; j < fs.size(); ++j)
        {
            CFace& f = fs[j];
            f.m_pproducer = &mhf_image;
            f.confidence = 0.99;
            f.trackId = i;
            float *pface = new float[4];
            memcpy(pface, &rects[j][0], rects[j].size() *sizeof(float));
            memcpy(f.xyMinMax, pface , 4 * sizeof(float));
//                std::cout << f.xyMinMax[0] << ", " << f.xyMinMax[1] << ", " << f.xyMinMax[2] << ", " << f.xyMinMax[3] << ", " <<std::endl;
//                std::cout << *pface << ", " << *(pface+1) << ", " << *(pface+2) << ", " << *(pface+3) << ", "<<std::endl;
        }
        std::vector<CDataShared*> srcp(4, &fs[0]);
        std::vector<CFaceKeyPoints> facePoints(fa_m_pdetector->getBatchSize());
        for (int i = 0; i < fs.size(); ++i) srcp[i] = &fs[i];

        fa_m_pdetector->preProcessV(srcp);

        cudaStreamSynchronize(fa_m_pdetector->getViceCudaStream());


        fa_m_pdetector->inference(4);

        exitIfCudaError(cudaStreamSynchronize(Stream));
        fa_m_pdetector->postProcessV(srcp, facePoints.data());
        exitIfCudaError(cudaStreamSynchronize(Stream));

        std::vector<CFaceInfor> facesInfor(fr_m_pdetector->getBatchSize());//Batchsize ==1
        // batch ==1
        for(int i =0 ;i<current_size;i++){
            std::vector<CDataShared*> nextBatch(1);     //Batchsize ==1
            nextBatch.resize(1);
            nextBatch[0] = &facePoints[i];
            fr_m_pdetector->preProcessV(nextBatch);
            cudaStreamSynchronize(fr_m_pdetector->getViceCudaStream());
            fr_m_pdetector->inference(1);            //Batchsize ==4
            cudaStreamSynchronize(fr_m_pdetector->getCudaStream());
            fr_m_pdetector->postProcessV(nextBatch, facesInfor.data());
            cudaStreamSynchronize(fr_m_pdetector->getCudaStream());
            std::vector<float> fea;
            fr_m_pdetector->get_feature(fea);
//            for (int k =0;k<512;k++){
//                std::cout<< k<<": "<<fea[k]<<" ";
//            }
            features.push_back(fea);
        }
    }
}

void SSD_Detection::detect_hand(cv::Mat &image, std::vector<float>& hand_boxs)
{

    int gpuIndex = m_pconfiger->readValue<int>("gpuIndex");
    int batchSize = m_pconfiger->readValue<int>("hand_detectionBatchSize");
    std::vector<CImageDetections> dstd(batchSize);
    auto Stream = hand_m_pdetector->getCudaStream();
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
    simage.transferData(true, hand_m_pdetector->getViceCudaStream());

    hand_m_pdetector->preProcessV(srcd);
    cudaStreamSynchronize(hand_m_pdetector->getViceCudaStream());
	hand_m_pdetector->inference(batchSize);
	exitIfCudaError(cudaStreamSynchronize(Stream));
	hand_m_pdetector->postProcessV(srcd, dstd.data());
	exitIfCudaError(cudaStreamSynchronize(Stream));

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
	exitIfCudaError(cudaStreamSynchronize(hop_m_pdetector->getCudaStream()));
	hop_m_pdetector->postProcessV(srcd, dstd.data());
	exitIfCudaError(cudaStreamSynchronize(hop_m_pdetector->getCudaStream()));
    dstd[0].m_pproducer = (CImage*)srcd[0];
//    std::cout << "hop result" << std::endl,
//    __outputDetections(dstd[0]);
    hop_boxs = dstd[0].detections;

    cudaFreeHost(cpuBuffer);
    cudaFree(gpuImage);
}
