#include <list>
#include <thread>
#include "deepViewerTest.h"
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

CDeepViewer::CDeepViewer()
{
    m_pconfiger = CConfiger::getOrCreateConfiger();
    int imageBufferLength = m_pconfiger->readValue<int>("imageBufferLength");
    int gpuIndex = m_pconfiger->readValue<int>("gpuIndex");
    cudaSetDevice(gpuIndex);

    std::vector<int> widthHegth;
    m_pconfiger->readValue("maxSrcImageShape", widthHegth);
    size_t maxSrcImageBites = widthHegth.front()*widthHegth.back() * 3;
    m_imagebuffer = new CSharedBuffer(imageBufferLength, "CImage", maxSrcImageBites, true);

    m_pfactories = CFactoryDirectory::getOrCreateInstance(true);
    std::string detectionType = m_pconfiger->readValue("detectionType");
    m_pdetector = (CModelEngine*)m_pfactories->createProduct(detectionType);

    m_detectionBuffer = new CSharedBuffer(imageBufferLength, "CImageDetections");

    // tracking
//    m_trackingBuffer = new CSharedBuffer(imageBufferLength, "CTrack");
//    int faceBufferLength = m_pconfiger->readValue<int>("faceBufferLength");
//    m_faceBuffer = new CSharedBuffer(faceBufferLength, "CFace");
//    m_p3DkeypointGenerator = (CModelEngine*)m_pfactories->createProduct("CKeyPointsGenerator");
//    m_faceKeyPointsBuffer = new CSharedBuffer(faceBufferLength, "CFaceKeyPoints", 219 * sizeof(float), true);
//    m_pfaceInforCalculator = (CModelEngine*)m_pfactories->createProduct("CFaceAttributeCalculator");
//    m_faceInforBuffer = new CSharedBuffer(faceBufferLength, "CFaceInfor");
    m_buildFeatureDataFlag = (m_pconfiger->readValue("recognizationRunModel") == "buildFeatureData");
    m_namesFeats = CImagesFeatures::getOrCreateInstance();
//    m_submitReult = ("submit" == m_pconfiger->readValue("outputResult"));
    if (m_buildFeatureDataFlag) m_submitReult = false;
}

CDeepViewer::~CDeepViewer()
{
    delete m_pdetector;
//    delete m_p3DkeypointGenerator;
//    delete m_pfaceInforCalculator;
    delete m_detectionBuffer;
//    delete m_trackingBuffer;
//    delete m_faceBuffer;
//    delete m_faceKeyPointsBuffer;
//    delete m_faceInforBuffer;
}

//void CDeepViewer::run()
//{
//    std::thread inputter(!m_namesFeats->isImagesPath() ? &CDeepViewer::__getVideoImages : &CDeepViewer::__getImages, this);
//    std::thread detector(&CDeepViewer::__detectFaceHead, this);
//    std::thread obejctTracker(&CDeepViewer::__trackHead, this);
//    std::thread keyPointsGenerator(&CDeepViewer::__generate3DkeyPoints, this);
//    std::thread faceAtributesCalculator(&CDeepViewer::__calculateFaceAttribute, this);
//    std::thread outputter(m_submitReult?&CDeepViewer::__submitResults:&CDeepViewer::__displayDetectionsAndFaceInfor, this);
//
//    inputter.join();
//    detector.join();
//    obejctTracker.join();
//    keyPointsGenerator.join();
//    faceAtributesCalculator.join();
//    outputter.join();
//}

void __outputDetection(const CImageDetections& vd)
{
    std::cout << ((CImage*)vd.m_pproducer)->videoID << " detections : " << vd.detectionCount << std::endl;
    auto p = vd.detections.data();
    for (int i = 0; i < vd.detectionCount; ++i)
        std::cout << *p++ << "," << *p++ << " : " << *p++ << "," << *p++ << "," << *p++ << "," << *p++ << std::endl;
    std::cout << std::endl;
}


void CDeepViewer::run()
{
    //read data
    std::cout << "image read start" << std::endl;
    cv::Mat img;
    CImage image;
    bool buildFeatureFlag = isStartWith(m_pconfiger->readValue("recognizationRunModel"), "build");
    std::vector<int> tt;
    splitStringAndConvert(m_pconfiger->readValue("itest"), tt, ',');
    int i = tt.front(), iend = tt.back();
    if (iend > m_namesFeats->getImageCount()) iend = m_namesFeats->getImageCount();

    //detection
    int idetect = 0;
    int batchSize = m_pdetector->getBatchSize();
    std::vector<CDataShared*> currentBatchImages, nextBatchImages;
    auto detectionReprocess = [&]()
    {
        m_imagebuffer->readData(nextBatchImages, batchSize);
        for (auto p : nextBatchImages)
            p->transferData(true, m_pdetector->getViceCudaStream());
        m_pdetector->preProcessV(nextBatchImages);
        exitIfCudaError(cudaStreamSynchronize(m_pdetector->getViceCudaStream()));
    };
    std::thread dataPreparation(detectionReprocess);
    std::vector<CImageDetections> detections(batchSize);

    CTimer timer;

    for (; i < iend ; ++i)
    {
        //read data
// 		std::string imgName = "/srv/VisionProject/models/ATM_caffe/dataset/Spider/000610.jpg" ; //m_namesFeats->getImageFileName(i);
        std::string imgName =m_namesFeats->getImageFileName(i);
        std::cout << " read " << imgName << std::endl;
        img = cv::imread(imgName);
        if (img.rows > 1080)
        {
            int heigt = 1080;
            cv::resize(img, img, cv::Size(heigt*img.cols*1.f / img.rows, heigt));
        }
        if (img.cols > 1920 || img.cols%32>0)
        {
            int width = (img.cols % 32==0? 1920: img.cols/32*32);
            cv::resize(img, img, cv::Size(width, img.rows* width*1.f / img.cols));
        }
        if (img.empty()) { std::cout << "Can't read " << imgName << std::endl; continue; }
        else  if (1)std::cout << "width_height" << img.cols << ", " << img.rows << std::endl;
        image.initalize(buildFeatureFlag?i:0, i, img.cols, img.rows, img.data);
        m_imagebuffer->writeToBuffer(&image, 1);

        //detection
        dataPreparation.join();
        timer.start(currentBatchImages.size());
        currentBatchImages = nextBatchImages;
        m_pdetector->inference(currentBatchImages.size());
        dataPreparation = std::thread(detectionReprocess);
        m_pdetector->postProcessV(currentBatchImages, detections.data());
        timer.stop();
        std::cout << "detections times :" << timer.getAvgMillisecond() << std::endl;
        for (int i = 0; i < currentBatchImages.size(); i++)
        {
            detections[i].m_pproducer = (CImage*)currentBatchImages[i];
            if (m_buildFeatureDataFlag)
//                __reserveBestFace(&detections[i]);
                std::cout << "__reserveBestFace " << std::endl;
 			else
 				 __outputDetection(detections[i]);
        }
//        m_detectionBuffer->writeToBuffer(detections.data(), currentBatchImages.size());




    }
    std::cout << "image read finished!" << std::endl;

}


void CDeepViewer::__getImages()
{
    cv::Mat img;
    CImage image;
    bool buildFeatureFlag = isStartWith(m_pconfiger->readValue("recognizationRunModel"), "build");
    std::vector<int> tt;
    splitStringAndConvert(m_pconfiger->readValue("itest"), tt, ',');
    int i = tt.front(), iend = tt.back();
    if (iend > m_namesFeats->getImageCount()) iend = m_namesFeats->getImageCount();
    for (; i < iend ; ++i)
    {
// 		std::string imgName = "/srv/VisionProject/models/ATM_caffe/dataset/Spider/000610.jpg" ; //m_namesFeats->getImageFileName(i);
        std::string imgName =m_namesFeats->getImageFileName(i);
        img = cv::imread(imgName);
        if (img.rows > 1080)
        {
            int heigt = 1080;
            cv::resize(img, img, cv::Size(heigt*img.cols*1.f / img.rows, heigt));
        }
        if (img.cols > 1920 || img.cols%32>0)
        {
            int width = (img.cols % 32==0? 1920: img.cols/32*32);
            cv::resize(img, img, cv::Size(width, img.rows* width*1.f / img.cols));
        }
        if (img.empty()) { std::cout << "Can't read " << imgName << std::endl; continue; }
        else  if (i==tt.front())std::cout << "width_height" << img.cols << ", " << img.rows << std::endl;
        image.initalize(buildFeatureFlag?i:0, i, img.cols, img.rows, img.data);
        m_imagebuffer->writeToBuffer(&image, 1);
    }
    std::cout << "image read finished!" << std::endl;
}

void CDeepViewer::__getVideoImages()
{

    int jumpFrame = m_pconfiger->readValue<int>("jumpFrame");

    std::string videoNames = m_pconfiger->readValue("videoNames");
    std::vector<std::string> nameSet;
    splitString(videoNames, nameSet, ',');
    std::list<std::pair<cv::VideoCapture, int>> capIDs;

    for (int i = 0; i < nameSet.size(); ++i)
    {
        capIDs.emplace_back();
        auto& capid = capIDs.back();
        bool flag = (nameSet[i].size() == 1 ? capid.first.open(nameSet[i][0] - '0') : capid.first.open(nameSet[i]));
        if (!flag) { capIDs.pop_back(); std::cout << "Can't open video of " << nameSet[i] << std::endl; }
        else capid.second = i;
    }

    std::vector<cv::Mat> frames(capIDs.size());
    std::vector<CImage> images;
    int itest = 0;
    while (capIDs.size())
    {
        itest++;
        images.resize(capIDs.size());
        auto dst = images.begin();
        for (auto ibegin = capIDs.begin(); ibegin != capIDs.end() && capIDs.size();)
        {
            int videoid = ibegin->second;
            auto& cap = ibegin->first;
            auto& frame = frames[videoid];
//            std::cout << ". " << ibegin ;
            if (cap.read(frame))
            {
                if (itest % jumpFrame == 0) {
                    dst++->initalize(videoid, itest, frame.cols, frame.rows, frame.data);
                }
                ibegin++;
            }
            else
            {
                std::cout << "finish reading video " << videoid << ", " << itest << std::endl;
                cap.release(); ibegin = capIDs.erase(ibegin);

                //重播
                for (int i = 0; i < nameSet.size(); ++i)
                {
                    capIDs.emplace_back();
                    auto& capid = capIDs.back();
                    bool flag = (nameSet[i].size() == 1 ? capid.first.open(nameSet[i][0] - '0') : capid.first.open(nameSet[i]));
                    if (!flag) { capIDs.pop_back(); std::cout << "Can't open video of " << nameSet[i] << std::endl; }
                    else capid.second = i;
                }
                ibegin = capIDs.begin();

            }

            static bool firstCome = true;
            if (firstCome) { firstCome = false; std::cout << frame.cols << ", " << frame.rows << std::endl; }
        }
        // 		if (itest>383)
        if (itest % jumpFrame == 0) {
            m_imagebuffer->writeToBuffer(images.data(), images.size());
        }
        m_imagebuffer->writeToBuffer(images.data(), images.size());
    }
    std::cout << "finish read all videos." << std::endl;
}


void CDeepViewer::__detectFaceHead()
{
    int idetect = 0;
    int batchSize = m_pdetector->getBatchSize();
    std::vector<CDataShared*> currentBatchImages, nextBatchImages;
    auto detectionReprocess = [&]()
    {
        m_imagebuffer->readData(nextBatchImages, batchSize);
        for (auto p : nextBatchImages)
            p->transferData(true, m_pdetector->getViceCudaStream());
        m_pdetector->preProcessV(nextBatchImages);
        exitIfCudaError(cudaStreamSynchronize(m_pdetector->getViceCudaStream()));
    };
    std::thread dataPreparation(detectionReprocess);
    std::vector<CImageDetections> detections(batchSize);
    CTimer timer;
    while (true)
    {
        dataPreparation.join();
        timer.start(currentBatchImages.size());
        currentBatchImages = nextBatchImages;
        m_pdetector->inference(currentBatchImages.size());
        dataPreparation = std::thread(detectionReprocess);
        m_pdetector->postProcessV(currentBatchImages, detections.data());
        timer.stop();
         		std::cout << "detections times :" << timer.getAvgMillisecond() << std::endl;
        for (int i = 0; i < currentBatchImages.size(); ++i)
        {
            detections[i].m_pproducer = (CImage*)currentBatchImages[i];
            if (m_buildFeatureDataFlag)
//                __reserveBestFace(&detections[i]);
                std::cout << "__reserveBestFace " << std::endl;
 			else
 				 __outputDetection(detections[i]);
        }
        m_detectionBuffer->writeToBuffer(detections.data(), currentBatchImages.size());
    }
}

#include "Hungarian/box_tracking.h"
void CDeepViewer::__trackHead()
{
    int maxVideoCount = m_pconfiger->readValue<int>("maxVideoCount");
    std::vector<BoxTracker*> trackers(maxVideoCount, 0);
    std::vector<CDataShared*> detections;
    float iou_cost_weight = 4, cost_th = 1;
    int MAX_MISMATCH_TIMES = m_pconfiger->readValue<int>("MAX_MISMATCH_TIMES");
    std::vector<Rect> det4trcking;
    CTrack track4fame;
    std::list<CDataShared*> cacheImageDetections;
    std::vector<CFace> faces;
    while (true)
    {
        m_detectionBuffer->readData(detections, 1);
        CImageDetections& det = *(CImageDetections*)detections.front();
        CImage& frame = *(CImage*)det.m_pproducer;
        int videoId = frame.videoID;
        if (trackers[m_buildFeatureDataFlag?0:videoId] == 0)
            trackers[m_buildFeatureDataFlag ? 0:videoId] = new BoxTracker(iou_cost_weight, cost_th, MAX_MISMATCH_TIMES);
        BoxTracker& tracker = *trackers[m_buildFeatureDataFlag ? 0:videoId];

        det4trcking.resize(0);
        int numHead = det.detectionCount - det.faceCount;
        float* pbox = det.detections.data() + 6 * det.faceCount;
        for (int i = 0; i < numHead; ++i)
        {
            Rect box;
            pbox++; pbox++;
            box.x = *pbox++;
            box.y = *pbox++;
            box.width = *pbox++ - box.x;
            box.height = *pbox++ - box.y;
            det4trcking.push_back(box);
        }
        cacheImageDetections.push_front(&det);
        if (m_buildFeatureDataFlag || tracker.tracking_Frame_Hungarian(track4fame.trackingIDs, det4trcking, frame.width, frame.height))
        {
            track4fame.m_pproducer = cacheImageDetections.back();
            CImageDetections& d = *(CImageDetections*)track4fame.m_pproducer;
            faces.resize(d.faceCount);
            for (int k = 0; k < d.faceCount; ++k)
            {
                CFace& f = faces[k];
                f.m_pproducer = d.m_pproducer;
                f.confidence = d.detections[k * 6 + 1];
                f.trackId = m_buildFeatureDataFlag? videoId : track4fame.trackingIDs[k];
                memcpy(f.xyMinMax, d.detections.data() + k * 6 + 2, 4 * sizeof(float));
            }
            m_faceBuffer->writeToBuffer(faces.data(), faces.size());

            if (m_buildFeatureDataFlag)
            {
                track4fame.trackingIDs.resize(2);
                track4fame.trackingIDs.front() = videoId;
                track4fame.trackingIDs.back() = 0;
            }
            m_trackingBuffer->writeToBuffer(&track4fame, 1);
            cacheImageDetections.pop_back();
        }
    }
}

void CDeepViewer::__reserveBestFace(CDataShared* vpdetection)
{
    static int i = 0;
    CImageDetections* vp = (CImageDetections*)vpdetection;
    if (vp->detectionCount == 0)
        std::cout << m_namesFeats->getImageFileName(i) << std::endl;
    else
    {
        int dst = 0;
        float* p = vp->detections.data() + 1, maxScore = *p;
        for (int k = 1; k < vp->faceCount; ++k)
        {
            p += 6;
            if (*p > maxScore) { dst = k; maxScore = *p; }
        }
        p = vp->detections.data();
        if (dst != 0) memcpy(p, p + 6 * dst, 6 * sizeof(float));
        memcpy(p + 6, p + 6 * (vp->faceCount + dst), 6 * sizeof(float));
        vp->faceCount = 1; vp->detectionCount = 2;
        vp->detections.resize(12);
    }
    i++;
}

void CDeepViewer::__generate3DkeyPoints()
{
    std::vector<CDataShared*> currentBatch, nextBatch;
    auto preprocess = [&]()
    {
        m_faceBuffer->readData(nextBatch, m_p3DkeypointGenerator->getBatchSize());
        m_p3DkeypointGenerator->preProcessV(nextBatch);
        exitIfCudaError(cudaStreamSynchronize(m_p3DkeypointGenerator->getViceCudaStream()));
    };

    std::vector<CFaceKeyPoints> facePoints(m_p3DkeypointGenerator->getBatchSize());
    std::vector<CDataShared*> pfacePoints(facePoints.size());
    std::thread dataPreparation(preprocess);
    cudaStream_t stream = m_p3DkeypointGenerator->getCudaStream();
    CTimer timer;
    while (true)
    {
        dataPreparation.join();
         		timer.start(facePoints.size());
        currentBatch = nextBatch;
        m_p3DkeypointGenerator->inference(currentBatch.size());
        dataPreparation = std::thread(preprocess);
        m_p3DkeypointGenerator->postProcessV(currentBatch, facePoints.data());
        pfacePoints.resize(currentBatch.size());
        std::cout <<"3d keypoints" << currentBatch.size() << std::endl;
        for (int i = 0; i < currentBatch.size(); ++i)
            pfacePoints[i] = &facePoints[i];
         		timer.stop();
         		std::cout << "3d keypoints times :" << timer.getAvgMillisecond() << std::endl;
        m_faceKeyPointsBuffer->writeToBuffer(pfacePoints, stream);
        for (int i = 0; i < pfacePoints.size(); ++i)
            pfacePoints[i]->transferData(false, stream);
        exitIfCudaError(cudaStreamSynchronize(stream));
    }
}

void CDeepViewer::__calculateFaceAttribute()
{
    std::vector<CDataShared*> currentBatch, nextBatch;
    auto preprocess = [&]()
    {
        m_faceKeyPointsBuffer->readData(nextBatch, m_pfaceInforCalculator->getBatchSize());
        m_pfaceInforCalculator->preProcessV(nextBatch);
        exitIfCudaError(cudaStreamSynchronize(m_pfaceInforCalculator->getViceCudaStream()));
    };

    std::vector<CFaceInfor> facesInfor(m_pfaceInforCalculator->getBatchSize());
    std::thread dataPreparation(preprocess);
    CTimer timer;
    while (true)
    {
        dataPreparation.join();
        currentBatch = nextBatch;
        timer.start(currentBatch.size());
        m_pfaceInforCalculator->inference(currentBatch.size());
        dataPreparation = std::thread(preprocess);
        m_pfaceInforCalculator->postProcessV(currentBatch, facesInfor.data());

         		std::cout << "face Attribute times :" << timer.getAvgMillisecond() << std::endl;
        m_faceInforBuffer->writeToBuffer(facesInfor.data(), currentBatch.size());
    }
}

extern int g_recognizationFlag;
void CDeepViewer::__displayDetectionsAndFaceInfor()
{
    CTimer timer;
    CPythonCaller* pyCaller = CPythonCaller::getOrCreateInstance();
    int numFrame = 1;
    std::vector<int> ages, para;
    std::vector<char> genders;
    std::vector<CDataShared*> detectionsAndTrackId, faceInfor;
    std::vector<std::pair<void*, size_t>> buffers;
    bool flagShowImage = (m_pconfiger->readValue<int>("displayImage") > 0);
    if (m_buildFeatureDataFlag) flagShowImage == false;
    const float displayScale = flagShowImage ?  m_pconfiger->readValue<float>("displayScale") :0 ;

    int fps = m_pconfiger->readValue<int>("fps");
    int out_w = m_pconfiger->readValue<int>("out_w");
    int out_h = m_pconfiger->readValue<int>("out_h");
    std::string rtmpPath = m_pconfiger->readValue("rtmpPath");

//    ls_handler = rtmpHandler("",rtmpPath,out_w,out_h,fps);
//    ls_handler = rtmpHandler("","rtmp://127.0.0.1:1935/rtmplive/room_1",out_w,out_h,fps);

    int img_count = 0;
    std::vector<cv::Scalar>colormap = {cv::Scalar(0, 0, 128),cv::Scalar(0, 128, 0),cv::Scalar(0, 128, 128),
    cv::Scalar(128, 0, 0),cv::Scalar(128, 0, 128),cv::Scalar(128, 128, 0),cv::Scalar(128, 128, 128),cv::Scalar(0, 0, 64),cv::Scalar(0, 0, 192)};
    std::string saveDirs = m_pconfiger->readValue("saveDirs");
    if (saveDirs.empty()) saveDirs = ".";
    std::vector<float*> points0, points1;

    while (true)
    {
        std::vector<cv::Point> boxes;
        std::vector<int> boxes_f;
        img_count++;
        buffers.resize(0);
        m_trackingBuffer->readData(detectionsAndTrackId, numFrame);
        CTrack* ptrack = (CTrack*)detectionsAndTrackId.front();
        CImageDetections* pdet = (CImageDetections*)(ptrack->m_pproducer);
        CImage* pimage = (CImage*)pdet->m_pproducer;


        buffers.emplace_back((void*)0, (size_t)0);
        buffers.emplace_back(pimage->getClodData(), pimage->getClodDataSize());
        para.resize(0);
        para.push_back(pimage->width);
        para.push_back(pimage->height);
        para.push_back(pimage->videoID);
        para.push_back(pdet->faceCount);
        para.push_back(pdet->detectionCount);
        timer.start(numFrame);

        cv::Mat image(pimage->height, pimage->width, CV_8UC3, pimage->getClodData());
        if (ptrack->trackingIDs.empty()) ptrack->trackingIDs.push_back(0);
        buffers.emplace_back((void*)ptrack->trackingIDs.data(), ptrack->trackingIDs.size() * sizeof(ptrack->trackingIDs.front()));
        buffers.emplace_back((void*)pdet->detections.data(), pdet->detections.size() * sizeof(pdet->detections.front()));
//        std::cout <<  " colormap:" << cv::Scalar(255, 128, 64) << std::endl;
        float* pbox = pdet->detections.data();
        for (int i = 0;  i < pdet->detectionCount; i++)
        {//draw detections
//            std::cout <<  " colormap:" << colormap[i%9] << std::endl;
            boxes.push_back(cv::Point((int)pbox[2],(int)pbox[3]));
            boxes.push_back(cv::Point((int)pbox[4],(int)pbox[5]));
            boxes_f.push_back(0);
//            cv::rectangle(image, cv::Point((int)pbox[2], (int)pbox[3]), cv::Point((int)pbox[4], (int)pbox[5]), colormap[i%9], 2, 8, 0);
//            std::cout <<  " detections boxes:" << cv::Point((int)pbox[2],(int)pbox[3])<<cv::Point((int)pbox[4],(int)pbox[5]) << std::endl;
            pbox += 6;
        }

        int finishedFaceCount = 0;
        ages.resize(0); genders.resize(0);
        points0.resize(0); points1.resize(0);
        std::string personNames = "$__$";
        int iface = 0;
        while (finishedFaceCount < pdet->faceCount)
        {
            m_faceInforBuffer->readData(faceInfor, pdet->faceCount - finishedFaceCount);
            for (int i = 0; i < faceInfor.size(); ++i)
            {//draw faces infor
                CFaceInfor* fi = (CFaceInfor*)faceInfor[i];
                CFace* f = (CFace*)fi->m_pproducer->m_pproducer;
                CFaceKeyPoints *fk = (CFaceKeyPoints *)fi->m_pproducer;
                float* p = (float*)fk->getClodData();
                if (points0.empty() || p > points0.front()) points0.emplace_back(p);
                else  points1.emplace_back(p);
                std::string trackid = std::to_string(m_buildFeatureDataFlag?pimage->videoID: ptrack->trackingIDs[i + finishedFaceCount]);
                std::string sage = " age:" + std::to_string(fi->age);  sage.resize(9);
                std::string score = std::to_string(pdet->detections[iface * 6 + 1]);
                std::string rescore = std::to_string(fi->rescore);
                score.resize(4); rescore.resize(4);
                std::string faceInfo = trackid ;
                std::string name = m_namesFeats->getPersonName(fi->faceId);
                faceInfo +=   name;
                personNames += name + "," ;
                ages.push_back(fi->age);
                genders.push_back(fi->gender ? 1 : 0);
//                if (!flagShowImage) std::cout << faceInfo << std::endl;
//                cv::putText(image, faceInfo, cv::Point(f->xyMinMax[0]- 50, f->xyMinMax[1]-20), 2, 1.0, cv::Scalar(255, 255, 255),1);
//                std::cout <<  " face  colormap[atoi(trackid.c_str())%9]:" <<  cv::Point(f->xyMinMax[0], f->xyMinMax[1]) << std::endl;
                iface++;
//                for (int k = 0;  k < boxes.size();)
//                {//draw face detections
//                    k+=2;
////                    std::cout <<  " colormap:" << colormap[i%9] << std::endl;
//                    if (boxes[k] == cv::Point(f->xyMinMax[0], f->xyMinMax[1]))
//                    {
//                        cv::rectangle(image, boxes[k],boxes[k+1], colormap[atoi(trackid.c_str())%9], 2, 8, 0);
//                        boxes_f[k/2] = 1;
//                    }
////                    cv::rectangle(image, cv::Point((int)pbox[2], (int)pbox[3]), cv::Point((int)pbox[4], (int)pbox[5]), colormap[i%9], 2, 8, 0);
////                    std::cout <<  " detections boxes:" << cv::Point((int)pbox[2],(int)pbox[3])<<cv::Point((int)pbox[4],(int)pbox[5]) << std::endl;
//                }

            }
            finishedFaceCount += faceInfor.size();
        }
//        std::cout <<  " boxes_f size(:" << boxes_f.size()<< std::endl;
//        for (int k = 0;  k < boxes_f.size(); k++)
//        {//draw head detections
////        std::cout <<  " boxes_f:" << boxes_f[k]<< std::endl;
//            if (boxes_f[k] == 0)
//            {
//                cv::rectangle(image, boxes[2*k],boxes[2*k+1], cv::Scalar(128, 128, 128), 2, 8, 0);
//                boxes_f[k] = 1;
//            }
//        }


        if (pdet->faceCount>0)personNames.pop_back();
        buffers.emplace_back((void*)ages.data(), ages.size() * sizeof(ages.front()));
        buffers.emplace_back((void*)genders.data(), genders.size() * sizeof(genders.front()));
        buffers.emplace_back((void*)(points0.size() ? points0.front() : 0), points0.size() * 219 * sizeof(float));
        buffers.emplace_back((void*)(points1.size() ? points1.front() : 0), points1.size() * 219 * sizeof(float));
        buffers.emplace_back(&g_recognizationFlag, sizeof(g_recognizationFlag));
        buffers.front().first = (void*)para.data();
        buffers.front().second = para.size() * sizeof(para.front());
        int box_data[1000]= {0};
        int face_box_data[1000]= {0};
        int face_point_data[3000]= {0};
        int track_point_data[3000]= {0};
        buffers.emplace_back((void*)box_data, sizeof(box_data));
        buffers.emplace_back((void*)face_box_data, sizeof(face_box_data));
        buffers.emplace_back((void*)face_point_data, sizeof(face_point_data));
        buffers.emplace_back((void*)track_point_data, sizeof(track_point_data));
        std::cout <<  " submitResults:" << std::endl;
//        pyCaller->call("submitResults", buffers, saveDirs+personNames);

        for (int k = 0;  k < 1000; k+=5)
        {

            if(box_data[k] == 0 && box_data[k+1] == 0 && box_data[k+2] == 0 && box_data[k+3] == 0)
            {
                break;
            }
            else{
            cv::rectangle(image, cv::Point(box_data[k], box_data[k+1]), cv::Point(box_data[k+2], box_data[k+3]),
             colormap[int(box_data[k+4]%9)], 4, 8, 0);
            }
        }
        for (int k = 0;  k < 1000; k+=32)
        {

            if(face_box_data[k] == 0 && face_box_data[k+1] == 0 && face_box_data[k+2] == 0 && face_box_data[k+3] == 0)
            {
                break;
            }
            else{
//            std::cout <<  " draw 3d:" << std::endl;
            vector<cv::Point> pts;

            for( int t = 0;t <20; t+=2){
                pts.push_back(cv::Point(face_box_data[k +t],face_box_data[k +t +1]));
            }

            cv::polylines(image,pts,1,colormap[int(box_data[k/32*5+4]%9)],2, 8, 0);
            cv::line(image,cv::Point(face_box_data[k +20], face_box_data[k +21]),
            cv::Point(face_box_data[k +22], face_box_data[k +23]), colormap[int(box_data[k/32*5+4]%9)],2, 8, 0);
            cv::line(image,cv::Point(face_box_data[k +24], face_box_data[k +25]),
            cv::Point(face_box_data[k +26], face_box_data[k +27]), colormap[int(box_data[k/32*5+4]%9)],2, 8, 0);
            cv::line(image,cv::Point(face_box_data[k +28], face_box_data[k +29]),
            cv::Point(face_box_data[k +30], face_box_data[k +31]), colormap[int(box_data[k/32*5+4]%9)],2, 8, 0);
//            cv::rectangle(image, cv::Point(box_data[k], box_data[k+1]), cv::Point(box_data[k+2], box_data[k+3]),
//             colormap[int(box_data[k+4]%9)], 2, 8, 0);
            }
        }
        for (int k = 0;  k < 3000; k+=2)
        {

            if(face_point_data[k] == 0 && face_point_data[k+1] == 0 )
            {
                break;
            }
            else{
//            std::cout <<  " point:"<< face_point_data[k] << std::endl;
            cv::circle(image, cv::Point(face_point_data[k], face_point_data[k+1]),1 ,cv::Scalar(255, 255,255), 1,8,0);
//            cv::rectangle(image, cv::Point(box_data[k], box_data[k+1]), cv::Point(box_data[k+2], box_data[k+3]),
//             colormap[int(box_data[k+4]%9)], 2, 8, 0);
            }
        }

        int draw_tracks = 1;
        if (draw_tracks)
        {
            int num = track_point_data[0];
            int start = 1;
            for (int k = 0;  k < 2000; ){
                if (num > 2){
                    std::cout<< "k" << k ;
                    std::cout<< "num" << num ;
                    for (int j = 0;  j < num/2 - 1 ; j+=2)
                    {
                        if (abs(track_point_data[k + j +1] - track_point_data[k + j + 3])< 400 &&
                            abs(track_point_data[k + j +2] - track_point_data[k + j + 4])< 400)
                        {
                        cv::line(image,cv::Point(track_point_data[k + j +1], track_point_data[k + j+ 2]),
                            cv::Point(track_point_data[k + j + 3], track_point_data[k + j+ 4]),
                            colormap[int(1)],5, 8, 0);
                        }

                    }
                    k = k + num + 1;
                    num = track_point_data[k];
                    start + 1;
                }
                else{
                    k += 3;
                    num += track_point_data[k];
                    start  = k + 1;

                }

            }
        }


        if (flagShowImage)
        {
            static cv::Mat img;
            if (displayScale>0.01)
                cv::resize(image, img, cv::Size(0,0), displayScale, displayScale);
            else img = image;
            cv::imshow(std::to_string(pimage->videoID), img);
        }
        if (timer.getCount() == 40){
        cv::imwrite("test.jpg",image);
        }

        cv::Mat rtmp_frame;
        cv::resize(image, rtmp_frame, cv::Size(out_w, out_h));
//        ls_handler.pushRTMP(rtmp_frame);

        __makeBuffersReusable(numFrame, pdet->faceCount);

        timer.stop();
        std::cout << timer.getCount() << " time per image :" << timer.getAvgMillisecond() << std::endl << std::endl;
        if (flagShowImage) cv::waitKey(1);
    }
}

extern int g_recognizationFlag;
void CDeepViewer::__submitResults()
{
    CTimer timer;
    CPythonCaller* pyCaller = CPythonCaller::getOrCreateInstance();
    int numFrame = 1;
    std::vector<CDataShared*> detectionsAndTrackId, faceInfor;
    std::vector<std::pair<void*, size_t>> buffers;
    std::vector<int> ages, para;
    std::vector<char> genders;
    bool flagShowImage = (m_pconfiger->readValue<int>("displayImage") > 0);
    if (m_buildFeatureDataFlag) flagShowImage == false;
    std::string saveDirs = m_pconfiger->readValue("saveDirs");
    if (saveDirs.empty()) saveDirs = ".";
    std::vector<float*> points0, points1;
    while (true)
    {
        timer.start(numFrame);
        buffers.resize(0);
        m_trackingBuffer->readData(detectionsAndTrackId, numFrame);
        CTrack* ptrack = (CTrack*)detectionsAndTrackId.front();

        CImageDetections* pdet = (CImageDetections*)(ptrack->m_pproducer);
        CImage* pimage = (CImage*)pdet->m_pproducer;

        buffers.emplace_back((void*)0, (size_t)0);
        buffers.emplace_back(pimage->getClodData(), pimage->getClodDataSize());
        para.resize(0);
        para.push_back(pimage->width);
        para.push_back(pimage->height);
        para.push_back(pimage->videoID);
        para.push_back(pdet->faceCount);
        para.push_back(pdet->detectionCount);

        int headCount = pdet->detectionCount - pdet->faceCount;
        if (ptrack->trackingIDs.empty()) ptrack->trackingIDs.push_back(0);
        buffers.emplace_back((void*)ptrack->trackingIDs.data(), ptrack->trackingIDs.size() * sizeof(ptrack->trackingIDs.front()));
        buffers.emplace_back((void*)pdet->detections.data(), pdet->detections.size() * sizeof(pdet->detections.front()));

        int finishedFaceCount = 0;
        ages.resize(0); genders.resize(0);
        points0.resize(0); points1.resize(0);
        std::string personNames = "$__$";
        while (finishedFaceCount < pdet->faceCount)
        {
            m_faceInforBuffer->readData(faceInfor, pdet->faceCount - finishedFaceCount);
            for (int i = 0; i < faceInfor.size(); ++i)
            {//faces infors
                CFaceInfor* fi = (CFaceInfor*)faceInfor[i];
                CFaceKeyPoints *fk = (CFaceKeyPoints *)fi->m_pproducer;
                float* p = (float*)fk->getClodData();
                if (points0.empty() || p > points0.front()) points0.emplace_back(p);
                else  points1.emplace_back(p);
                CFace* f = (CFace*)fi->m_pproducer->m_pproducer;
                std::string name = m_namesFeats->getPersonName(fi->faceId);
                personNames += name + ",";
                ages.push_back(fi->age);
                genders.push_back(fi->gender ? 1 : 0);
            }
            finishedFaceCount += faceInfor.size();
        }

        if (pdet->faceCount>0)personNames.pop_back();
        buffers.emplace_back((void*)ages.data(), ages.size() * sizeof(ages.front()));
        buffers.emplace_back((void*)genders.data(), genders.size() * sizeof(genders.front()));
        buffers.emplace_back((void*)(points0.size() ? points0.front() : 0), points0.size() * 219 * sizeof(float));
        buffers.emplace_back((void*)(points1.size() ? points1.front() : 0), points1.size() * 219 * sizeof(float));
        buffers.emplace_back(&g_recognizationFlag, sizeof(g_recognizationFlag));
        buffers.front().first = (void*)para.data();
        buffers.front().second = para.size() * sizeof(para.front());

        pyCaller->call("submitResults", buffers, saveDirs+personNames);

        if (flagShowImage)
        {
            cv::Mat image(pimage->height, pimage->width, CV_8UC3, pimage->getClodData());
            cv::imshow(std::to_string(pimage->videoID), image);
            cv::waitKey(1);
        }
        __makeBuffersReusable(numFrame, pdet->faceCount);
        timer.stop();
        std::cout << timer.getCount() << " time per image :" << timer.getAvgMillisecond() << std::endl << std::endl;
    }
}

void CDeepViewer::__makeBuffersReusable(int vframeCount, int vfaceCount)
{
    m_faceInforBuffer->makeBufferReusable(vfaceCount);
    m_faceKeyPointsBuffer->makeBufferReusable(vfaceCount);
    m_faceBuffer->makeBufferReusable(vfaceCount);
    m_detectionBuffer->makeBufferReusable(vframeCount);
    m_trackingBuffer->makeBufferReusable(vframeCount);
    m_imagebuffer->makeBufferReusable(vframeCount);
}
