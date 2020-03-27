#pragma once
#include <vector>

//#include "rtmpHandler.h"
#include <stdio.h>
#include <time.h>

class CConfiger;
class CDataShared;
class CModelEngine;
class CSharedBuffer;
class CImagesFeatures;
class CFactoryDirectory;
class CDeepViewer
{
public:
	CDeepViewer();
	~CDeepViewer();
	void run();

private:
	void __getImages();
	void __getVideoImages();
	void __detectFaceHead();
	void __trackHead();
	void __reserveBestFace(CDataShared* vpdetection);
	void __generate3DkeyPoints();
	void __calculateFaceAttribute();
	void __displayDetectionsAndFaceInfor();
	void __submitResults();
	void __makeBuffersReusable(int vframeCount, int vfaceCount);
	CFactoryDirectory* m_pfactories;
	CModelEngine *m_pdetector, *m_p3DkeypointGenerator, *m_pfaceInforCalculator;
	CSharedBuffer *m_imagebuffer, *m_detectionBuffer, *m_faceBuffer, *m_faceKeyPointsBuffer, *m_faceInforBuffer, *m_cpuDstDetImages, *m_cpuDstFaceImages, *m_trackingBuffer;
	bool  m_buildFeatureDataFlag, m_submitReult;
	CConfiger* m_pconfiger;
	CImagesFeatures* m_namesFeats;

	//	ls add
//	rtmpHandler ls_handler;
};
