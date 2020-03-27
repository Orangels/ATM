#include "faceRecognization.h"
#include "Common.h"
#include "dataStructure.h"
#include "FileFunction.h"
#include "cudaCommon.h"
#include "imagesFeatures.h"
CFactory<CFaceRecognization> g_faceRecognization("CFaceRecognization");

CFaceRecognization::CFaceRecognization()
{
	std::string faceRecognizationModelPrefix = m_pconfiger->readValue("faceRecognizationModelPrefix");
	_setBatchSize(m_pconfiger->readValue<int>("faceRecognizationBatchSize"));
	_loadNetwork(m_modelPath + faceRecognizationModelPrefix, getCudaStream());
	MAX_MISMATCH_TIMES = m_pconfiger->readValue<int>("MAX_MISMATCH_TIMES");
	confidenceStep = m_pconfiger->readValue<float>("confidenceStep");
	int maxVideoCount = m_pconfiger->readValue<int>("maxVideoCount");
	ignoreThresh = m_pconfiger->readValue<float>("ignoreThresh");
	//m_namesFeatures = CImagesFeatures::getOrCreateInstance();
	m_videoFrameFaceConfidece.resize(maxVideoCount);
	_cudaFree(m_inputViceBuffer, true);
	m_system = ("system" == m_pconfiger->readValue("on_off"));
	m_developer = ("developer" == m_pconfiger->readValue("on_off"));
	if (m_system)
		splitStringAndConvert(m_pconfiger->readValue("yawPitchRol"), m_yawPitchRol, ',');
}

CFaceRecognization::~CFaceRecognization()
{
}
#include "caffePlugin.h"
void CFaceRecognization::preProcessV(std::vector<CDataShared *>& vsrc)
{
// 	cudaMemcpyAsync(m_inputViceBuffer, vsrc.back(), modelInputBytes, cudaMemcpyDeviceToDevice, getCudaStream());
// 	cudaStreamSynchronize(getCudaStream());

//	if (__ignore(vsrc.front()))
    if (0)
		vsrc.front() = NULL;
	else
		m_inputViceBuffer = vsrc.back();
}

#include "caffePlugin.h"
void CFaceRecognization::postProcessV(std::vector<CDataShared *>& vmodelInputes, CDataShared* voutput)
{
// 	CCaffePlugin::_printGPUdata((float*)modelIObuffers.front(), 112 * 112 * 3, getCudaStream(),  "recog input");
	if (1)
	{
		CFaceInfor* pi = (CFaceInfor*)voutput;
		// xxs add
		float* feat = (float*)modelIObuffers.back();
		std::vector<float> r(512);
        cudaMemcpyAsync(r.data(), feat, sizeof(float) * 512, cudaMemcpyDeviceToHost,
                            getCudaStream());
        cudaStreamSynchronize(getCudaStream());
//        batch_face_feature.clear();
        std::vector<std::vector<float>>().swap(batch_face_feature);
        batch_face_feature.push_back(r);

//        for(int j= 0;j<512;j++)
//        {
//            std::cout<<" "<<j<<": "<<r[j];
//        }

		//pi->faceId = m_namesFeatures->whoAmI((float*)modelIObuffers.back(), getCudaStream());
		if (m_system)
		{
			CFace* pface = (CFace*)pi->m_pproducer->m_pproducer;
			CImage* pimg = (CImage*)pface->m_pproducer;
			std::get<2>(m_videoFrameFaceConfidece[pimg->videoID][pface->trackId]) = pi->faceId;
		}
	}

	m_inputViceBuffer = NULL;//avoid  memory be freed twice.
}

volatile int g_recognizationFlag = 0;
bool CFaceRecognization::__ignore(CDataShared * vpFaceInfor)
{
	return true;
}
void CFaceRecognization::get_feature(std::vector<std::vector<float>>& fea)
{
    fea = batch_face_feature;
}