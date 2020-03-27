#include "Common.h"
#include "hand_ssdDetector.h"
#include "sharedBuffer.h"
#include "cropAndResize.h"
#include "postDetections.h"

CFactory<hand_CSSDEngine> hand_g_ssdCreator("ssd_hand");


hand_CSSDEngine::hand_CSSDEngine()
{
	std::string detectionModelPrefix = m_pconfiger->readValue("hand_detectionModelPrefix");
	_setBatchSize(m_pconfiger->readValue<int>("hand_detectionBatchSize"));
	_loadNetwork(m_modelPath + detectionModelPrefix, getCudaStream());
}

hand_CSSDEngine::~hand_CSSDEngine()
{
}


#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
void hand_CSSDEngine::postProcessV(std::vector<CDataShared*>& vmodelInputes, CDataShared* voutput)
{
	static SFilterPara para{ 0.1,100,2 };
	if (para.area < 101)
	{
		CConfiger* pconf = CConfiger::getOrCreateConfiger();
		std::vector<float> scoreAreaAspect;
		splitStringAndConvert(pconf->readValue("hand_scoreAreaAspect"), scoreAreaAspect, ',');
		if (scoreAreaAspect.size() < 3)
			std::cout << "error: invalid scoreAreaAspect in detector configurations" << std::endl;
		else
		{
			para.confidence = scoreAreaAspect[0];
			para.area = scoreAreaAspect[1];
			para.aspect = scoreAreaAspect[2];
		}
	}
	static std::vector<int> hwidthHeight(getBatchSize() * 2, 0);
	bool flagUpload = false;
	for (int i = 0; i < vmodelInputes.size(); ++i)
	{
		CImage* pimage = (CImage*)vmodelInputes[i];
		int L = pimage->modifiedLength;
		int srcWidth = L > 0 ? L : pimage->width;
		if (srcWidth != hwidthHeight[i * 2]) { hwidthHeight[i * 2] = srcWidth; flagUpload = true; }
		int srcHight = L < 0 ? -L : pimage->height;
		if (srcHight != hwidthHeight[i * 2 + 1]) { hwidthHeight[i * 2 + 1] = srcHight; flagUpload = true; }
	}

	postDetections(vmodelInputes.size(), flagUpload? hwidthHeight.data():NULL , (float*)modelIObuffers.back(),  para, m_dimsOut.h(),  getCudaStream());
	cudaMemcpyAsync(resultBufferHost, modelIObuffers.back(), modelOutputBytes*vmodelInputes.size(), cudaMemcpyDeviceToHost, getCudaStream());

	cudaStreamSynchronize(getCudaStream());
	CImageDetections* pout = (CImageDetections*)voutput;
	for (int i = 0; i < vmodelInputes.size(); ++i)
		getheadFacePair(resultBufferHost + i*modelOutputBytes / sizeof(float), pout++);
}