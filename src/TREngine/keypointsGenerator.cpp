#include "keypointsGenerator.h"
#include "Common.h"

CFactory<CKeyPointsGenerator> g_keypointsCreator("CKeyPointsGenerator");

CKeyPointsGenerator::CKeyPointsGenerator()
{
	std::string keypointsModelPrefix = m_pconfiger->readValue("keypointsModelPrefix");
	_setBatchSize(m_pconfiger->readValue<int>("keypointsBatchSize"));
	_loadNetwork(m_modelPath + keypointsModelPrefix, getCudaStream());
	int bites = getBatchSize() * 4 * sizeof(float);
	cudaMalloc((void**)&m_gpuFaceBoxes, bites);
	cudaMallocHost((void**)&m_cpuFaceBoxes, bites);
	cudaMalloc((void**)&m_pointsBuffer, 219*getBatchSize()*sizeof(float));
}

CKeyPointsGenerator::~CKeyPointsGenerator()
{
	cudaFree(m_gpuFaceBoxes);
}

#include "cropAndResize.h"
#include "dataStructure.h"
void CKeyPointsGenerator::preProcessV(std::vector<CDataShared *>& vsrc)
{
	SCropResizePara para;
	float* dst = (float*)m_inputViceBuffer;
	for (int i = 0; i < vsrc.size(); ++i)
	{
		CFace* pf = (CFace*)vsrc[i];

		CImage* pi = (CImage*)pf->m_pproducer;
		para.srcImage = (unsigned char*)pi->getClodData(true);
		para.srcWidthBites = pi->width*pi->chanel;
		para.dstArray = dst;
		para.srcXoffset = int(pf->xyMinMax[0])*3;
		para.srcYoffset = pf->xyMinMax[1];
		para.dstWidth = m_indims.w();
		para.dstHeight = m_indims.h();
		para.xScaleS_R = (pf->xyMinMax[2]-pf->xyMinMax[0]) / para.dstWidth;
		para.yScaleS_R = (pf->xyMinMax[3]-pf->xyMinMax[1]) / para.dstHeight;
		para.numPixelResized = para.dstWidth * para.dstHeight;
		cropAndResize(para, getViceCudaStream());
		dst += m_indims.c()*m_indims.h()*m_indims.w();
	}
}

#include "protoIO.h"
#include "Cuda3DArray.h"
#include "cudaCommon.h"
#include "caculate3Dkeypoints.h"
#include "postPara3DKeyPoints.pb.h"
#include "caffePlugin.h"
void CKeyPointsGenerator::postProcessV(std::vector<CDataShared *>& vmodelInputes, CDataShared* voutput)
{
	float* pbox = m_cpuFaceBoxes;
	CFaceKeyPoints* dst = (CFaceKeyPoints*)voutput;
	for (int i = 0; i < vmodelInputes.size(); ++i)
	{
		CFace* f = (CFace*)vmodelInputes[i];
		memcpy(pbox, f->xyMinMax, sizeof(f->xyMinMax));
		dst->m_pproducer = f;
		pbox += 4;
		dst++;
	}

	cudaStream_t cudaStream = getCudaStream();
	cudaMemcpyAsync(m_gpuFaceBoxes, m_cpuFaceBoxes, vmodelInputes.size() * 4 * sizeof(float), cudaMemcpyHostToDevice, cudaStream);
	static cudaTextureObject_t m_shpBase, m_expBase;
	if (m_pshpBase == NULL)
	{
		cudaChannelFormatDesc floatElementFormat = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
		m_pshpBase = new CCuda3DArray(40, 204, 0, floatElementFormat);
		m_pexpBase = new CCuda3DArray(10, 204, 0, floatElementFormat);
		C3DPara para;
		CProtoIO pio;
		pio.readProtoFromBinaryFile("/srv/VisionProject/postPara3DKeyPoints.dat", &para);
// 		std::cout << "3d post " << para.wshpbase_size() << " : " << para.wshpbase().data()[0] << std::endl;
		m_pshpBase->copyData2GPUArray(para.wshpbase().data(), -1, cudaStream);
		m_pexpBase->copyData2GPUArray(para.wexpbase().data(), -1, cudaStream);
//		std::cout << para.wexpbase_size() << " : " << para.wexpbase(0) << std::endl;
		exitIfCudaError(cudaStreamSynchronize(cudaStream));
		m_shpBase = m_pshpBase->offerTextureObject(false, cudaFilterModePoint, cudaReadModeElementType);
		m_expBase = m_pexpBase->offerTextureObject(false, cudaFilterModePoint, cudaReadModeElementType);
		para.Clear();
	}
 	static int itest = 0;
// 	for (int i = 0; i < getBatchSize(); ++i)
// 		CCaffePlugin::_printGPUdata((float*)modelIObuffers.back() + i * 62, 62, cudaStream, std::to_string(i)+ "cls62");
//
	caculate3Dkeypoints((float*)modelIObuffers.back(),  m_pointsBuffer, vmodelInputes.size(), m_gpuFaceBoxes, m_shpBase, m_expBase, cudaStream);
// 	for (int i = 0; i < getBatchSize(); ++i){
//		CCaffePlugin::_printGPUdata(m_pointsBuffer + i * 219, 219, cudaStream, std::to_string(i) + "p73");
// 	}
// 	std::cout << "####" << std::endl; itest++;

	dst = (CFaceKeyPoints*)voutput;
	for (int i = 0; i < vmodelInputes.size(); ++i)
		dst++->setClodData(m_pointsBuffer+i*219, true);

}
