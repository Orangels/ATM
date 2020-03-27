#include <algorithm>
#include <numeric>
#include <sstream>
#include <fstream>
#include <iostream>
#include "modelEngine.h"
#include "FileFunction.h"
#include "Common.h"
#include "pluginImplement.h"
#include "caffePrototxtReader.h"
#include "cropAndResize.h"
#include "dataStructure.h"
#include "caffePlugin.h"

CModelEngine::CModelEngine()
{
	m_pluginFactory = new PluginFactory;
	m_pconfiger = CConfiger::getOrCreateConfiger();
	m_modelPath = m_pconfiger->readValue("modelPath");
	cudaStreamCreate(&cudaStream);
	cudaStreamCreate(&m_viceStream);
}

CModelEngine::~CModelEngine()
{
	delete m_pluginFactory;
	cudaStreamDestroy(cudaStream);
	cudaStreamDestroy(m_viceStream);
	for (int i = 0; i < modelIObuffers.size(); ++i)
		_cudaFree(modelIObuffers[i], true);
	_cudaFree(m_inputViceBuffer, true);
	_cudaFree((void*&)resultBufferHost, false);
}

void CModelEngine::preProcessV(std::vector<CDataShared*>& vsrc)
{
	SCropResizePara para;
	float* dst = (float*)m_inputViceBuffer;
	for (auto src : vsrc)
	{
		CImage* p = (CImage*)src;
		para.srcImage = (unsigned char*)p->getClodData(true);
		para.srcWidthBites = p->width*p->chanel;
		para.dstArray = dst;
		para.dstHeight = m_indims.h();
		para.dstWriteWidth = para.dstWidth = m_indims.w();
		
		if (m_aspectModelInput>0.1)
		{
			float srcRatio = p->width*1.f / p->height;
			if (srcRatio != m_aspectModelInput)
			{
				if (srcRatio < m_aspectModelInput)
				{
					para.dstWidth = srcRatio*para.dstHeight + 0.5;
					p->modifiedLength = m_indims.w()*1.f / para.dstWidth * p->width+0.5;
				}
				else
				{
					para.dstHeight = para.dstWidth / srcRatio + 0.5;
					p->modifiedLength = -(m_indims.h()*1.f/para.dstHeight*p->height+0.5);
				}
				cudaMemsetAsync(dst, 0, modelInputBytes, getViceCudaStream());
			}
		}

		para.xScaleS_R = p->width * 1.f / para.dstWidth;
		para.yScaleS_R = p->height* 1.f / para.dstHeight;
		para.numPixelResized = para.dstWriteWidth * para.dstHeight;
		cropAndResize(para, getViceCudaStream());
// 		CCaffePlugin::_printGPUdata(dst, m_indims.c()*m_indims.h()*m_indims.w(), getViceCudaStream(), "model gpu input");
		dst += m_indims.c()*m_indims.h()*m_indims.w();
	}
}

#include "caffePlugin.h"
void CModelEngine::inference(int vbatchSize)
{
	if (NULL == m_pContext)
	{
		m_pContext = m_pEngine->createExecutionContext();
		// 		m_pContext->setProfiler(&m_profiler);
		cudaEventCreate(&startCuda);
		cudaEventCreate(&endCuda);
		m_pluginFactory->destroyPlugin();
		m_pInfer->destroy();
	}
	std::swap(m_inputViceBuffer, modelIObuffers.front());
// 	static int itest = -1;
// 	if (m_indims.w() == 112)
// 	{
// 		itest++;
// 		int nIn = m_indims.c()*m_indims.h() *m_indims.w();
// 		for (int i = 0; i < m_batchSize; ++i)
// 			CCaffePlugin::_printGPUdata((float*)modelIObuffers[0] + i*nIn, nIn, cudaStream, std::to_string(itest) + " model gpu input");
// 	}
	if (5 == /*++*/m_iter % 50)cudaEventRecord(startCuda, cudaStream);
	m_pContext->enqueue(vbatchSize, modelIObuffers.data(), cudaStream, NULL);
// 	if (m_indims.w()==112)
// 	{
// 		int nOt = m_dimsOut.c()*m_dimsOut.h() *m_dimsOut.w();
// 		for (int i = 0; i < m_batchSize; ++i)
// 			CCaffePlugin::_printGPUdata((float*)modelIObuffers.back() + i*nOt, nOt, cudaStream, std::to_string(itest) + " model gpu otput");
// 	}
// 	std::cout << std::endl;
	if (5 == m_iter % 50)cudaEventRecord(endCuda, cudaStream);
	// 	m_pContext->execute(batchSize, modelIObuffers);
	if (5 == m_iter % 50)cudaEventSynchronize(endCuda);
	if (5 == m_iter % 50)
	{
		float totalTime = 0;
		cudaEventElapsedTime(&totalTime, startCuda, endCuda);
		std::cout << m_iter << ", cudaInferenceTime = " << totalTime << std::endl;
	}
}

bool CModelEngine::_loadNetwork(const std::string& vModelPrefix, cudaStream_t vpCudaStream /*= NULL*/)
{
	std::string cacheFileName = deleteExtentionName(vModelPrefix)+std::to_string(m_batchSize)+"." + "tensorcache";
	std::cout << "attempting to open cache file " << cacheFileName << std::endl;

	if (isExist(cacheFileName)) __creatTrtModelFromCache(cacheFileName);
	else
	{
		bool isOnnxModel = isEndWith(vModelPrefix, "onnx");
		std::cout << "cache file not found, creating tensorRT model form " << vModelPrefix << "*." << std::endl;
		bool flagConvertOk = isOnnxModel ? __onnx2TRTModel(vModelPrefix) : __caffeToTRTModel(vModelPrefix);
		if (false == flagConvertOk) { std::cout << "failed to convert model from  " << vModelPrefix << "*." << std::endl; return false; }
		__createInferenceEngine(m_pGieModelStream->data(), m_pGieModelStream->size());
	}

	if (NULL == vpCudaStream) cudaStreamCreate(&cudaStream);
	else cudaStream = vpCudaStream;
	allocateIOBuffer();
	m_aspectModelInput = (m_pconfiger->readValue<int>("originalAspect") != 0) ? m_indims.w()*1.f / m_indims.h() : 0;
	return true;
}


void CModelEngine::_cudaFree(void* &p, bool vGPU/*=true*/)
{
	if (p)
	{
		if (vGPU) cudaFree(p);
		else cudaFreeHost(p);
		p = NULL;
	}
}

bool CModelEngine::__caffeToTRTModel(std::string vCaffeModelPrefix)
{
	std::string  netFileName(vCaffeModelPrefix + ".prototxt");
	std::string  weightFileName(vCaffeModelPrefix + ".caffemodel");
	auto m_pCaffeReader = new CCaffePrototxtReader(vCaffeModelPrefix);
	if (2 > m_batchSize) m_batchSize = m_pCaffeReader->getDim()[0];
	m_pluginFactory->m_pcaffeReader = m_pCaffeReader;
	std::cout << netFileName << std::endl << weightFileName << std::endl;

	IBuilder* builder = createInferBuilder(m_logger);
	if (m_fp16) m_fp16 = builder->platformHasFastFp16();
	if (m_fp16)
	{
		//builder->setHalf2Mode(true);
		builder->setFp16Mode(true);
		std::cout << "use float16" << std::endl;
	}
	builder->setMinFindIterations(3);
	builder->setAverageFindIterations(2);
	builder->setMaxBatchSize(m_batchSize);
	builder->setMaxWorkspaceSize(32 << 20);
	ICaffeParser* caffeParser = createCaffeParser();
	caffeParser->setPluginFactory(m_pluginFactory);
	DataType modelDataType = m_fp16 ? DataType::kHALF : DataType::kFLOAT;
	INetworkDefinition* network = builder->createNetwork();
	netFileName = extractFileName(netFileName);
	weightFileName = extractFileName(weightFileName);
	const IBlobNameToTensor* blobNameToTensor = caffeParser->parse(netFileName.c_str(), weightFileName.c_str(), *network, modelDataType);
	assert(blobNameToTensor != nullptr);
	std::string outBlobName = m_pCaffeReader->getOutputBlobName();
	network->markOutput(*blobNameToTensor->find(outBlobName.c_str()));

	std::cout << "buildCudaEngineing:" << std::endl;
	ICudaEngine* engine = builder->buildCudaEngine(*network);
	assert(engine);
	network->destroy();
	caffeParser->destroy();

	m_pGieModelStream = engine->serialize();
	if (!m_pGieModelStream)
	{
		std::cout << "failed to serialize CUDA engine" << std::endl;
		return false;
	}
	__saveModelCache2File(vCaffeModelPrefix, m_pGieModelStream, m_batchSize);
	engine->destroy();
	builder->destroy();
	m_pluginFactory->destroyPlugin();
	shutdownProtobufLibrary();

	std::cout << "caffeToTRTModel Finished" << std::endl;
	return true;
}


void CModelEngine::__saveModelCache2File(const std::string& vCacheFileName, const IHostMemory* vpModelCache, int vMaxBatchSize)
{
	std::string cacheFileName = vCacheFileName+"."+std::to_string(m_batchSize) + ".tensorcache";
	std::ofstream Fout(cacheFileName, std::ios::binary);
	if (Fout.fail()) { std::cout << "Fail to wirte data to file " << cacheFileName << std::endl; return; }
	Fout.write((const char*)vpModelCache->data(), vpModelCache->size());
	Fout.write((const char*)&vMaxBatchSize, sizeof(vMaxBatchSize));
	Fout.close();
}


void CModelEngine::__creatTrtModelFromCache(const std::string& vCacheFileName)
{
	std::cout << "loading network profile from cache..." << std::endl;
	size_t modelSize = 0;
	void* pBuffer = NULL;
	readBinaryDataFromFile2Memory(vCacheFileName, pBuffer, modelSize);
	modelSize -= sizeof(m_batchSize);
	__createInferenceEngine(pBuffer, modelSize);
	memcpy(&m_batchSize, (char*)pBuffer + modelSize, sizeof(m_batchSize));
	free(pBuffer);
	std::cout << "batchSize from cache: " << m_batchSize << std::endl;
}

#include "onnxTensorrtConvertor.h"
bool CModelEngine::__onnx2TRTModel(const std::string& vOnnxFileName)
{
	std::cout << vOnnxFileName << std::endl;
	COnnxTrtConvertor otConvertor(vOnnxFileName, m_batchSize);
	if (2 > m_batchSize) m_batchSize = otConvertor.getModelBatchSize();
	IBuilder* builder = createInferBuilder(m_logger);
	if (m_fp16) m_fp16 = false;// builder->platformHasFastFp16();
	if (m_fp16)
	{
		builder->setFp16Mode(true);
		std::cout << "use float16" << std::endl;
	}
	builder->setMinFindIterations(3);
	builder->setAverageFindIterations(2);

	builder->setMaxBatchSize(m_batchSize);
	builder->setMaxWorkspaceSize(16 << 20);
	DataType modelDataType = m_fp16 ? DataType::kHALF : DataType::kFLOAT;
	INetworkDefinition* network = builder->createNetwork();
	otConvertor.parse(network, modelDataType, m_pluginFactory);

	std::cout << "network --> engine..." << std::endl;
	ICudaEngine* engine = builder->buildCudaEngine(*network);
	m_pGieModelStream = engine->serialize();
	network->destroy();
	assert(engine);

	if (!m_pGieModelStream) { std::cout << "failed to serialize tensorrt engine" << std::endl; return false; }
	__saveModelCache2File(deleteExtentionName(vOnnxFileName, false), m_pGieModelStream, m_batchSize);
	engine->destroy();
	builder->destroy();
	std::cout << "onnxToTRTModel Finished." << std::endl;
	return true;
}

void copyDimention(const Dims& src, Dims& dst)
{
	dst.nbDims = src.nbDims;
	memcpy(dst.d, src.d, src.nbDims * sizeof(src.d[0]));
	for (int i = src.nbDims; i < 3; ++i) dst.d[i] = 1;
}

float* allocateLockedCPUMemory(size_t vBities)
{
	float* ptr;
	assert(!cudaMallocHost(&ptr, vBities));
	return ptr;
}

nvinfer1::DimsCHW CModelEngine::allocateIOBuffer(size_t voBufferSize /*= 0*/)
{
	bool isDetetion = false;
	int nGPUBuffer = m_pEngine->getNbBindings();
	modelIObuffers.resize(nGPUBuffer);
	for (int b = 0; b < nGPUBuffer; b++)
	{
		Dims dimention = m_pEngine->getBindingDimensions(b);
		size_t bites = std::accumulate(dimention.d, dimention.d + dimention.nbDims, sizeof(float), std::multiplies<int>());
		if ((b + 1) == nGPUBuffer && bites < voBufferSize) bites = voBufferSize;
		bites *= m_batchSize;
		cudaMalloc(&modelIObuffers[b], bites);
		if (0 == b)
		{
			cudaMalloc(&m_inputViceBuffer, bites);
			modelInputBytes = bites / m_batchSize;
			copyDimention(dimention, m_indims);
		}
		else if ((b + 1) == nGPUBuffer)
		{
			copyDimention(dimention, m_dimsOut);
			modelOutputBytes = bites / m_batchSize;
			std::string outBlobName(m_pEngine->getBindingName(b));
			int index = outBlobName.find("detection");
			isDetetion = (index >= 0 && index < outBlobName.size());
			resultBufferHost = allocateLockedCPUMemory(bites);
			std::cout << "model output shape: " << bites << ". (" << m_batchSize << "," << m_dimsOut.c() << "," << m_dimsOut.h() << "," << m_dimsOut.w() << ")." << std::endl;
		}
	}
	return m_dimsOut;
}


void CModelEngine::__createInferenceEngine(void* vpPlanFileData, size_t vBites)
{
	m_pInfer = createInferRuntime(m_logger);
	m_pEngine = m_pInfer->deserializeCudaEngine(vpPlanFileData, vBites, m_pluginFactory);
	printf("Bindings after deserializing:\n");
	for (int bi = 0; bi < m_pEngine->getNbBindings(); bi++)
	{
		Dims dimention = m_pEngine->getBindingDimensions(bi);
		std::cout << bi << " [" << m_batchSize;
		for (int i = 0; i < dimention.nbDims; ++i) std::cout << ", " << dimention.d[i];
		std::cout << "]. ";
		if (m_pEngine->bindingIsInput(bi) == true) printf("Binding (%s): Input.\n", m_pEngine->getBindingName(bi));
		else printf("Binding (%s): Output.\n", m_pEngine->getBindingName(bi));
	}
}
