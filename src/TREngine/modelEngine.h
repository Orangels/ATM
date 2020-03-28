#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include "NvCaffeParser.h"
#include "NvInferPlugin.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;

class Logger : public ILogger
{
	void log(Severity severity, const char* msg) override
	{
		if (severity != Severity::kINFO) std::cout << msg << std::endl;
	}
};


class CConfiger;
class CDataShared;
class PluginFactory;
class CModelEngine
{
public:
	CModelEngine();
	virtual ~CModelEngine();

	virtual void preProcessV(std::vector<CDataShared*>& vsrc);
	void inference(int vbatchSize);
	virtual void postProcessV(std::vector<CDataShared*>& vmodelInputes, CDataShared* voutput)=0;
	inline cudaStream_t getCudaStream() { return cudaStream; }
	inline cudaStream_t getViceCudaStream(){ return m_viceStream; }
	inline int getBatchSize() const { return m_batchSize; }
	virtual void get_feature(std::vector<std::vector<float>>& fea){ }
	float* resultOnHost;

protected:
	bool _loadNetwork(const std::string& vModelPrefix, cudaStream_t vpCudaStream = NULL);
	void _setBatchSize(unsigned int vBatchSize) { m_batchSize = vBatchSize; }
	std::vector<void*> modelIObuffers;
	DimsCHW m_dimsOut, m_indims;
	void* m_inputViceBuffer;
	float* resultBufferHost = NULL;
	size_t modelOutputBytes = 0;
	size_t modelInputBytes = 0;
	CConfiger* m_pconfiger;
	std::string m_modelPath;
	ICudaEngine* getCudaEngine() { return m_pEngine; }
	void _cudaFree(void* &p, bool vGPU=true);
	float m_aspectModelInput;
private:
	int m_batchSize;

	IExecutionContext* m_pContext = NULL;

	cudaStream_t cudaStream;
	cudaEvent_t startCuda, endCuda;
	PluginFactory* m_pluginFactory;
	IHostMemory *m_pGieModelStream{ nullptr };
	void __createInferenceEngine(void* vpPlanFileData, size_t vBites);
	bool __caffeToTRTModel(std::string vCaffeModelPrefix);
	void __saveModelCache2File(const std::string& vCacheFileName, const IHostMemory* vpModelCache, int vMaxBatchSize);
	void __creatTrtModelFromCache(const std::string& vCacheFileName);
	bool __onnx2TRTModel(const std::string& vOnnxFileName);
	nvinfer1::DimsCHW allocateIOBuffer(size_t voBufferSize = 0);
	IRuntime* m_pInfer;
	Logger m_logger;
	ICudaEngine* m_pEngine;
	bool m_fp16 = true;
	size_t m_iter = 0;
	cudaStream_t m_viceStream;
};

