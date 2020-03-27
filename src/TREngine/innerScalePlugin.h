#pragma once
#include "caffePlugin.h"
#include <cuda_runtime_api.h>
class CInnerScalePlugin : public CCaffePlugin 
{
public:
	CInnerScalePlugin();

	~CInnerScalePlugin() {}

	virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override;

	virtual IPlugin* getPlugin() override;
	virtual void _createByModelAndWeightV(const std::string& vLayerName, CCaffePrototxtReader* vpReader, const nvinfer1::Weights* vpWeightOrBias, int vNumWeightOrBias) override;

private:
	int m_InputChannels;
	Weights m_kernelWeights;
	void* m_pDeviceKernel{ nullptr };
};
