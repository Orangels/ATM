#pragma once
#include "caffePlugin.h"
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cublas_v2.h>
class CNormalizePlugin : public CCaffePlugin 
{
public:
	CNormalizePlugin();

	~CNormalizePlugin() {}

	virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override;

	virtual IPlugin* getPlugin() override;
	virtual void _createByModelAndWeightV(const std::string& vLayerName, CCaffePrototxtReader* vpReader, const nvinfer1::Weights* vpWeightOrBias, int vNumWeightOrBias) override;


	virtual void createPluginBySerializedData(const void* vpWeights, int vWeightBiytes) override;

private:

	int m_InputChannels;
	Weights m_kernelWeights;
	void* m_pDeviceKernel{ nullptr };
protected:
	virtual void serialize(void* buffer) override;


	virtual size_t getSerializationSize() override;

};
