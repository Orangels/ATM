#pragma once
#include "caffePlugin.h"
#include <cuda_runtime_api.h>

class CData2trtTensorPlugin : public CCaffePlugin 
{
public:
	CData2trtTensorPlugin() {}
	~CData2trtTensorPlugin() {}
	
	virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override;

	virtual size_t getSerializationSize() override
	{
		return sizeof(m_dataType) + _getBites(m_outputDim) + m_kernelWeights.count  * type2size(m_dataType);
	}

	virtual void serialize(void* buffer) override;

	virtual void createPluginBySerializedData(const void* vpWeights, int vWeightBiytes) override;

	virtual IPlugin* getPlugin() override;
	virtual void _createByModelAndWeightV(const std::string& vLayerName, CCaffePrototxtReader* vpReader, const nvinfer1::Weights* vpWeightOrBias, int vNumWeightOrBias) override;

private:

	std::vector<void*> m_outPointerBuf;
	int m_dataBites;
	Weights m_kernelWeights;
	void* m_pDeviceKernel{ nullptr };

};
