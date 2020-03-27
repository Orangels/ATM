#pragma once
#include "caffePlugin.h"
#include <cuda_runtime_api.h>
#include "cudnn.h"

class CPoolingPlugin : public CCaffePlugin
{
public:
	CPoolingPlugin();

	~CPoolingPlugin() {}

	Dims getOutputDimensions(int index, const Dims* inputs, int vNumInputArrary) override;

	virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override;

	virtual size_t getSerializationSize() override
	{
		return  sizeof(m_dataType) + sizeof(m_poolParam) + _getBites(m_inputDim) + _getBites(m_outputDim);
	}

	virtual void serialize(void* buffer) override;

	virtual void createPluginBySerializedData(const void* vpWeights, int vWeightBiytes) override;

	virtual IPlugin* getPlugin() override;

private:
	SPooling_param m_poolParam;
	cudnnHandle_t m_cudnnHandle = NULL;
	cudnnTensorDescriptor_t   bottom_desc_, top_desc_;
	cudnnPoolingDescriptor_t  pooling_desc_;
	cudnnPoolingMode_t        mode_;
	DimsCHW m_inputDim;
	void _init();

	Weights m_kernelWeights;
	void* m_pDeviceKernel{ nullptr };
};
