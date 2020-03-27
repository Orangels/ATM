#pragma once
#include "caffePlugin.h"
#include <cuda_runtime_api.h>
#include "cudnn.h"

class CSoftmaxPlugin : public CCaffePlugin 
{
public:
	CSoftmaxPlugin() {}
	~CSoftmaxPlugin() {}
	
	virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override;

	virtual IPlugin* getPlugin() override;

private:
	cudnnHandle_t m_cudnnHandle = NULL;
	cudnnTensorDescriptor_t   m_x = NULL;	cudnnTensorDescriptor_t m_y = NULL;
	int m_InputChannels;
	Weights m_kernelWeights;
	void* m_pDeviceKernel{ nullptr };
};
