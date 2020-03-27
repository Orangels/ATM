#include "flattenPlugin.h"
#include <cuda_runtime.h>
#include "Common.h"
CFactory<CFlattenPlugin> flattenCreator("Flatten");
nvinfer1::Dims CFlattenPlugin::getOutputDimensions(int index, const Dims* inputs, int vNumInputArrary) 
{
	assert(0 == index);
	assert(1 == vNumInputArrary);
	m_outputDim.nbDims = inputs[0].nbDims;
	int m_numInputElement = 1;
	for (int i = 0; i < m_outputDim.nbDims; ++i)
	{
		m_numInputElement *= inputs[0].d[i];
		m_outputDim.d[i] = 1;
	}
	int m_axis = m_pcafferReader->getFlattenAxis(m_layerName);
	m_outputDim.d[m_axis - 1] = m_numInputElement;
	return m_outputDim;
}

nvinfer1::IPlugin* CFlattenPlugin::getPlugin()
{
	return this;
}

int CFlattenPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
	unsigned int m_numInputElement = batchSize;
	for (int i = 0; i < m_outputDim.nbDims; ++i)
		m_numInputElement *= m_outputDim.d[i];
// 	_printGPUdata((float*)inputs[0], m_numInputElement, stream, "flatten src");
	CHECK(cudaMemcpyAsync(outputs[0], inputs[0], m_numInputElement * sizeof(float), cudaMemcpyDeviceToDevice, stream));
	return 0;
}

