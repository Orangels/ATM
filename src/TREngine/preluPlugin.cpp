#include "preluPlugin.h"
#include "Common.h"
#include "cudaUtility.h"

CFactory<CPreluPlugin> preluCreator("prelu");

void CPreluPlugin::_createByModelAndWeightV(const std::string& vLayerName, CCaffePrototxtReader* vpReader, const nvinfer1::Weights* vpWeightOrBias, int vNumWeightOrBias)
{
	if (vNumWeightOrBias != 1)
	{
		std::cout << vLayerName << ": " << vNumWeightOrBias << std::endl;
		assert(vNumWeightOrBias == 1);
	}
	if (vNumWeightOrBias > 0)
	{
		m_kernelWeights = vpWeightOrBias[0];
		m_InputChannels = m_kernelWeights.count;
		assert(m_kernelWeights.type == DataType::kFLOAT || m_kernelWeights.type == DataType::kHALF);

		m_kernelWeights.values = malloc(m_kernelWeights.count * type2size(m_kernelWeights.type));
		memcpy(const_cast<void*>(m_kernelWeights.values), vpWeightOrBias[0].values, m_kernelWeights.count*type2size(m_kernelWeights.type));
		if (m_kernelWeights.values)
			_convertAndCopyToDevice(m_pDeviceKernel, m_kernelWeights);
	}
}


int CPreluPlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
{
// 	_printGPUdata((float*)inputs[0], batchSize* m_outputDim.c()* m_outputDim.h()* m_outputDim.w(), stream, "prelu src");
	preluCuda(batchSize, m_outputDim.c(), m_outputDim.h(), m_outputDim.w(), (float*)inputs[0], (float*)m_pDeviceKernel, (float*)outputs[0], stream);
// 	_printGPUdata((float*)outputs[0], batchSize* m_outputDim.c()* m_outputDim.h()* m_outputDim.w(), stream, "prelu dst");
	return 0;
}

void CPreluPlugin::serialize(void* buffer)
{
	void* pHostData = buffer, *a = pHostData;
	_write(pHostData, m_dataType);
	_writeDim(pHostData, m_outputDim);
	_convertAndCopyToBuffer(pHostData, m_kernelWeights);
	free((void*)m_kernelWeights.values);
	assert(pHostData == a + getSerializationSize());
}

void CPreluPlugin::createPluginBySerializedData(const void* vpWeights, int vWeightBiytes)
{
	const void* pHostData = vpWeights, *a = pHostData;
	_read(pHostData, m_dataType);
	_readDim(pHostData, m_outputDim);
	m_InputChannels = m_outputDim.d[0];
	deserializeToDevice((const char*&)pHostData, m_pDeviceKernel, m_outputDim.d[0] *type2size(m_dataType));
	assert(pHostData == a + vWeightBiytes);
}

nvinfer1::IPlugin* CPreluPlugin::getPlugin()
{
	return this;
}



