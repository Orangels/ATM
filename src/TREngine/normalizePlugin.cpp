#include "normalizePlugin.h"
#include "Common.h"
#include "cudaUtility.h"

CFactory<CNormalizePlugin> g_normalize("Normalize");

void CNormalizePlugin::_createByModelAndWeightV(const std::string& vLayerName, CCaffePrototxtReader* vpReader, const nvinfer1::Weights* vpWeightOrBias, int vNumWeightOrBias)
{
	assert(vNumWeightOrBias <= 1 && vNumWeightOrBias>=0);
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


void CNormalizePlugin::createPluginBySerializedData(const void* vpWeights, int vWeightBiytes)
{
	const void* pHostData = vpWeights, *a = pHostData;
	_read(pHostData, m_dataType);
	_readDim(pHostData, m_outputDim);
	m_InputChannels = m_outputDim.d[0];
	m_kernelWeights.count = m_InputChannels;
	m_kernelWeights.values = nullptr;
	deserializeToDevice((const char*&)pHostData, m_pDeviceKernel, m_kernelWeights.count*type2size(m_dataType));
	assert(pHostData == a + vWeightBiytes);
}

void CNormalizePlugin::serialize(void* buffer)
{
	void* pHostData = buffer, *a = pHostData;
	_write(pHostData, m_dataType);
	_writeDim(pHostData, m_outputDim);
	_convertAndCopyToBuffer(pHostData, m_kernelWeights);
	assert(pHostData == a + getSerializationSize());
	free((void*)m_kernelWeights.values);
}

size_t CNormalizePlugin::getSerializationSize()
{
	return sizeof(m_dataType) + _getBites(m_outputDim) + m_kernelWeights.count  * type2size(m_dataType);
}

CNormalizePlugin::CNormalizePlugin()
{
}


int CNormalizePlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
{
	normalizeCuda(batchSize, m_outputDim.c(), m_outputDim.h(), m_outputDim.w(), (float*)inputs[0], (float*)m_pDeviceKernel, (float*)outputs[0], stream);
	return 0;
}



nvinfer1::IPlugin* CNormalizePlugin::getPlugin()
{
	return this;
}
