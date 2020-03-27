#include "caffePlugin.h"

bool CCaffePlugin::supportsFormat(DataType type, PluginFormat format) const
{
	return (type == m_dataType) && format == PluginFormat::kNCHW;
}

void CCaffePlugin::configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize)
{
	// 		assert((type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kNCHW);
	// 		m_dataType = type;
}

int CCaffePlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
	unsigned int m_numInputElement = batchSize;
	for (int i = 0; i < m_outputDim.nbDims; ++i)
		m_numInputElement *= m_outputDim.d[i];
	CHECK(cudaMemcpyAsync(outputs[0], inputs[0], m_numInputElement * sizeof(float), cudaMemcpyDeviceToDevice, stream));
	return 0;
}


int CCaffePlugin::initialize()
{
	return 0;
}

size_t CCaffePlugin::getSerializationSize()
{
	return  sizeof(m_dataType) + sizeof(m_outputDim.d[0]) * m_outputDim.nbDims + sizeof(m_outputDim.nbDims);
}

void CCaffePlugin::serialize(void* buffer)
{
	_write(buffer, m_dataType);
	_writeDim(buffer, m_outputDim);
}

// #include "Common.h"
CCaffePlugin::CCaffePlugin()
{
// 	CConfiger* pcon = CConfiger::getOrCreateConfiger("../configer.txt");
// 	bool useFp16 = pcon->readValue("useFp16") == "true";
// 	m_dataType = useFp16 ? DataType::kHALF : DataType::kFLOA
}

void CCaffePlugin::createPluginBySerializedData(const void* vpWeights, int vWeightBiytes)
{
	_read(vpWeights, m_dataType);
	_readDim(vpWeights, m_outputDim);
}


void CCaffePlugin::_convertAndCopyToBuffer(void*& buffer, const Weights& weights)
{
	if (weights.type != m_dataType)
		for (int64_t v = 0; v < weights.count; ++v)
			if (m_dataType == DataType::kFLOAT)
				reinterpret_cast<float*>(buffer)[v] = __half2float(static_cast<const __half*>(weights.values)[v]);
			else
				reinterpret_cast<__half*>(buffer)[v] = __float2half(static_cast<const float*>(weights.values)[v]);
	else
		memcpy(buffer, weights.values, weights.count * type2size(m_dataType));
	(char*&)buffer += weights.count * type2size(m_dataType);
}

void CCaffePlugin::_convertAndCopyToDevice(void*& deviceWeights, const Weights& weights)
{
	if (weights.type != m_dataType) // Weights are converted in host memory first, if the type does not match
	{
		size_t size = weights.count*(m_dataType == DataType::kFLOAT ? sizeof(float) : sizeof(__half));
		void* buffer = malloc(size);
		for (int64_t v = 0; v < weights.count; ++v)
			if (m_dataType == DataType::kFLOAT)
				static_cast<float*>(buffer)[v] = __half2float(static_cast<const __half*>(weights.values)[v]);
			else
				static_cast<__half*>(buffer)[v] = __float2half(static_cast<const float*>(weights.values)[v]);

		deviceWeights = copyToDevice(buffer, size);
		free(buffer);
	}
	else
		deviceWeights = copyToDevice(weights.values, weights.count * type2size(m_dataType));
}

void* CCaffePlugin::copyToDevice(const void* data, size_t count)
{
	void* deviceData;
	CHECK(cudaMalloc(&deviceData, count));
	CHECK(cudaMemcpy(deviceData, data, count, cudaMemcpyHostToDevice));
	return deviceData;
}

size_t CCaffePlugin::type2size(DataType type)
{
	return type == DataType::kFLOAT ? sizeof(float) : sizeof(__half);
}



