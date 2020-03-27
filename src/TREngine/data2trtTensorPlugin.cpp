#include "data2trtTensorPlugin.h"
#include "Common.h"
#include "cudaUtility.h"
#include <algorithm>
#include <numeric>

CFactory<CData2trtTensorPlugin> data2trtTensor("data2trtTensor");

void CData2trtTensorPlugin::_createByModelAndWeightV(const std::string& vLayerName, CCaffePrototxtReader* vpReader, const nvinfer1::Weights* vpWeightOrBias, int vNumWeightOrBias)
{
	assert(vNumWeightOrBias == 2);
	m_kernelWeights = vpWeightOrBias[0];
	assert(m_kernelWeights.type == DataType::kFLOAT || m_kernelWeights.type == DataType::kHALF);
	const nvinfer1::Weights& shapeDim = vpWeightOrBias[1];
	m_outputDim.nbDims = shapeDim.count;
	memcpy(m_outputDim.d, shapeDim.values, sizeof(int)*shapeDim.count);

	m_kernelWeights.values = malloc(m_kernelWeights.count * type2size(m_kernelWeights.type));
	memcpy(const_cast<void*>(m_kernelWeights.values), vpWeightOrBias[0].values, m_kernelWeights.count*type2size(m_kernelWeights.type));


	if (m_kernelWeights.values)
		_convertAndCopyToDevice(m_pDeviceKernel, m_kernelWeights);
}


int CData2trtTensorPlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
{
// 	_printGPUdata((float*)m_pDeviceKernel, m_dataBites / 4, stream, "data plugin src: ");

	if (m_outPointerBuf.size() < batchSize) m_outPointerBuf.resize(batchSize, NULL);
	for (int i = 0; i < batchSize; ++i)
	{
		if (outputs[i] != m_outPointerBuf[i])
		{
			m_outPointerBuf[i] = outputs[i];//todo fpl6? type2size(m_dataType)
			cudaMemcpyAsync(outputs[i], m_pDeviceKernel, m_dataBites, cudaMemcpyDeviceToDevice, stream);
			//std::cout << "data2trtTensor" << std::endl;
		}
	}
// 	_printGPUdata((float*)outputs[0], m_dataBites / 4, stream, convert2String(m_dataBites)+"data plugin dst: " );
	return 0;
}

void CData2trtTensorPlugin::serialize(void* buffer)
{
	void* pHostData = buffer, *a = pHostData;
	_write(pHostData, m_dataType);
	_writeDim(pHostData, m_outputDim);

// 	float* p = (float*)m_kernelWeights.values;
// 	float sum = std::accumulate(p, p + m_kernelWeights.count, 0.f, [](float& f1, float& f2) { return std::abs(f1) + std::abs(f2); });
// 	std::cout << "serialize sum: " << sum << std::endl;

	_convertAndCopyToBuffer(pHostData, m_kernelWeights);
	assert(pHostData == a + getSerializationSize());
	free((void*)m_kernelWeights.values);
}

void CData2trtTensorPlugin::createPluginBySerializedData(const void* vpWeights, int vWeightBiytes)
{
	const void* pHostData = vpWeights, *a = pHostData;
	_read(pHostData, m_dataType);
	_readDim(pHostData, m_outputDim);
	m_dataBites = std::accumulate(m_outputDim.d, m_outputDim.d + m_outputDim.nbDims, 1, std::multiplies<int>())*type2size(m_dataType);
	float* p = (float*)pHostData;
	float sum  = getSum(p, m_dataBites / 4);
	std::cout << m_dataBites/4 << "data sum: " << sum << std::endl;
	deserializeToDevice((const char*&)pHostData, m_pDeviceKernel, m_dataBites);
	assert(pHostData == a + vWeightBiytes);
}

nvinfer1::IPlugin* CData2trtTensorPlugin::getPlugin()
{
	return this;
}



