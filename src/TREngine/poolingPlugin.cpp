#include "poolingPlugin.h"
#include "Common.h"
#include "cudaUtility.h"

CFactory<CPoolingPlugin> poolCreator("pooling");

CPoolingPlugin::CPoolingPlugin()
{
	
}

nvinfer1::Dims CPoolingPlugin::getOutputDimensions(int index, const Dims* inputs, int vNumInputArrary)
{
	assert(index == 0 && vNumInputArrary == 1 && inputs[0].nbDims == 3);
	m_poolParam = m_pcafferReader->getPoolingParam(m_layerName);
	m_inputDim.d[0] = inputs[0].d[0];
	m_inputDim.d[1] = inputs[0].d[1];
	m_inputDim.d[2] = inputs[0].d[2];
	m_outputDim.d[0] = inputs[0].d[0];

	float width = 1 + (inputs[0].d[1] + 2 * m_poolParam.pad - m_poolParam.kernel_size) / float(m_poolParam.stride);
	float heght = 1 + (inputs[0].d[2] + 2 * m_poolParam.pad - m_poolParam.kernel_size) / float(m_poolParam.stride);

	m_outputDim.d[1] = width + m_poolParam.ceil_mode;
	m_outputDim.d[2] = heght + m_poolParam.ceil_mode;
	_init();
	return m_outputDim;
}


int CPoolingPlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
{
	if (NULL == m_cudnnHandle)
	{
		cudnnSetStream(m_cudnnHandle, stream);//CUDNN_DATA_HALF  CUDNN_DATA_FLOAT
	}
	cudnnDataType_t nnDataType = (m_dataType == DataType::kHALF) ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT;
	cudnnSetTensor4dDescriptor(bottom_desc_, CUDNN_TENSOR_NCHW, nnDataType, batchSize, m_inputDim.d[0], m_inputDim.d[1], m_inputDim.d[2]);
	cudnnSetTensor4dDescriptor(top_desc_, CUDNN_TENSOR_NCHW, nnDataType, batchSize, m_outputDim.d[0], m_outputDim.d[1], m_outputDim.d[2]);
	
	
	float onef{ 1.0f }, zerof{ 0.0f };
	void *pOne = &onef, *pZero = &zerof;
	if (m_dataType == DataType::kHALF)
	{
		__half oneh, zeroh;
		oneh = __float2half(onef);
		zeroh = __float2half(zerof);
		pOne = &oneh;
		pZero = &zeroh;
	}
// 	_printGPUdata((float*)inputs[0], m_inputDim.d[0] * m_inputDim.d[1] * m_inputDim.d[2], stream, "pool src");
	cudnnPoolingForward(m_cudnnHandle, pooling_desc_, pOne, bottom_desc_, inputs[0], pZero, top_desc_, outputs[0]);
// 	_printGPUdata((float*)outputs[0], m_outputDim.d[0] * m_outputDim.d[1] * m_outputDim.d[2], stream, "pool dst");
	return 0;
}

void CPoolingPlugin::serialize(void* buffer)
{
	void* pHostData = static_cast<char*>(buffer), *a = pHostData;
	_write(pHostData, m_dataType);
	_write(pHostData, m_poolParam);
	_writeDim(pHostData, m_inputDim);
	_writeDim(pHostData, m_outputDim);
	assert(pHostData == a + getSerializationSize());
}

void CPoolingPlugin::createPluginBySerializedData(const void* vpWeights, int vWeightBiytes)
{
	const void* pHostData = static_cast<const char*>(vpWeights), *a = pHostData;
	_read(pHostData, m_dataType);
	_read(pHostData, m_poolParam);
	_readDim(pHostData, m_inputDim);
	_readDim(pHostData, m_outputDim);
	assert(pHostData == a + vWeightBiytes);
	_init();
}

nvinfer1::IPlugin* CPoolingPlugin::getPlugin()
{
	return this;
}


void CPoolingPlugin::_init()
{
	cudnnCreate(&m_cudnnHandle);
	cudnnCreateTensorDescriptor(&bottom_desc_);
	cudnnCreateTensorDescriptor(&top_desc_);
	cudnnCreatePoolingDescriptor(&pooling_desc_);
	cudnnSetPooling2dDescriptor(pooling_desc_, (cudnnPoolingMode_t)m_poolParam.pool, (cudnnNanPropagation_t)0,
		m_poolParam.kernel_size, m_poolParam.kernel_size, m_poolParam.pad, m_poolParam.pad,
		m_poolParam.stride, m_poolParam.stride);
}

