#include "avgChanPlugin.h"
#include "Common.h"
#include "cudaUtility.h"

CFactory<CAvgChanPlugin> avgChan("avgChan");

void CAvgChanPlugin::_createByModelAndWeightV(const std::string& vLayerName, CCaffePrototxtReader* vpReader, const nvinfer1::Weights* vpWeightOrBias, int vNumWeightOrBias)
{
	assert(vNumWeightOrBias == 0);
}


int CAvgChanPlugin::initialize()
{
	return 0;
}

void CAvgChanPlugin::terminate()
{
}


CAvgChanPlugin::CAvgChanPlugin()
{
}

nvinfer1::Dims CAvgChanPlugin::getOutputDimensions(int index, const Dims* inputs, int vNumInputArrary)
{
	assert(index == 0 && vNumInputArrary == 1 && inputs[0].nbDims == 3);
	m_inputDim.d[0] = inputs[0].d[0];
	m_inputDim.d[1] = inputs[0].d[1];
	m_inputDim.d[2] = inputs[0].d[2];
	DimsCHW m_outputDim;
	m_outputDim.d[0] = inputs[0].d[0];
	m_outputDim.d[1] = 1;
	m_outputDim.d[2] = 1;
	return m_outputDim;
}



int CAvgChanPlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
{
	avgChanCuda(batchSize, m_inputDim.c(), m_inputDim.h(), m_inputDim.w(), (float*)inputs[0], (float*)outputs[0], stream);

	return 0;
}

void CAvgChanPlugin::serialize(void* buffer)
{
	_writeDim(buffer, m_inputDim);
}

void CAvgChanPlugin::createPluginBySerializedData(const void* vpWeights, int vWeightBiytes)
{
	_readDim(vpWeights, m_inputDim);
}

nvinfer1::IPlugin* CAvgChanPlugin::getPlugin()
{
	return this;
}

