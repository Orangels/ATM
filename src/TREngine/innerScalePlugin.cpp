#include "innerScalePlugin.h"
#include "Common.h"
#include "cudaUtility.h"

CFactory<CInnerScalePlugin> innerScale("InnerScale");

void CInnerScalePlugin::_createByModelAndWeightV(const std::string& vLayerName, CCaffePrototxtReader* vpReader, const nvinfer1::Weights* vpWeightOrBias, int vNumWeightOrBias)
{
	assert(vNumWeightOrBias == 0);
}


CInnerScalePlugin::CInnerScalePlugin()
{
}


int CInnerScalePlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
{
	innerScaleCuda(batchSize, m_outputDim.c(), m_outputDim.h(), m_outputDim.w(), (float*)inputs[1], (float*)inputs[0], (float*)outputs[0], stream);
// 	_printGPUdata((float*)outputs[0], batchSize*m_outputDim.c()*m_outputDim.h()*m_outputDim.w(), stream, "innerScale result");
	return 0;
}



nvinfer1::IPlugin* CInnerScalePlugin::getPlugin()
{
	return this;
}
