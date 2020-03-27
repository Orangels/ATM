#include "reshapePlugin.h"
#include "Common.h"

CFactory<CReshapePlugin> g_reshapeCreater("Reshape");
CReshapePlugin::CReshapePlugin()
{
}

CReshapePlugin::~CReshapePlugin()
{
}

void CReshapePlugin::_createByModelAndWeightV(const std::string& vLayerName, CCaffePrototxtReader* vpReader, const nvinfer1::Weights* vpWeightOrBias, int vNumWeightOrBias)
{
	std::vector<int> dims;
	int ilayer = vpReader->getLayerLocation(vLayerName);
	vpReader->getSequence(dims, "dim", ilayer);
	dims.resize(4, 0);
	memcpy(m_outputDim.d, dims.data() + 1, sizeof(m_outputDim.d[0] * (dims.size() - 1)));
}

nvinfer1::IPlugin* CReshapePlugin::getPlugin()
{
	return this;
}

nvinfer1::Dims CReshapePlugin::getOutputDimensions(int index, const Dims* inputs, int vNumInputArrary)
{
	size_t n = 1;
	int nOut = -1;
	assert(0 == index && 1==vNumInputArrary);
	for (int i = 0; i < m_outputDim.nbDims; ++i)
	{
		n *= inputs[0].d[i];
		if (0 == m_outputDim.d[i])
			m_outputDim.d[i] = inputs[0].d[i];
		nOut *= m_outputDim.d[i];
	}
	for (int i = 0; i < m_outputDim.nbDims; ++i)
	{
		if (-1 == m_outputDim.d[i])
		{
			m_outputDim.d[i] = n / nOut;
			break;
		}
	}
	return m_outputDim;
}

