#include "permutePlugin.h"
#include "Common.h"

CFactory<CPermutePlugin> g_permuteCreater("perm");
CPermutePlugin::CPermutePlugin()
{
}

CPermutePlugin::~CPermutePlugin()
{
}

void CPermutePlugin::_createByModelAndWeightV(const std::string& vLayerName, CCaffePrototxtReader* vpReader, const nvinfer1::Weights* vpWeightOrBias, int vNumWeightOrBias)
{
	vpReader->getPermuteOrder(vLayerName, m_axesOrder);
}

nvinfer1::IPlugin* CPermutePlugin::getPlugin()
{
	if (NULL == m_pPlugin)
	{
		nvinfer1::plugin::Quadruple order;
		memcpy(order.data, m_axesOrder.data(), sizeof(int) * 4);
		m_pPlugin = createSSDPermutePlugin(order);
	}
	return m_pPlugin;
}

void CPermutePlugin::createPluginBySerializedData(const void* vpWeights, int vWeightBiytes)
{
	m_pPlugin = createSSDPermutePlugin(vpWeights, vWeightBiytes);
}
