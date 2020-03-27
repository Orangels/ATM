#include "concatPlugin.h"
#include "Common.h"

CFactory<CConcatPlugin> g_concatCreater("Concat");
CConcatPlugin::CConcatPlugin()
{
}

CConcatPlugin::~CConcatPlugin()
{
}


nvinfer1::IPlugin* CConcatPlugin::getPlugin()
{
	if (NULL == m_pPlugin)
	{
		int ilayer = m_pcafferReader->getLayerLocation(m_layerName);
		int m_axis = m_pcafferReader->findNum<int>("axis", ilayer, ilayer + 15);
		if (0 > m_axis)m_axis = 1;
		m_pPlugin = createConcatPlugin(m_axis, false);
	}
	return m_pPlugin;
}


void CConcatPlugin::createPluginBySerializedData(const void* vpWeights, int vWeightBiytes)
{
	m_pPlugin = createConcatPlugin(vpWeights, vWeightBiytes);
}
