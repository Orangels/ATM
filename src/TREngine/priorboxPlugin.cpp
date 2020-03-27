#include "priorboxPlugin.h"
#include "Common.h"

CFactory<CPriorboxPlugin> g_priorboxCreater("Priorbox");
CPriorboxPlugin::CPriorboxPlugin()
{
}

CPriorboxPlugin::~CPriorboxPlugin()
{
}


nvinfer1::IPlugin* CPriorboxPlugin::getPlugin()
{
	if (NULL == m_pPlugin)
	{
		PriorBoxParameters pbpara;
		int iLayer = m_pcafferReader->getLayerLocation(m_layerName);
		int iParam = m_pcafferReader->findOffset("prior_box_param", iLayer);
		float  offset, min_size, max_size;
		std::vector<float> variance;
		m_pcafferReader->getSequence(variance, "variance", iParam);
		variance.resize(4, 0);
		memcpy(pbpara.variance, variance.data(), sizeof(pbpara.variance));

		min_size = m_pcafferReader->findNum<float>("min_size", iParam, iParam + 15);
		max_size = m_pcafferReader->findNum<float>("max_size", iParam, iParam + 15);
		offset = m_pcafferReader->findNum<float>("offset", iParam, iParam + 15);
		pbpara.maxSize = &max_size;
		pbpara.minSize = &min_size;
		pbpara.numMaxSize = pbpara.numMinSize = 1;
		pbpara.offset = offset;
		std::vector<float> aspect_ratio(1, 1);
		m_pcafferReader->getSequence(aspect_ratio, "aspect_ratio", iParam);
		pbpara.aspectRatios = aspect_ratio.data();
		pbpara.numAspectRatios = aspect_ratio.size();
		if (aspect_ratio.front() == aspect_ratio[1])
		{
			pbpara.aspectRatios++;
			pbpara.numAspectRatios--;
		}
		pbpara.flip = (m_pcafferReader->findValue("flip", iParam, iParam + 15) == "true");
		pbpara.clip = (m_pcafferReader->findValue("clip", iParam, iParam + 15) == "true");
		pbpara.imgH = 0;// m_pcafferReader->getDim()[3];
		pbpara.imgW = 0;// m_pcafferReader->getDim()[2];
		int step = 0;// m_pcafferReader->findNum<int>("step", iParam, iParam + 15);
		pbpara.stepH = step;
		pbpara.stepW = step;
		m_pPlugin = createSSDPriorBoxPlugin(pbpara);
	}
	return m_pPlugin;
}

void CPriorboxPlugin::createPluginBySerializedData(const void* vpWeights, int vWeightBiytes)
{
	m_pPlugin = createSSDPriorBoxPlugin(vpWeights, vWeightBiytes);
}
