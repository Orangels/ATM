#include "detectionOutputPlugin.h"
#include "Common.h"

CFactory<CDetectionOutputPlugin> g_DetectionOutputCreater("DetectionOutput");
CDetectionOutputPlugin::CDetectionOutputPlugin()
{
}

CDetectionOutputPlugin::~CDetectionOutputPlugin()
{
}


nvinfer1::IPlugin* CDetectionOutputPlugin::getPlugin()
{
	if (NULL == m_pPlugin)
	{
		DetectionOutputParameters params;
		int iLayer = m_pcafferReader->getLayerLocation(m_layerName);
		int iParam = m_pcafferReader->findOffset("detection_output_param", iLayer);

		params.backgroundLabelId = m_pcafferReader->findNum<int>("background_label_id", iParam, iParam+15);
		params.codeType = CodeTypeSSD::CENTER_SIZE;
		params.keepTopK = m_pcafferReader->findNum<int>("keep_top_k", iParam, iParam+15);
		params.shareLocation = ("true" == m_pcafferReader->findValue("share_location", iParam, iParam+15));
		
		params.varianceEncodedInTarget = false;
		params.topK = m_pcafferReader->findNum<int>("top_k", iParam, iParam + 15);
		params.nmsThreshold = m_pcafferReader->findNum<float>("nms_threshold", iParam, iParam + 15);
		params.numClasses = m_pcafferReader->findNum<int>("num_classes", iParam, iParam + 15);
		params.inputOrder[0] = 0; 
		params.inputOrder[1] = 1;
		params.inputOrder[2] = 2;
		params.confidenceThreshold = m_pcafferReader->findNum<float>("confidence_threshold", iParam, iParam + 15);
		params.confSigmoid = false; 
		params.isNormalized = true;
		m_pPlugin = createSSDDetectionOutputPlugin(params);
	}
	return m_pPlugin;
}

void CDetectionOutputPlugin::createPluginBySerializedData(const void* vpWeights, int vWeightBiytes)
{
	m_pPlugin = createSSDDetectionOutputPlugin(vpWeights, vWeightBiytes);
}
