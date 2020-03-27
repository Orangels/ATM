#pragma once
#include "caffePlugin.h"

class CPermutePlugin : public CCaffePlugin
{
public:
	CPermutePlugin();	
	~CPermutePlugin();
	virtual void _createByModelAndWeightV(const std::string& vLayerName, CCaffePrototxtReader* vpReader, const nvinfer1::Weights* vpWeightOrBias, int vNumWeightOrBias) override;
	virtual void createPluginBySerializedData(const void* vpWeights, int vWeightBiytes) override;
	virtual IPlugin* getPlugin() override;

private:
	std::vector<int> m_axesOrder;
	IPlugin* m_pPlugin = NULL;
};

