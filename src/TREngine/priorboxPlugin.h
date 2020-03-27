#pragma once
#include "caffePlugin.h"

class CPriorboxPlugin : public CCaffePlugin
{
public:
	CPriorboxPlugin();	
	~CPriorboxPlugin();
	virtual void _createByModelAndWeightV(const std::string& vLayerName, CCaffePrototxtReader* vpReader, const nvinfer1::Weights* vpWeightOrBias, int vNumWeightOrBias) override {}
	virtual void createPluginBySerializedData(const void* vpWeights, int vWeightBiytes) override;
	virtual IPlugin* getPlugin() override;

private:
	IPlugin* m_pPlugin = NULL;
};

