#pragma once
#include "caffePlugin.h"

class CConcatPlugin : public CCaffePlugin
{
public:
	CConcatPlugin();	
	~CConcatPlugin();
	virtual void createPluginBySerializedData(const void* vpWeights, int vWeightBiytes) override;
	virtual IPlugin* getPlugin() override;

private:
	IPlugin* m_pPlugin = NULL;
};

