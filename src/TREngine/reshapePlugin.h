#pragma once
#include "caffePlugin.h"

class CReshapePlugin : public CCaffePlugin
{
public:
	CReshapePlugin();	
	~CReshapePlugin();
	virtual void _createByModelAndWeightV(const std::string& vLayerName, CCaffePrototxtReader* vpReader, const nvinfer1::Weights* vpWeightOrBias, int vNumWeightOrBias) override;
	virtual IPlugin* getPlugin() override;

private:
	std::vector<int> m_axesOrder;
	IPlugin* m_pPlugin = NULL;
protected:
	virtual Dims getOutputDimensions(int index, const Dims* inputs, int vNumInputArrary) override;

};

