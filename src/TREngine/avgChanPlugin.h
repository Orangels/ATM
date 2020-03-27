#pragma once
#include "caffePlugin.h"
class CAvgChanPlugin : public CCaffePlugin 
{
public:
	CAvgChanPlugin();

	~CAvgChanPlugin() {}

	Dims getOutputDimensions(int index, const Dims* inputs, int vNumInputArrary) override;

	virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override;

	virtual size_t getSerializationSize() override
	{
		return  _getBites(m_inputDim);
	}

	virtual void serialize(void* buffer) override;

	virtual void createPluginBySerializedData(const void* vpWeights, int vWeightBiytes) override;

	virtual IPlugin* getPlugin() override;
	virtual void _createByModelAndWeightV(const std::string& vLayerName, CCaffePrototxtReader* vpReader, const nvinfer1::Weights* vpWeightOrBias, int vNumWeightOrBias) override;


	virtual int initialize() override;
	virtual void terminate() override;

private:
	DimsCHW m_inputDim;
	SPooling_param m_poolParam;

	int m_InputChannels;
	Weights m_kernelWeights;

	void* m_pDeviceKernel{ nullptr };

};
