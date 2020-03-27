#pragma once
#include "caffePlugin.h"

class CFlattenPlugin : public CCaffePlugin
{
public:

	CFlattenPlugin() {}
	Dims getOutputDimensions(int index, const Dims* inputs, int vNumInputArrary) override;

	virtual IPlugin* getPlugin() override;

protected:
	virtual int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

};

