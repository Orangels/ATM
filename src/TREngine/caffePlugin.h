#pragma once
#include <string>
#include <iostream>
#include <assert.h>
#include "NvCaffeParser.h"
#include "NvInferPlugin.h"
#include "caffePrototxtReader.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cublas_v2.h>
#include "Common.h"
__half __float2half(float f);
float __half2float(__half h);

using namespace nvinfer1;
using namespace plugin;
using namespace nvcaffeparser1;

#define CHECK(status)                                                                                           \
    {                                                                                                                           \
        if (status != 0)                                                                                                \
        {                                                                                                                               \
            std::cout << "Cuda failure: " << cudaGetErrorString(status) \
                      << " at line " << __LINE__  << std::endl;                                                                     \
            abort();                                                                                                    \
        }                                                                                                                               \
    }

class CCaffePlugin : public IPluginExt
{
public:
	CCaffePlugin();
	virtual IPlugin* getPlugin() = 0;
	virtual void createPluginBySerializedData(const void* vpWeights, int vWeightBiytes);
	void createByModelAndWeight(const std::string& vLayerName, CCaffePrototxtReader* vpReader, const nvinfer1::Weights* vpWeightOrBias, int vNumWeightOrBias)
	{
		m_pcafferReader = vpReader;
		m_layerName = vLayerName;
		_createByModelAndWeightV(vLayerName, vpReader, vpWeightOrBias, vNumWeightOrBias);
	}

	std::string m_layerName="";
	template<typename T>
	static void _printGPUdata(const T* vpDevice, int n, cudaStream_t stream, const std::string& vInfor)
	{
		std::vector<T> r(n);
		cudaMemcpyAsync(r.data(), vpDevice, sizeof(T)*n, cudaMemcpyDeviceToHost, stream);
		cudaStreamSynchronize(stream);
		outPutSum(r.data(), n, vInfor);
// 		std::cout << vInfor << ": " << _getSum(r.data(), n) << std::endl;
	}

protected:
	size_t _getBites(const Dims& vDim) { return sizeof(vDim.nbDims) + vDim.nbDims * sizeof(vDim.d[0]);}
	void _convertAndCopyToBuffer(void*& buffer, const Weights& weights);

	DataType m_dataType{ DataType::kFLOAT};
	virtual void _createByModelAndWeightV(const std::string& vLayerName, CCaffePrototxtReader* vpReader, const nvinfer1::Weights* vpWeightOrBias, int vNumWeightOrBias) {}
	CCaffePrototxtReader* m_pcafferReader=NULL;
	DimsCHW m_outputDim;

	void terminate() override { }
	int getNbOutputs() const override { return 1; }
	size_t getWorkspaceSize(int maxBatchSize) const override { return 0; }
	bool supportsFormat(DataType type, PluginFormat format) const override;

	void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override;

	virtual Dims getOutputDimensions(int index, const Dims* inputs, int vNumInputArrary) override
	{
		assert(index == 0);
		if (inputs) 
		{
			m_outputDim.nbDims = inputs[0].nbDims;
			for (int i = 0; i < inputs[0].nbDims; ++i) m_outputDim.d[i] = inputs[0].d[i];
		}
		return m_outputDim;
	}

	virtual int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;
	int initialize() override;


	virtual size_t getSerializationSize() override;
	virtual void serialize(void* buffer) override;
	void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) override {}


	template<typename T>
	static double _getSum(T* p, int n, bool vAbs = true)
	{
		DataType m_dataType = DataType::kFLOAT;
		double sumN = 0;
		float temp = 0;
		while (n-- > 0)
		{
			if (DataType::kHALF == m_dataType)
				temp = __half2float(*(__half*)p);
			else 
				temp = *(float*)p;
			if (vAbs && temp < 0) temp *= -1;
//			ls add
//			std::cout << 219-n << "==" << temp << std::endl;
			sumN += temp;
			p++;
		}
		return sumN;
	}


	template<typename T> void _write(void*& buffer, const T& val)
	{
		*reinterpret_cast<T*>(buffer) = val;
		(const char*&)buffer += sizeof(T);
	}
	template<typename T> void _read(const void*& buffer, T& val)
	{
		val = *reinterpret_cast<const T*>(buffer);
		(const char*&)buffer += sizeof(T);
	}
	void _readDim(const void*& buffer, Dims& voDim)
	{
		_read(buffer, voDim.nbDims);
		for (int i=0; i<voDim.nbDims; ++i)
			_read(buffer, voDim.d[i]);
	}
	void _writeDim(void*& buffer, const  Dims& voDim)
	{
		_write(buffer, voDim.nbDims);
		for (int i=0; i<voDim.nbDims; ++i)
			_write(buffer, voDim.d[i]);
	}

	void deserializeToDevice(const char*& srcHostBuffer, void*& dstDeviceWeights, size_t size)
	{
		dstDeviceWeights = copyToDevice(srcHostBuffer, size);
		srcHostBuffer += size;
	}
	void _convertAndCopyToDevice(void*& deviceWeights, const Weights& weights);
	void* copyToDevice(const void* data, size_t count);
	size_t type2size(DataType type);


};

const std::string inline deleteTailNum(std::string vTypeNum)
{
	std::string type = vTypeNum.substr(0, vTypeNum.find('_'));
	return type;
}
