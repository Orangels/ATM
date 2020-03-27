#pragma once
#include <vector>
#include <string>
#include <NvInfer.h>
using namespace nvinfer1;
class PluginFactory;
class CCaffePrototxtReader;

namespace onnx
{
	class TensorShapeProto;
	class ModelProto;
}
class CConfiger;
class COnnxTrtConvertor
{
public:
	COnnxTrtConvertor(const std::string& vOnnxFileName, int vMaxBatchSize);
	~COnnxTrtConvertor();
	bool parse(INetworkDefinition* vpNetwork, DataType vNetDataType, PluginFactory* vpPluginFactory=NULL);
	int getModelBatchSize();

private:
	void __convetMention(const std::vector<int>& src, Dims& dst);
	bool __isNodeInput(const std::string& vInputName);
	void __printModelBasicInfor(const void* vpOnnxModel);
	ITensor* __squzze(INetworkDefinition*vpNet, ITensor* vSrc, int vAxis, int vN, int vOffset = -1);
	ITensor* __unsquzze(INetworkDefinition*vpNet, ITensor* vSrc, int vAxis, int vN, int vOffset = -1);
	ITensor* __squzze(INetworkDefinition*vpNet, ITensor* vSrc, std::vector<int>& vAxis, int vOffset=-1);
	ITensor* __unsquzze(INetworkDefinition*vpNet, ITensor* vSrc, std::vector<int>& vAxis, int vOffset = -1);
	PluginFactory* m_pPluginFactory;
	CConfiger * m_pConfiger;
	std::unordered_map<std::string, ITensor*> m_tensorSet;
	std::unordered_map<int, std::vector<int>> paraSet;
	std::unordered_map<int, const onnx::TensorShapeProto*> valueShapes;
	void __addMeanVar(INetworkDefinition* vpNetwork);
	std::vector<int> m_nchw;
	onnx::ModelProto* m_pOnnx_model;
	bool m_neadConvertBRG = false;
	void __markOutput(const std::string& outBlobName, INetworkDefinition* vpNetwork);
};

/*
#include "onnxTensorrtConvertor.h"
bool TensorNet::__onnx2TRTModel(const std::string& vOnnxFileName, unsigned int maxBatchSize, std::ostream& voStringStream)
{
std::cout << vOnnxFileName << std::endl;
IBuilder* builder = createInferBuilder(m_logger);
if (m_fp16) m_fp16 = false;// builder->platformHasFastFp16();
if (m_fp16)
{
builder->setFp16Mode(true);
std::cout << "use float16" << std::endl;
}
builder->setMinFindIterations(3);
builder->setAverageFindIterations(2);
builder->setMaxBatchSize(maxBatchSize);
builder->setMaxWorkspaceSize(16 << 20);
DataType modelDataType = m_fp16 ? DataType::kHALF : DataType::kFLOAT;
INetworkDefinition* network = builder->createNetwork();
COnnxTrtConvertor otConvertor(m_pCaffeReader);
otConvertor.parse(vOnnxFileName, network, modelDataType, &m_pluginFactory);

std::cout << "network --> engine..." << std::endl;
ICudaEngine* engine = builder->buildCudaEngine(*network);
m_pGieModelStream = engine->serialize();
network->destroy();
assert(engine);

if (!m_pGieModelStream) {std::cout << "failed to serialize tensorrt engine" << std::endl; return false;}
voStringStream.write((const char*)m_pGieModelStream->data(), m_pGieModelStream->size());
engine->destroy();
builder->destroy();
shutdownProtobufLibrary();
std::cout << "onnxToTRTModel Finished." << std::endl;
return true;
}

//     if (!__onnx2TRTModel("../ageGender.onnx", maxBatchSize, stdStringStream))
{
std::cout << "failed to load ageGender.onnx." << std::endl;
return 0;
}

*/