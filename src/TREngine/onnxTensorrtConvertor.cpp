#include <iostream>
#include <algorithm>
#include "load_onnx.hpp"
#include "onnxTensorrtConvertor.h"
#include "pluginImplement.h"
#include "caffePrototxtReader.h"
#include "pythonCaller.h"
#include "protoIO.h"
#include "Common.h"

COnnxTrtConvertor::COnnxTrtConvertor(const std::string& vOnnxFileName, int vMaxBatchSize)
{
	CProtoIO pio;
	m_pOnnx_model = new onnx::ModelProto;
	pio.readProtoFromBinaryFile(vOnnxFileName, m_pOnnx_model);
	const onnx::GraphProto& graph = m_pOnnx_model->graph();
	const ::onnx::ValueInfoProto& data = graph.input(0);
	auto& d = data.type().tensor_type().shape();
	m_nchw.resize(data.type().tensor_type().shape().dim_size());
	std::cout << "nchw: ";
	for (int i = 0; i < m_nchw.size(); ++i)
	{
		m_nchw[i] = d.dim(i).dim_value();
		std::cout << m_nchw[i] << ", ";
	}std::cout << std::endl;
}

COnnxTrtConvertor::~COnnxTrtConvertor()
{
	delete m_pOnnx_model;
}

template<typename T>
const std::vector<int>& _convertWeight(const instant::array& src, nvinfer1::Weights& dst, DataType vModelDataType, bool vShow = false, bool vConverBRG=false)
{
	dst.type = vModelDataType;
	dst.count = std::accumulate(src.dims().cbegin(), src.dims().cend(), 1, std::multiplies<int>());
	dst.values = src.data();	
	if (vShow) 
		outPutSum((float*)dst.values, dst.count, "weightBais");
	if (vConverBRG)
	{
		const int n = src.dims()[0];
		if (n > 1)
		{
			const int c = src.dims()[1];
			char* pSrc = (char*)src.data();
			size_t batchBites = dst.count / n * sizeof(T);
			int chanelBites = batchBites / c;
			void* pTemp = malloc(chanelBites);
			for (int i = 0; i < n; ++i)
			{
				for (int k = 0; k < c / 2; ++k)
				{
					memcpy(pTemp, pSrc + k*chanelBites, chanelBites);
					memcpy(pSrc + k*chanelBites, pSrc + (c - 1 - k)*chanelBites, chanelBites);
					memcpy(pSrc + (c - 1 - k)*chanelBites, pTemp, chanelBites);
				}
				pSrc += batchBites;
			}
			free(pTemp);
		}
	}
	return src.dims();
}

const onnx::AttributeProto& _getAttribute(const ::onnx::NodeProto& vNode, const std::string& vAttrName)
{
	for (int i = 0; i < vNode.attribute_size(); ++i)
	{
		const auto& att = vNode.attribute(i);
		if (vAttrName == att.name())
			return att;
	}
	return vNode.attribute(0);
}
template<typename T>
void _getAtrriValues(const onnx::AttributeProto& vAttri, std::vector<T>& voValues)
{
	voValues.resize(0); 
// 	if (vAttri.has_i())
	{
		voValues.reserve(vAttri.ints_size());
		for (int i = 0; i < vAttri.ints_size(); ++i) voValues.push_back(vAttri.ints(i));
	}
}

#include "StringFunction.h"
void COnnxTrtConvertor::__addMeanVar(INetworkDefinition* vpNetwork)
{
	static  std::vector<float> mean,stds;
	std::string sMean = m_pConfiger->readValue("mean"),
		sVar = m_pConfiger->readValue("varience");
	splitStringAndConvert(sMean, mean, ',');
	splitStringAndConvert(sVar, stds, ',');
	for (int i = 0; i < 3; ++i)
	{
		stds[i] = 1 / stds[i];
		mean[i] = -mean[i] * stds[i];
	}
	m_tensorSet["-1"] = m_tensorSet["0"];
	static Weights shift{ DataType::kFLOAT, mean.data(), 3 }, 
		scale{ DataType::kFLOAT, stds.data(), 3 }, 
		power{ DataType::kFLOAT, nullptr, 0 };
	ITensor* pTensor = m_tensorSet["-1"];
	nvinfer1::ScaleMode mode = nvinfer1::ScaleMode::kCHANNEL;
	IScaleLayer* pLayer = vpNetwork->addScale(*pTensor, mode, shift, scale, power);
	m_tensorSet["0"] = pLayer->getOutput(0);
}


void COnnxTrtConvertor::__markOutput(const std::string& outBlobName, INetworkDefinition* vpNetwork)
{
	static int i = 0;
	auto blob = m_tensorSet[outBlobName];
	std::cout << i++ << " onnx output array: [" << m_nchw.front();
	auto dim = blob->getDimensions();
	for (int i = 0; i < dim.nbDims; ++i)std::cout << ", " << dim.d[i];
	std::cout << "], " << outBlobName << std::endl;
	blob->setName(outBlobName.c_str());
	vpNetwork->markOutput(*blob);
}

bool COnnxTrtConvertor::parse(INetworkDefinition* vpNetwork, DataType vNetDataType, PluginFactory* vpPluginFactory/*=NULL*/)
{
	if (NULL != vpPluginFactory) m_pPluginFactory = vpPluginFactory;
	m_pConfiger = CConfiger::getOrCreateConfiger();
	m_neadConvertBRG = (0 != m_pConfiger->readValue<int>("neadConvertBRG"));
	onnx::ModelProto& onnx_model = *m_pOnnx_model;
	__printModelBasicInfor(&onnx_model);
	const onnx::GraphProto& graph = onnx_model.graph();
	static std::unordered_map<std::string, instant::array> tableWeights = instant::make_parameter_table(graph);
	instant::getDataShape(graph, valueShapes);
	const auto& inputShape = valueShapes[0];
	m_tensorSet["0"] = vpNetwork->addInput("data", vNetDataType, Dims3{ m_nchw[1], m_nchw[2], m_nchw[3] });
	if (m_pConfiger->readValue<int>("withMeanVar")!=0)
		__addMeanVar(vpNetwork);
	ITensor* pLast = m_tensorSet["0"];
	DataType wtType = DataType::kFLOAT;
	CPythonCaller* pyCaller = CPythonCaller::getOrCreateInstance();
	std::vector<int> debugLayer;
	splitStringAndConvert(m_pConfiger->readValue("debugLayer"), debugLayer, ',');
	int numLayer =  m_pConfiger->readValue<int>("numLayer");
	int i = 0;
	for (; i < graph.node_size()&&i<debugLayer.back(); ++i)
	{
		const ::onnx::NodeProto& node = graph.node(i);
		std::string outputDataName = node.output(0);
		std::string inputDataName0 = node.input_size()>0 && i>0 ? node.input(0) : "0";
		std::cout << i << ", " << node.op_type() << ", " << node.name() << std::endl;
		if (node.op_type() == "Conv" )
		{
			nvinfer1::Weights weight, bias;
			auto wtShape = _convertWeight<float>(tableWeights[node.input(1)], weight, wtType, i>=debugLayer.front()&&i<debugLayer.back(), m_neadConvertBRG);
			if (node.input_size() != 2)
				_convertWeight<float>(tableWeights[node.input(2)], bias, wtType, i>=debugLayer.front()&&i<debugLayer.back(), m_neadConvertBRG);
			else 
				memset(&bias, 0, sizeof(bias));

			if (m_neadConvertBRG) m_neadConvertBRG = false;
			IConvolutionLayer* pLayer = vpNetwork->addConvolution(*m_tensorSet[inputDataName0], wtShape[0], DimsHW{ wtShape[2], wtShape[3] }, weight, bias);
			const auto& group = _getAttribute(node, "group");
			pLayer->setNbGroups(group.i());
			const auto& strides = _getAttribute(node, "strides");
			pLayer->setStride(DimsHW{ strides.ints(0), strides.ints(1) });
			const auto& pads = _getAttribute(node, "pads");
			pLayer->setPadding(DimsHW{ pads.ints(0), pads.ints(1) });
			const auto&dilation = _getAttribute(node, "dilation");
			pLayer->setDilation(DimsHW{ dilation.ints(0), dilation.ints(1) });
			m_tensorSet[outputDataName] = pLast = pLayer->getOutput(0);

		}
		else if (node.op_type() == "ConvTranspose")
		{
			nvinfer1::Weights weight, bias;
			auto wtShape = _convertWeight<float>(tableWeights[node.input(1)], weight, wtType, i>=debugLayer.front()&&i<debugLayer.back());
			if (node.input_size() != 2)
				_convertWeight<float>(tableWeights[node.input(2)], bias, wtType, i>=debugLayer.front()&&i<debugLayer.back());
			else memset(&bias, 0, sizeof(bias));
			IDeconvolutionLayer* pLayer = vpNetwork->addDeconvolution(*m_tensorSet[inputDataName0], wtShape[0], DimsHW{ wtShape[2], wtShape[3] }, weight, bias);
			const auto& group = _getAttribute(node, "group");
			pLayer->setNbGroups(group.i());
			const auto& strides = _getAttribute(node, "strides");
			pLayer->setStride(DimsHW{ strides.ints(0), strides.ints(1) });
			const auto& pads = _getAttribute(node, "pads");
			pLayer->setPadding(DimsHW{ pads.ints(0), pads.ints(1) });
			const auto& kernelSize = _getAttribute(node, "kernel_shape");
			pLayer->setKernelSize(DimsHW{ kernelSize.ints(0), kernelSize.ints(1) });
			m_tensorSet[outputDataName] = pLast = pLayer->getOutput(0);
		}
		else if (node.op_type() == "BatchNormalization")
		{
			instant::array wegt = tableWeights[node.input(1)];
			instant::array bias = tableWeights[node.input(2)];
			instant::array mean = tableWeights[node.input(3)];
			instant::array vari = tableWeights[node.input(4)];
			const auto& epsilon = _getAttribute(node, "epsilon");
			int nChanel = std::accumulate(wegt.dims().cbegin(), wegt.dims().cend(), 1, std::multiplies<int>());
			float *pW = (float*)wegt.data(), *pV = (float*)vari.data();
			float *pM = (float*)mean.data(), *pB = (float*)bias.data();
			float eps = epsilon.f();
			if (i>=debugLayer.front()&&i<debugLayer.back())
			{
				std::cout << " BatchNormalization: eps=" << eps << ", numChanel=" << nChanel << std::endl;
				outPutSum((float*)wegt.data(), std::accumulate(wegt.dims().cbegin(), wegt.dims().cend(), 1, std::multiplies<int>()), "weight");
				outPutSum((float*)bias.data(), std::accumulate(bias.dims().cbegin(), bias.dims().cend(), 1, std::multiplies<int>()), "bias");
				outPutSum((float*)mean.data(), std::accumulate(mean.dims().cbegin(), mean.dims().cend(), 1, std::multiplies<int>()), "mean");
				outPutSum((float*)vari.data(), std::accumulate(vari.dims().cbegin(), vari.dims().cend(), 1, std::multiplies<int>()), "vari");
			}
			for (int i = 0; i < nChanel; ++i)
			{
				pW[i] /= sqrt(pV[i] + eps);//1e-5
				pB[i] -= pW[i] * pM[i];
			}

			Weights shift, scale, power{ DataType::kFLOAT, nullptr, 0 };
			_convertWeight<float>(wegt, scale, wtType);
			_convertWeight<float>(bias, shift, wtType);
			ITensor* pTensor = m_tensorSet[inputDataName0];
			nvinfer1::ScaleMode mode = nvinfer1::ScaleMode::kCHANNEL;
			int srcDimNum = pTensor->getDimensions().nbDims;
			int diffDim = 3 - srcDimNum;
			if (diffDim > 0) pTensor = __unsquzze(vpNetwork, pTensor, srcDimNum, diffDim, 0);
			IScaleLayer* pLayer = vpNetwork->addScale(*pTensor, mode, shift, scale, power);
			pTensor = pLayer->getOutput(0);
			if (diffDim>0) pTensor = __squzze(vpNetwork, pTensor, srcDimNum, diffDim, 0);
			m_tensorSet[outputDataName] = pLast = pTensor;
		}
		else if (node.op_type() == "Relu")
		{
			IActivationLayer* pLayer = vpNetwork->addActivation(*m_tensorSet[inputDataName0], ActivationType::kRELU);
			m_tensorSet[outputDataName] = pLast = pLayer->getOutput(0);
		}
		else if (node.op_type() == "PRelu")
		{
			nvinfer1::Weights weight;
			auto wtShape = _convertWeight<float>(tableWeights[node.input(1)], weight, wtType);
			std::string preluLayerName = "prelu_" + std::to_string(i);
			nvinfer1::IPluginExt* pp = (nvinfer1::IPluginExt*)m_pPluginFactory->createPlugin(preluLayerName.c_str(), &weight, 1);
			IPluginLayer* pLayer = vpNetwork->addPluginExt((ITensor* const*)&m_tensorSet[inputDataName0], 1, *pp);
			pLayer->setName(preluLayerName.c_str());
			m_tensorSet[outputDataName] = pLast = pLayer->getOutput(0);
		}
		else if (node.op_type() == "Add")
		{
			std::string  inputDataName1 = node.input(1);
			IElementWiseLayer* pLayer = vpNetwork->addElementWise(*m_tensorSet[inputDataName0], *m_tensorSet[inputDataName1], ElementWiseOperation::kSUM);
			m_tensorSet[outputDataName] = pLast = pLayer->getOutput(0);
		}
		else if (node.op_type() == "Constant")
		{
			const auto& data = _getAttribute(node, "value");			
			auto& dst = paraSet[convertFromString<int>(node.output(0))];
			if (data.has_t())
			{
				if ((instant::dtype_t)data.t().data_type() == instant::dtype_t::int64)
				{
					std::vector<long long> d(data.t().raw_data().size() / 8);
					memcpy(d.data(), data.t().raw_data().c_str(), data.t().raw_data().size());
					dst.reserve(d.size());
					for (int i = 0; i < d.size(); ++i)
						dst.push_back(d[i]);
				}
				else if ((instant::dtype_t)data.t().data_type() == instant::dtype_t::float_)
				{
					dst.resize(data.t().raw_data().size() / 4);
					memcpy(dst.data(), data.t().raw_data().c_str(), data.t().raw_data().size());
				}
			}
		}
		else if (node.op_type() == "Shape")
		{
			int iInput = convertFromString<int>(node.input(0));
			auto& dstShape = paraSet[convertFromString<int>(node.output(0))];
			const auto& srcShap = valueShapes[iInput];
			dstShape.reserve(srcShap->dim_size());
			for (int s = 0; s < srcShap->dim_size(); ++s)
				dstShape.push_back(srcShap->dim(s).dim_value());
		}
		else if (node.op_type() == "Softmax")
		{
			ITensor* pt = m_tensorSet[inputDataName0];
			ISoftMaxLayer* pLayer = vpNetwork->addSoftMax(*pt);
			const auto& axis = _getAttribute(node, "axis");
			pLayer->setAxes(axis.i()-1);
			m_tensorSet[outputDataName] = pLast = pLayer->getOutput(0);
		}
		else if (node.op_type() == "ReduceSum")
		{
			const auto& axes = _getAttribute(node, "axes");
			int axis = axes.ints(0);
			if (axis == -1) 
				axis = m_tensorSet[inputDataName0]->getDimensions().nbDims;
			const auto& keepdims = _getAttribute(node, "keepdims");
			IReduceLayer* pLayer = vpNetwork->addReduce(*m_tensorSet[inputDataName0], ReduceOperation::kSUM, axis, keepdims.i());
			m_tensorSet[outputDataName] = pLast = pLayer->getOutput(0);
		}
		else if (node.op_type() == "Gather")
		{
			int iShape = convertFromString<int>(node.input(0));
			int iIndex = convertFromString<int>(node.input(1));
			if (paraSet.find(iShape) != paraSet.cend())
			{
				auto& shapeDim = paraSet[iShape];
				auto& indexSet = paraSet[iIndex];
				auto& dst = paraSet[convertFromString<int>(node.output(0))];
				dst.reserve(indexSet.size());
				for (int i = 0; i < indexSet.size(); ++i)
					dst.push_back(shapeDim[indexSet[i]]);
			}
		}
		else if (node.op_type() == "Unsqueeze")
		{
			auto&  axes = _getAttribute(node, "axes");
			std::vector<int> indexSet;
			_getAtrriValues(axes, indexSet);
			int iShape = convertFromString<int>(node.input(0));
			if (paraSet.find(iShape) != paraSet.cend())
			{
				auto shapeDim = paraSet[iShape];
				//for (auto& i : indexSet) shapeDim.insert(shapeDim.begin() + i, 1);
				paraSet[convertFromString<int>(node.output(0))] = shapeDim;
			}
			else
			{
				ITensor* pTensor = m_tensorSet[inputDataName0];
				pTensor = __unsquzze(vpNetwork, pTensor, indexSet,  -1);
 				m_tensorSet[outputDataName] = pLast = pTensor;
			}
		}
		else if (node.op_type() == "Squeeze")
		{
			auto&  axes = _getAttribute(node, "axes");
			std::vector<int> indexSet;
			_getAtrriValues(axes, indexSet);
			int iShape = convertFromString<int>(node.input(0));
			if (paraSet.find(iShape) == paraSet.cend())
			{
				ITensor* pTensor = m_tensorSet[inputDataName0];
				pTensor = __squzze(vpNetwork, pTensor, indexSet, -1);
				m_tensorSet[outputDataName] = pLast = pTensor;
			}
		}
		else if (node.op_type() == "MatMul")
		{
			ITensor* pm1 = NULL;
			int i1Input = convertFromString<int>(node.input(1));
			if (paraSet.find(i1Input) != paraSet.cend())
			{
				const std::vector<int>& src = paraSet[i1Input];
				std::string layerName = "data2trtTensor_" + std::to_string(i);
				Weights dst;
				dst.count = src.size();
				dst.values = src.data();
				nvinfer1::IPluginExt* pp = (nvinfer1::IPluginExt*)m_pPluginFactory->createPlugin(layerName.c_str(), NULL, 1);
				IPluginLayer* pLayer = vpNetwork->addPluginExt(NULL, 0, *pp);
				pLayer->setName(layerName.c_str());
				pm1 = pLayer->getOutput(0);
			}
			else pm1 = m_tensorSet[node.input(1)];
			ITensor* pA = m_tensorSet[inputDataName0];
			pA = __unsquzze(vpNetwork, pA, 0, 1, 0);
			IMatrixMultiplyLayer* pLayer = vpNetwork->addMatrixMultiply(*pA, false, *pm1, false);
			m_tensorSet[outputDataName] = pLast = __squzze(vpNetwork, pLayer->getOutput(0), 0, 1, 0);
		}
		else if (node.op_type() == "Mul")
		{
			int i1Input = convertFromString<int>(node.input(1));
			Weights shift{ DataType::kFLOAT, nullptr, 0 }, scale, power{ DataType::kFLOAT, nullptr, 0 };
			auto& ratios = paraSet[i1Input];
			scale.count = ratios.size();
			scale.values = ratios.data();
			scale.type = DataType::kFLOAT;
			ITensor* pTensor = m_tensorSet[inputDataName0];
			int srcDimNum = pTensor->getDimensions().nbDims;
			int diffDim = 3 - srcDimNum;
			if (diffDim > 0) pTensor = __unsquzze(vpNetwork, pTensor, srcDimNum, diffDim, 0);
			IScaleLayer* pLayer =vpNetwork->addScale(*pTensor, ScaleMode::kELEMENTWISE, shift, scale, power);
			//pLayer->setPrecision(wtType);
			pTensor = pLayer->getOutput(0);
			if (diffDim > 0) pTensor = __squzze(vpNetwork, pTensor, srcDimNum, diffDim, 0);
			m_tensorSet[outputDataName] = pLast = pTensor;
		}
		else if (node.op_type() == "Concat")
		{
			int iInput = convertFromString<int>(node.input(0));
			if (paraSet.find(iInput) != paraSet.cend())
			{
				auto& dst = paraSet[convertFromString<int>(node.output(0))];
				for (int i=0; i<node.input_size(); ++i)
				{
					int iInput = convertFromString<int>(node.input(i));
					auto& srcSet = paraSet[iInput];
					for (auto& src : srcSet) dst.push_back(src);
				}
			}
			else
			{
				std::vector<ITensor*> conTensors(node.input_size());
				for (int i = 0; i < node.input_size(); ++i)
					conTensors[i] = m_tensorSet[node.input(i)];
				
				IConcatenationLayer* pLayer = vpNetwork->addConcatenation(conTensors.data(), conTensors.size());
				const auto& axisAttr = _getAttribute(node, "axis");
				pLayer->setAxis(axisAttr.i()-1);
				m_tensorSet[outputDataName] = pLast = pLayer->getOutput(0);
			}
		}
		else if (node.op_type() == "Reshape")
		{//trt dims excluding batch  infor
			IShuffleLayer* pLayer = vpNetwork->addShuffle(*m_tensorSet[inputDataName0]);

			Dims dim;
			int iShape = convertFromString<int>(node.input(1));
			std::vector<int> shapeDim;
			if (iShape > 0)
				shapeDim = paraSet[iShape]; 
			else 
			{
				nvinfer1::Weights sdata; 
				shapeDim = _convertWeight<long long>(tableWeights[node.input(1)], sdata, wtType);
				int dim = shapeDim.front();
				shapeDim.resize(0);
				long long* p = (long long*)sdata.values;
				while (dim--) shapeDim.push_back(*p++);
			}
			dim.nbDims = shapeDim.size() - 1;
			memcpy(dim.d, shapeDim.data() + 1, dim.nbDims * sizeof(int));
			pLayer->setReshapeDimensions(dim);
			m_tensorSet[outputDataName] = pLast = pLayer->getOutput(0);
		}
		else if (node.op_type() == "Transpose")
		{
			std::string inputName = node.input(0);
			if (!__isNodeInput(inputName))
			{
				nvinfer1::Weights w[2];
				nvinfer1::Weights& weight=w[0], &bais=w[1];
				//pyCaller.appandSampleParameter() 
				std::vector<int> perm;
				const auto& permAttr = _getAttribute(node, "perm");
				_getAtrriValues(permAttr, perm);
				instant::array& srcWt = tableWeights[inputName];
				std::vector<int>& srcDim = (std::vector<int>&)srcWt.dims();
				int numElement = std::accumulate(srcDim.cbegin(), srcDim.cend(), 1, std::multiplies<int>());
				std::vector<std::pair<void*, size_t>> buffers;
				buffers.emplace_back(srcWt.data(), numElement * sizeof(float));
				buffers.emplace_back(srcDim.data(), srcDim.size() * sizeof(int));
				buffers.emplace_back(perm.data(), perm.size() * sizeof(int));
				pyCaller->call("permute", buffers, "f");

				auto wtShape = _convertWeight<float>(srcWt, weight, wtType, true);
				
				bais.count = srcDim.size();
				bais.values = srcDim.data();
				bais.type = DataType::kINT32;
				std::string layerName = "data2trtTensor_" + std::to_string(i);
				nvinfer1::IPluginExt* pp = (nvinfer1::IPluginExt*)m_pPluginFactory->createPlugin(layerName.c_str(), w, 2);

				IPluginLayer* pLayer = vpNetwork->addPluginExt(NULL, 0, *pp);
				pLayer->setName(layerName.c_str());
				m_tensorSet[outputDataName] = pLast = pLayer->getOutput(0);
				pLast->setName(layerName.c_str());
				vpNetwork->markOutput(*pLast);
			}
		}
		else
		{//
			std::cout << "No implement : " << node.op_type() << std::endl;
		}
		if (i>=debugLayer.front()&&i<debugLayer.back())
		{
			pLast->setName((std::to_string(i) + "_" + node.name()).c_str());
			vpNetwork->markOutput(*pLast);
		}
	}

	//rf_c2_det_context_conv3_1_Y
	if (debugLayer.back()>= graph.node_size())
	for (int i = 0; i < graph.output_size(); ++i)
	{
		const ::onnx::ValueInfoProto& out = graph.output(i);
		std::string outBlobName = out.name();
		__markOutput(outBlobName, vpNetwork);
// 		std::cout << i << " onnx output array: [" << m_nchw.front() ;
// 		auto dim = out.type().tensor_type().shape();
// 		for (int i = 1; i < dim.dim_size(); ++i)std::cout << ", " << dim.dim(i).dim_value();
// 		std::cout <<"], " << outBlobName << std::endl;
// 		pLast = m_tensorSet[outBlobName];
// 		pLast->setName(outBlobName.c_str());
// 		vpNetwork->markOutput(*pLast);
	}
	else
	{
		pLast->setName(std::to_string(debugLayer.back()).c_str());
		vpNetwork->markOutput(*pLast);
	}
// 	__markOutput("rf_c2_det_context_conv3_1_Y", vpNetwork);

	return true;
}

int COnnxTrtConvertor::getModelBatchSize()
{
	return m_nchw[0];
}

void COnnxTrtConvertor::__convetMention(const std::vector<int>& src, Dims& dst)
{
	dst.nbDims = src.size();
	memcpy(dst.d, src.data(), src.size() * sizeof(int));
}

bool COnnxTrtConvertor::__isNodeInput(const std::string& vInputName)
{
// 	int iInput = convertFromString<int>(vInputName);
// 	if (iInput <= 0 || iInput>10000) return false;
	return m_tensorSet.find(vInputName) != m_tensorSet.cend();
}

void COnnxTrtConvertor::__printModelBasicInfor(const void* vpOnnxModel)
{
	const onnx::ModelProto& onnx_model = *(const onnx::ModelProto*)vpOnnxModel;
	std::cout << "ONNX version is " << onnx_model.ir_version() << std::endl;
	std::cout << "domain is " << onnx_model.domain() << std::endl;
	std::cout << "model version is " << onnx_model.model_version() << std::endl;
	std::cout << "producer name is " << onnx_model.producer_name() << std::endl;
	std::cout << "producer version is " << onnx_model.producer_version() << std::endl;
}

nvinfer1::ITensor* COnnxTrtConvertor::__squzze(INetworkDefinition*vpNet, ITensor* vSrc, int vAxis, int vN, int vOffset)
{
	std::vector<int> axes(vN);
	for (int i = 0; i < vN; ++i) axes[i] = vAxis + i + vOffset;
	return __squzze(vpNet, vSrc, axes, 0);
}

nvinfer1::ITensor* COnnxTrtConvertor::__squzze(INetworkDefinition*vpNet, ITensor* vSrc, std::vector<int>& vAxis, int vOffset)
{
	int n = 0;
	Dims src = vSrc->getDimensions();
	for (int& i : vAxis)
	{
		i += vOffset;
		if (i < src.nbDims) { src.d[i] = 0; n++; }
	}
	if (0 == n) return vSrc;
	Dims dst = src;
	std::remove_copy_if(src.d, src.d + src.nbDims, dst.d, [](const int& d) {return (0 == d); });
	IShuffleLayer* pSh = vpNet->addShuffle(*vSrc);
	dst.nbDims -= n;
	pSh->setReshapeDimensions(dst);
	return pSh->getOutput(0);
}

nvinfer1::ITensor* COnnxTrtConvertor::__unsquzze(INetworkDefinition*vpNet, ITensor* vSrc, int vAxis, int vN, int vOffset)
{
	std::vector<int> axes(vN, vAxis);
	return __unsquzze(vpNet, vSrc, axes, vOffset);
}

nvinfer1::ITensor* COnnxTrtConvertor::__unsquzze(INetworkDefinition*vpNet, ITensor* vSrc, std::vector<int>& vAxes, int vOffset)
{
	if (vAxes.empty()) return vSrc;

	for (auto& a : vAxes) a += vOffset;
	Dims dst, src = vSrc->getDimensions();
	for (int vAxis : vAxes)
	{
		if (vAxis > src.nbDims) vAxis = src.nbDims;
		dst = src;
		memcpy(dst.d + vAxis + 1, src.d + vAxis, (src.nbDims - vAxis) * sizeof(int));
		dst.d[vAxis] = 1;
		dst.nbDims++;
		src = dst;
	}
	IShuffleLayer* pSh = vpNet->addShuffle(*vSrc);
	pSh->setReshapeDimensions(dst);
	return pSh->getOutput(0);
}

