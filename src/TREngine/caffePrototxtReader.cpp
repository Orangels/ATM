#include "caffePrototxtReader.h"
#include "FileFunction.h"
#include "protoIO.h"
#include "caffe.pb.h"

void CCaffePrototxtReader::__renameLayers(std::string vModelFileName)
{
	CProtoIO pio;
	caffe::NetParameter caffeModel;
	bool flagWeightFile = isEndWith(vModelFileName, "caffemodel");
	if (flagWeightFile) 
		 pio.readProtoFromBinaryFile(vModelFileName, &caffeModel);
	else pio.readProtoFromTextFile(vModelFileName, &caffeModel);
	std::unordered_map<std::string, int> typeIndex;
	for (int i = 0; i < caffeModel.layer_size(); ++i)
	{
		caffe::LayerParameter* l = caffeModel.mutable_layer(i);
		std::string layerType = l->type();
		int& index = typeIndex[layerType];
		l->set_name(layerType + "_" + l->name());
		index++;
	}
	std::string dstFileName = extractFileName(vModelFileName);
	if (flagWeightFile)
		pio.writeProto2BinaryFile(dstFileName, &caffeModel);
	else pio.writeProtoToTextFile(caffeModel, dstFileName);
}

void CCaffePrototxtReader::__trim(std::string& vValue)
{
	int i = vValue.find('\"');
	if (i >= 0 && i < vValue.size())
	{
		vValue = vValue.substr(vValue.find('\"') + 1);
		vValue = vValue.substr(0, vValue.rfind('\"'));
	}
	else
	{
		trim(vValue);
		ltrim(vValue, '\r');
		ltrim(vValue, '\t');
	}
}

void CCaffePrototxtReader::__readDim()
{
	m_dim.reserve(4);
	int iLineDim = findOffset("dim:", 1);
	if (0 > iLineDim)
		iLineDim = findOffset("input_dim", 1);
	m_dim.push_back(analyzeNum<int>(m_cfgData[iLineDim]));
	m_dim.push_back(analyzeNum<int>(m_cfgData[iLineDim+1]));
	m_dim.push_back(analyzeNum<int>(m_cfgData[iLineDim+2]));
	m_dim.push_back(analyzeNum<int>(m_cfgData[iLineDim+3]));
}


std::string CCaffePrototxtReader::__analyzeValue(std::string vKeyValue)
{
	std::string Value = vKeyValue.substr(vKeyValue.find(':') + 1);
	__trim(Value);
	return Value;
}

std::string CCaffePrototxtReader::findValue(std::string vText, int vStart, int vEnd)
{
	int iLine = findOffset(vText, vStart);
	if (iLine < 0 || iLine >= vEnd)
		return "";
	return __analyzeValue(m_cfgData[iLine]);
}

float CCaffePrototxtReader::getConfidence()
{
	if (m_confidence <= 0)
	{
		int i = findOffset("confidence_threshold", m_cfgData.size() - 1, true);
		m_confidence = findNum<float>("confidence_threshold", i, i + 1);
	}
	return m_confidence;
}

int CCaffePrototxtReader::getMaxDetectionNum()
{
	if (0 > m_maxDetectionNum)
	{
		int i = findOffset("keep_top_k", m_cfgData.size() - 1, true);
		m_maxDetectionNum = findNum<int>("keep_top_k", i, i + 1);
	}
	return m_maxDetectionNum;
}

int CCaffePrototxtReader::findOffset(std::string vText, int vStart, bool vInverted)
{
	for (int i = vStart; i < m_cfgData.size() && i>=0; )
	{
		auto line = m_cfgData[i];
		if (isStartWith(line, vText))
			return i;
		if (false == vInverted) ++i;
		else --i;
	}
// 	std::cout << "Fail to find " << vText << std::endl;
	return -1;
}


CCaffePrototxtReader::CCaffePrototxtReader(const std::string& vCaffeModelPrefix)
{
	__renameLayers(vCaffeModelPrefix + ".prototxt");
	__renameLayers(vCaffeModelPrefix + ".caffemodel");
	std::string caffePrototxtFileName = vCaffeModelPrefix+".prototxt";
	if (isExist(vCaffeModelPrefix + "_ori.prototxt"))
	{
		__renameLayers(vCaffeModelPrefix + "_ori.prototxt");
		caffePrototxtFileName = extractFileName(vCaffeModelPrefix) + "_ori.prototxt";
	}
	else caffePrototxtFileName = extractFileName(caffePrototxtFileName);

	readPathOrText(caffePrototxtFileName, m_cfgData, [](std::string& l) {return l.size() > 2; }, 0, false);

	for (int i = 0; i < m_cfgData.size(); ++i)
	{
		std::string line = m_cfgData[i];
		if (isStartWith(line, "layer"))
		{
			int index = findOffset("name:", i);
			std::string name = __analyzeValue(m_cfgData[index]);
			m_nameMapIndexs[name] = i;
		}
	}
}

std::string CCaffePrototxtReader::getOutputBlobName() 
{ 
	if ("" == m_outputBlobName)
	{
		int i = findOffset("top:", m_cfgData.size()-1, true);
		m_outputBlobName = m_cfgData[i].substr(m_cfgData[i].find(':') + 1);
		__trim(m_outputBlobName);
	}
	return m_outputBlobName;
}

const std::vector<int>& CCaffePrototxtReader::getDim() 
{ 
	if (m_dim.empty())
		__readDim();
	return m_dim; 
}

void CCaffePrototxtReader::getPermuteOrder(const std::string& vLayerName, std::vector<int>& voAxesOrder)
{
	int iLine = m_nameMapIndexs[vLayerName];
	getSequence(voAxesOrder, "order", iLine);
	voAxesOrder.resize(4, 0);
}

int CCaffePrototxtReader::getClassNum() 
{
	if (0 == m_numClass)
	{
		int iLine = findOffset("num_classes:", m_cfgData.size()-1, true);
		if (iLine < 0)
			iLine = findOffset("num_output:", m_cfgData.size()-1, true);
		m_numClass = analyzeNum<int>(m_cfgData[iLine]);
	}
	return m_numClass;
}

SPooling_param CCaffePrototxtReader::getPoolingParam(const std::string& vPoolLayerName)
{
	SPooling_param poolParam;
	int iLayer = m_nameMapIndexs[vPoolLayerName];
	int iParam = findOffset("pooling_param", iLayer+1);
	int temp = findNum<int>("kernel_size", iParam + 1, iParam+10);
	poolParam.kernel_size = temp;
	temp = findNum<int>("stride", iParam + 1, iParam + 10);
	if (temp >= 0)
		poolParam.stride = temp;
	temp = findNum<int>("pad", iParam + 1, iParam + 10);
	if (temp >= 0)
		poolParam.pad = temp;
	std::string pool = findValue("pool", iParam + 1, iParam + 10);
	if ("AVE" == pool)poolParam.pool = 1;
	else if ("STOCHASTIC" == pool)poolParam.pool = 1;
	std::string ceilmode = findValue("ceil_mode", iParam + 1, iParam + 10);
	if ("true" == ceilmode || "ceil" == ceilmode) poolParam.ceil_mode = 0.9;
	return poolParam;
}

std::string CCaffePrototxtReader::getLayerType(const std::string& vLayerName)
{
	int iLayer = m_nameMapIndexs[vLayerName];
	int iType = findOffset("type:", iLayer + 1);
	std::string layerType = __analyzeValue(m_cfgData[iType]);
	return layerType;
}

int CCaffePrototxtReader::getLayerLocation(const std::string& vLayerName)
{
	return m_nameMapIndexs[vLayerName];
}

int CCaffePrototxtReader::getFlattenAxis(const std::string& vLayerName)
{
	int iLayer = m_nameMapIndexs[vLayerName];
	int axis = findNum<int>("axis:", iLayer+2, iLayer + 10);
	if (axis < 0)
		axis = 3;
	return axis;
}
