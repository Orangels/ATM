#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include "Singleton.h"
#include "StringFunction.h"

struct SPooling_param
{
	int  kernel_size = 2;
	int  stride = 2;
	int  pool = 0;
// 	enum PoolingParameter_PoolMethod {
// 		PoolingParameter_PoolMethod_MAX = 0,
// 		PoolingParameter_PoolMethod_AVE = 1,
// 		PoolingParameter_PoolMethod_STOCHASTIC = 2
// 	};
	int pad = 0;
	float ceil_mode = 0;
};

class CCaffePrototxtReader 
{
public:
	void getPermuteOrder(const std::string& vLayerName, std::vector<int>& voAxesOrder);

	template<typename T>void getSequence(std::vector<T>& voArrary, const std::string& vKey, int vStart);
	CCaffePrototxtReader(const std::string& vCaffeModelPrefix);
	const std::vector<int>& getDim();
	std::string getOutputBlobName();
	int getClassNum();
	SPooling_param getPoolingParam(const std::string& vPoolLayerName);
	std::string getLayerType(const std::string& vLayerName);
	int getLayerLocation(const std::string& vLayerName);
	int getFlattenAxis(const std::string& vLayerName);	
	int findOffset(std::string vText, int vStart, bool vInverted=false);
	template<typename T> T findNum(std::string vText, int vStart, int vEnd)
	{
		int iLine = findOffset(vText, vStart);
		if (iLine < 0 || iLine >= vEnd)
			return -11;
		return analyzeNum<T>(m_cfgData[iLine]);
	}	
	
	template<typename T>T analyzeNum(std::string vKeyValue)
	{
		std::string Value = vKeyValue.substr(vKeyValue.find(':') + 1);
		return convertFromString<T>(Value);
	}
	std::string findValue(std::string vText, int vStart, int vEnd);
	float getConfidence();
	int getMaxDetectionNum();
	void release() {
		if (m_cfgData.size())
		{
			m_cfgData.clear(); 
			m_cfgData = std::vector<std::string>();
			m_nameMapIndexs = std::unordered_map<std::string, int>();
		}
	}
private:
	void __renameLayers(std::string vModelFileName);
	void __trim(std::string& vValue);
	void __readDim();
	std::string __analyzeValue(std::string vKeyValue);
	int m_numClass = 0;
	int m_maxDetectionNum = -1;
	float m_confidence = -1;
	std::vector<int> m_dim;
	std::string m_outputBlobName = "";
	std::vector<std::string> m_cfgData;
	std::unordered_map<std::string, int> m_nameMapIndexs;
};

template<typename T>
void CCaffePrototxtReader::getSequence(std::vector<T>& voArrary, const std::string& vKey, int vStart)
{
	int offset =  findOffset(vKey, vStart);
	voArrary.push_back(analyzeNum<T>(m_cfgData[offset]));
	while ((offset+1) == findOffset(vKey, offset+1))
	{
		offset++;
		voArrary.push_back(analyzeNum<T>(m_cfgData[offset]));
	}
}

