#include "pluginImplement.h"
// #include "mathFunctions.h"
#include <vector>
#include <algorithm>
#include "Common.h"
#include "caffePrototxtReader.h"
#include "caffePlugin.h"

PluginFactory::PluginFactory():m_pcaffeReader(NULL)
{
	m_pfactories = CFactoryDirectory::getOrCreateInstance();
}


PluginFactory::~PluginFactory()
{
}

/******************************/
// PluginFactory
/******************************/


nvinfer1::IPlugin* PluginFactory::createPlugin(const char* vlayerName, const nvinfer1::Weights* weights, int nbWeights)
{
	std::string layerName(vlayerName);
	std::string layerType = deleteTailNum(layerName);
	if (m_pluginSet.find(layerName) == m_pluginSet.end())
	{
		CCaffePlugin* pPlugin = (CCaffePlugin*)m_pfactories->createProduct(layerType);
		pPlugin->createByModelAndWeight(layerName.c_str(), m_pcaffeReader, weights, nbWeights);
		m_pluginSet[layerName] = pPlugin->getPlugin();
	}
	return m_pluginSet[layerName];
}

IPlugin* PluginFactory::createPlugin(const char* vlayerName, const void* vpSerialData, size_t serialLength)
{
	const char* serialData = (const char*)vpSerialData;
	std::string layerName(vlayerName);
	std::string layerType = deleteTailNum(layerName);
	if (m_pluginSet.find(layerName) == m_pluginSet.end())
	{
		CCaffePlugin* pPlugin = (CCaffePlugin*)m_pfactories->createProduct(layerType);
		pPlugin->createPluginBySerializedData(serialData, serialLength);
		m_pluginSet[layerName] = pPlugin->getPlugin();
		pPlugin->m_layerName = vlayerName;
	}

	return m_pluginSet[layerName];
}

bool PluginFactory::isPlugin(const char* vlayerName)
{
	std::string layerType = deleteTailNum(vlayerName);
	bool inFactory =  m_pfactories->existFactory(layerType);
	return inFactory;
}

void PluginFactory::destroyPlugin()
{
	for (auto& plugin : m_pluginSet)
	{
		//plugin.second->terminate(); wrong call manner.
	}
 }
