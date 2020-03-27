#pragma once
#include <unordered_map>
#include <vector>
#include <string>
#include <cassert>
#include <iostream>
#include <cudnn.h>
#include <cstring>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <memory>
#include <algorithm>
#include "NvCaffeParser.h"
#include "NvInferPlugin.h"

void cudaSoftmax(int n, int channels,  float* x, float*y, cudaStream_t stream);


using namespace nvinfer1;
using namespace plugin;
using namespace nvcaffeparser1;


enum FunctionType
{
    SELECT=0,
    SUMMARY
};

class bboxProfile {
public:
    bboxProfile(float4& p, int idx): pos(p), bboxNum(idx) {}

    float4 pos;
    int bboxNum = -1;
    int labelID = -1;
};

class tagProfile {
public:
    tagProfile(int b, int l): bboxID(b), label(l) {}
    int bboxID;
    int label;
};


class CFactoryDirectory;
class CCaffePrototxtReader;
class PluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactory
{
public:
	PluginFactory();
	~PluginFactory();
    virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights) override;
    IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override;
	bool isPlugin(const char* layerType) override;//根据name获得layer.type, 根据type判断
    void destroyPlugin();
	CCaffePrototxtReader* m_pcaffeReader;
	bool m_forCaffe = true;

private:
// 	void(*nvPluginDeleter)(IPlugin*) { [](IPlugin* ptr) {ptr->terminate(); } };

	CFactoryDirectory* m_pfactories;
	std::unordered_map<std::string, IPlugin*> m_pluginSet;
 };
