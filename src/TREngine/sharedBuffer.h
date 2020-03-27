#pragma once
#include <vector>
#include <atomic>
#include "dataStructure.h"

class CSemaphore;
class CSharedBuffer
{
public:
	~CSharedBuffer();
	void makeBufferReusable(int vBufferCount);
	void writeToBuffer(CDataShared* vpData, int vdataCount);
	void writeToBuffer(std::vector<CDataShared*>& viodata, void* vcudaSteam=NULL);
	void readData(std::vector<CDataShared*>& voContainer, int vNum);
	void popData(std::vector<CDataShared*>& voContainer, int vNum);
	CSharedBuffer(int vBufferCount, const std::string& velementType, int vclodDataBites = 0, bool vgpuSupport = false);

private:
	CDataShared* __at(int vindex);
	void __init();
	int m_bufferCount, m_elementBites, m_clodDataBites;
	char* m_queue;
	CSemaphore *m_dataSignal, *m_roomSignal;
	CFactoryDirectory* m_pfactories;
	char *m_buffer, *m_gpuBuffer;
	bool m_gpuSupport;
};