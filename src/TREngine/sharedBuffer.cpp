#include <iostream>
#include <cuda_runtime.h>
#include "Common.h"
#include "sharedBuffer.h"
#include "cudaCommon.h"

void CSharedBuffer::writeToBuffer(std::vector<CDataShared*>& viodata, void* vcudaSteam)
{
	if (viodata.empty()) return;
	int writeOffset = m_roomSignal->wait(viodata.size());
	int num = writeOffset & 255;
	writeOffset >>= 8;
	std::vector<CDataShared*> dataLeft(viodata.cbegin()+num, viodata.cend());
	viodata.resize(num);
	bool needSynchronize = false;
	char* pbuffer = NULL;
	for (auto& src : viodata)
	{
		auto dst= __at(writeOffset);
		if (m_clodDataBites)
		{
			pbuffer = m_buffer + writeOffset*m_clodDataBites;
			if (src->getClodData())
				memcpy(pbuffer, src->getClodData(), src->getClodDataSize());
			if (m_gpuSupport && src->getClodData(true))
			{
				cudaMemcpyAsync(m_gpuBuffer + writeOffset*m_clodDataBites, src->getClodData(true), src->getClodDataSize(), cudaMemcpyDeviceToDevice, (cudaStream_t)vcudaSteam);
				needSynchronize = true;
			}
		}
		src->copyTo(dst);
		if (m_clodDataBites)
		{
			dst->setClodData(pbuffer);
			if (m_gpuSupport)dst->setClodData(m_gpuBuffer + writeOffset*m_clodDataBites, true);
		}
		writeOffset = (writeOffset + 1) % m_bufferCount;
		src = dst;
	}
	if (needSynchronize)
		exitIfCudaError(cudaStreamSynchronize((cudaStream_t)vcudaSteam));
// 	for (int i = 0; i < viodata.size(); ++i)
		m_dataSignal->signal(viodata.size());
	if (dataLeft.size()) writeToBuffer(dataLeft, vcudaSteam);
}


void CSharedBuffer::writeToBuffer(CDataShared* vpSrc, int vdataCount)
{
	char* p = (char*)vpSrc;
	std::vector<CDataShared*> src(vdataCount);
	for (int i = 0; i < vdataCount; ++i)
		src[i] = (CDataShared*)(p+m_elementBites*i);
	writeToBuffer(src);
}


void CSharedBuffer::readData(std::vector<CDataShared*>& voContainer, int vNum)
{
	voContainer.resize(0);
	int offset = m_dataSignal->wait(vNum);
	vNum = offset & 255;
	offset >>= 8;
	for (int i = 0; i < vNum; ++i)
		voContainer.push_back(__at(offset + i));
}

void CSharedBuffer::popData(std::vector<CDataShared*>& voContainer, int vNum)
{
	readData(voContainer, vNum);
	makeBufferReusable(vNum);
}


CSharedBuffer::CSharedBuffer(int vBufferCount, const std::string& velementType, int vclodDataBites /*= 0*/, bool vgpuSupport /*= false*/) : m_clodDataBites(vclodDataBites), m_bufferCount(vBufferCount), m_gpuSupport(vgpuSupport)
{
	m_pfactories = CFactoryDirectory::getOrCreateInstance(true);
	m_dataSignal = new CSemaphore(m_bufferCount, 0);
	m_roomSignal = new CSemaphore(m_bufferCount, vBufferCount);
	m_queue = (char*)m_pfactories->createProduct(velementType, vBufferCount);
	m_elementBites = m_pfactories->getSizeof(velementType);

	cudaMallocHost((void**)&m_buffer, m_bufferCount*m_clodDataBites);
// 	m_buffer = (char*)malloc(m_bufferCount*m_clodDataBites);
	if (m_gpuSupport)
		cudaMalloc((void**)&m_gpuBuffer, m_bufferCount*m_clodDataBites);
}


CDataShared* CSharedBuffer::__at(int vindex)
{
	CDataShared* dst = (CDataShared*)(m_queue + (vindex%m_bufferCount)*m_elementBites);
	return dst;
}


CSharedBuffer::~CSharedBuffer()
{
	delete m_dataSignal;
	delete m_roomSignal;
	cudaFreeHost(m_buffer);
	if (m_gpuSupport) cudaFree(m_gpuBuffer);
}

void CSharedBuffer::makeBufferReusable(int vBufferCount)
{
// 	while (vBufferCount-->0) 
		m_roomSignal->signal(vBufferCount);
}
