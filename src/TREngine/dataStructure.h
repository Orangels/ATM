#pragma once
#include <chrono>
#include <vector>
#include <memory.h>

class CDataShared
{
public:
	virtual ~CDataShared(){}
	virtual void* getClodData(bool vgpu=false) { return 0; }
	virtual size_t getClodDataSize() { return 0; }
	virtual void setClodData(void* vpSrc, bool vgpu=false) {}
	virtual void copyTo(CDataShared* vpdst) = 0;
	void transferData(bool vfromCPU2GPU = true, void* vpCudaStream=NULL);
	CDataShared* m_pproducer=NULL;
};

class CCPUBigData : public CDataShared
{
public:
	CCPUBigData() {}
	virtual ~CCPUBigData() {}
	virtual void* getClodData(bool vgpu = false) override { return m_cpuData; }
	virtual void setClodData(void* vpSrc, bool vgpu = false) override { m_cpuData = vpSrc; }
	size_t getClodDataSize() { return m_clodDataBites; }
	void setClodDataSize(size_t vBiteCount) { m_clodDataBites = vBiteCount; }
	virtual void copyTo(CDataShared* vpdst)override =0;
private:
	void* m_cpuData=NULL;
	size_t m_clodDataBites = 0;
};


class CCPUGPUData: public CCPUBigData
{
public:
	void* getClodData(bool vgpu = false) override { if (vgpu)return m_gpuData; return CCPUBigData::getClodData(vgpu); }
	void setClodData(void* vpSrc, bool vgpu = false) override { if (vgpu) m_gpuData = vpSrc; else CCPUBigData::setClodData(vpSrc, vgpu); }
	virtual void copyTo(CDataShared* vpdst)override = 0;

private:
	void *m_gpuData = NULL;
};

class CImage: public CCPUGPUData
{
public:
	CImage() {}
	void initalize(int vVideoID, size_t vframID, int vWidth, int vHeight, void* vpData = 0, int vChanel = 3, int vElementBites = 1);
	inline void setImageGenerationTime(std::chrono::steady_clock::time_point vTime){birthTime = vTime;}
	std::chrono::steady_clock::time_point birthTime;
	int width, height, chanel, videoID;
	int modifiedLength = 0;
	size_t framID;
	virtual void copyTo(CDataShared* vpdst) override
	{
		CImage* dst = (CImage*)vpdst;
		*dst = *this;
	}
};

class CImageDetections : public CDataShared
{
public:
	CImageDetections() {}
	~CImageDetections() {}
	int faceCount, detectionCount;
	std::vector<float> detections;
	
	inline CImageDetections& operator=(const CImageDetections& v)
	{
		this->m_pproducer = v.m_pproducer;
		this->faceCount = v.faceCount;
		this->detectionCount = v.detectionCount;
		const auto& vd= v.detections;
		this->detections.resize(vd.size());
		if (vd.size())
			memcpy(detections.data(), vd.data(), sizeof(vd.front())*vd.size());
	} 

	virtual void copyTo(CDataShared* vpdst) override
	{
		CImageDetections* dst = (CImageDetections*)vpdst;
		*dst = *this;
	}

private:
};

class CTrack :public CDataShared
{
public:
	std::vector<int> trackingIDs;
	void copyTo(CDataShared* vpdst) override
	{
		CTrack* dst = (CTrack*)vpdst;
		dst->m_pproducer = m_pproducer;
		dst->trackingIDs.resize(trackingIDs.size());
		memcpy(dst->trackingIDs.data(), trackingIDs.data(), trackingIDs.size() * sizeof(trackingIDs.front()));
	}

protected:
private:
};

class CFace : public CDataShared
{
public:
	CFace() {}
	~CFace() {}
	int trackId;
	float confidence;
	float xyMinMax[4];

	void copyTo(CDataShared* vpdst) override
	{
		memcpy(vpdst, this, sizeof(CFace));
	}
private:
};


class CFaceKeyPoints : public CCPUGPUData
{
public:
	CFaceKeyPoints() { setClodDataSize(219 * sizeof(float)); }
	~CFaceKeyPoints() {}
	virtual void copyTo(CDataShared* vpdst) override
	{
		CFaceKeyPoints* dst = (CFaceKeyPoints*)vpdst;
		*dst = *this;
	}
private:
};

class CFaceInfor : public CDataShared
{
public:
	CFaceInfor() {}
	~CFaceInfor() {}
	float age;
	float rescore;
	bool gender;
	float blur;
	int faceId;

	virtual void copyTo(CDataShared* vpdst) override
	{
		memcpy(vpdst, this, sizeof(CFaceInfor));
	}

private:
};
