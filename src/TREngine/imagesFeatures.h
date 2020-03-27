#pragma once
#include <vector>
#include <string>
#include "Singleton.h"
class CProtoIO;
class CConfiger;
class CGPUFaceQuery;
class CFaceFeatures;
class CImagesFeatures : public CSingleton<CImagesFeatures>
{
public:
	~CImagesFeatures();
	void addFeature(int vfaceId, float* vfeature, void* vcudaStream);
	int getPersonCount() { return m_personCount; }
	inline int getImageCount() {return m_imgNames.size(); }
	int whoAmI(float* vpfeature, void* vcudaStream);
	bool isImagesPath();
	const std::string getImageFileName(int i);
	const std::string getPersonName(int vfaceId);
	bool isOnlineRecognization() { return m_onlineFlag; }
	
private:
	float m_recognizationConfidence;
	void _normalize(float* vp);
	bool m_onlineFlag, m_pinyin;
	int m_personCount;
	std::string m_videoOrImgPath;
	CImagesFeatures();
	friend class CSingleton<CImagesFeatures>;
	float* m_buffer;
	CFaceFeatures* m_pfeatrues;
	CProtoIO* m_pio;
	CConfiger* m_pconfiger;
	CGPUFaceQuery* m_pfaceQuery=NULL;
	std::vector<std::string> m_imgNames;
};


