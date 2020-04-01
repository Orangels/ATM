#pragma once
#include <vector>
#include <unordered_map>
#include "modelEngine.h"
#include "protoIO.h"
#include "affineInterface.h"

class CImage;
class CFaceFeatures;
class CImagesFeatures;
class CFaceRecognization : public CModelEngine
{
public:
	CFaceRecognization();
	~CFaceRecognization();
	virtual void preProcessV(std::vector<CDataShared *>& vsrc) override;
	virtual void postProcessV(std::vector<CDataShared *>& vmodelInputes, CDataShared* voutput) override;
	virtual void get_feature(std::vector<float>& fea);

private:
	std::vector<float> m_yawPitchRol;
	size_t m_flagRecognization;
	float ignoreThresh = 0.3;
	float confidenceStep = 0.06;
	int MAX_MISMATCH_TIMES = 10;
	bool __ignore(CDataShared * vpFaceInfor);
	bool m_system, m_developer;
	CImagesFeatures* m_namesFeatures;
	std::vector<std::unordered_map<int, std::tuple<size_t, float, int>> > m_videoFrameFaceConfidece;
	float* m_pointsBuffer;
	Affine_ affine_instance;
	void affine(int vbatchSize, float* vp68points, CImage* vimage, float* vdst);
	std::vector<float> batch_face_feature;
};