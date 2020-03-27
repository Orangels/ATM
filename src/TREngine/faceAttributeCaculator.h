#pragma once	
#include "modelEngine.h"
#include "affineInterface.h"
class CImage;
class CFaceAttributeCalculator : public CModelEngine
{
public:
	CFaceAttributeCalculator();
	~CFaceAttributeCalculator();
	virtual void preProcessV(std::vector<CDataShared *>& vsrc) override;
	virtual void postProcessV(std::vector<CDataShared *>& vmodelInputes, CDataShared* voutput) override;
	virtual void get_feature(std::vector<std::vector<float>>& fea);

private:
	float __softmax(float x0, float x1);
	void affine(int vbatchSize, float* vp68points, CImage* vimage, float* vdst);
	float* m_pointsBuffer;
	Affine_ affine_instance;
	CModelEngine* m_pfaceRecognization = NULL;
	std::vector<CDataShared *> m_faceAttriOI;
	std::vector<std::vector<float>> batch_face_feature;
};
