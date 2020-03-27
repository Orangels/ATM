#pragma once
#include "modelEngine.h"
class CCuda3DArray;
class CKeyPointsGenerator : public CModelEngine
{
public:
	CKeyPointsGenerator();
	~CKeyPointsGenerator();
	virtual void preProcessV(std::vector<CDataShared *>& vsrc) override;
	virtual void postProcessV(std::vector<CDataShared *>& vmodelInputes, CDataShared* voutput) override;
private:
	CCuda3DArray* m_pshpBase = NULL, *m_pexpBase = NULL;
	float *m_gpuFaceBoxes = NULL, *m_cpuFaceBoxes=NULL;
	float* m_pointsBuffer;
};
