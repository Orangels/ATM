#pragma once
#include "modelEngine.h"
#include "dataStructure.h"

class CSharedBuffer;
class hop_CSSDEngine : public CModelEngine
{
public:
	hop_CSSDEngine();
	~hop_CSSDEngine();

	virtual void postProcessV(std::vector<CDataShared*>& vmodelInputes, CDataShared* voutput) override;
    void getheadFacePair_(float* viopDetections, CImageDetections* vodetections);
private:
	std::vector<float> m_buffer;
};

