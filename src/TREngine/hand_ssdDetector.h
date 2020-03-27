#pragma once
#include "modelEngine.h"

class CSharedBuffer;
class hand_CSSDEngine : public CModelEngine
{
public:
	hand_CSSDEngine();
	~hand_CSSDEngine();

	virtual void postProcessV(std::vector<CDataShared*>& vmodelInputes, CDataShared* voutput) override;

private:
	std::vector<float> m_buffer;
};

