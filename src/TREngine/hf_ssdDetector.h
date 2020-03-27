#pragma once
#include "modelEngine.h"

class CSharedBuffer;
class CSSDEngine : public CModelEngine
{
public:
	CSSDEngine();
	~CSSDEngine();

	virtual void postProcessV(std::vector<CDataShared*>& vmodelInputes, CDataShared* voutput) override;

private:
	std::vector<float> m_buffer;
};

