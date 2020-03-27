#pragma once
class CGPUFaceQuery
{
public:
	CGPUFaceQuery(void* vfaces);
	~CGPUFaceQuery();
	int query(float* vpGPUFeature, void* vcudaStream, float vconfidence, int vfeatureLength = 512);
	int calculateL2Normal(float* viopGPUFeature, void* vcudaStream, int vfeatureLength = 512);

private:
	float* m_faceMatrix, *m_similarities;
	void* m_handle=NULL;
	int m_rows, m_cols;
};

