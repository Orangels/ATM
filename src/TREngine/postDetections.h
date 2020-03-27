#pragma once
#include <vector>
struct SFloat7
{
	float f[7];
};

struct SFilterPara
{
	float confidence;
	float area;
	float aspect;
	bool operator ==(const SFilterPara& v)
	{
		return (v.confidence == confidence && v.area == area && v.aspect == aspect);
	}
};
void postDetections(int vBatchSize, int* vpwidthHeight, float* viopDetections, SFilterPara& vthresh, int vkeep_top_k=100, cudaStream_t vStream=0);

class CImageDetections;
void getheadFacePair(float* viopDetections, CImageDetections* vodetections);
