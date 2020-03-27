#include <cuda_runtime.h>
#include "postDetections.h"
#include "Common.h"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>

__constant__ SFilterPara g_filterPara[1];
__global__ void filtetrBoxesAndSort(float* viopDetections, int* vpwidthHeight)
{//img_area=800000, min_thresh=0.5, min_abso_area=50 * 50, min_rela_area=0.01, max_aspect_ratio = 2.0
	float* pio = viopDetections + blockDim.x * 7 * blockIdx.x;

	float temp = g_filterPara->confidence;// tex2D<float>(vPara, 0, 0);//confidenceThresh
	__shared__ float sConsIndex[100];
	float conf = pio[2 + threadIdx.x * 7];
	sConsIndex[threadIdx.x] = conf;
	__syncthreads();
	__shared__ int sNumCondidat;
	if (conf >= temp || 0 == threadIdx.x)
	{
		if (sConsIndex[threadIdx.x + 1] < temp)
			sNumCondidat = threadIdx.x + (conf >= temp);
	}
	__syncthreads();
	int numCondidat = sNumCondidat;
	
	//area max_aspect
	const float exclude = -100;
	if (threadIdx.x < numCondidat)
	{
		float4 xyMinMax;// = *(float4*)(pio + 3 + threadIdx.x * 7);
		memcpy(&xyMinMax, pio + 3 + threadIdx.x * 7, sizeof(xyMinMax));
		int2 widthHeight = *(int2*)(vpwidthHeight+(blockIdx.x<<1));
		xyMinMax.x *= widthHeight.x; 
		xyMinMax.y *= widthHeight.y; 
		xyMinMax.z *= widthHeight.x;
		xyMinMax.w *= widthHeight.y;
		memcpy(pio + 3 + threadIdx.x * 7, &xyMinMax, sizeof(xyMinMax));
		xyMinMax.z -= xyMinMax.x;//width
		xyMinMax.w -= xyMinMax.y;//hegth
		xyMinMax.x = xyMinMax.z*xyMinMax.w;//area
		temp = g_filterPara->area;// tex2D<float>(vPara, 1, 0);//areaThresh  
		if (xyMinMax.x < temp) sConsIndex[threadIdx.x] = exclude;
		else
		{
			temp = g_filterPara->aspect;//tex2D<float>(vPara, 2, 0);//max_aspectThresh 
			xyMinMax.y = xyMinMax.z / xyMinMax.w;//max_aspect
			if (1.f > xyMinMax.y) xyMinMax.y = 1.f / xyMinMax.y;
			if (xyMinMax.y > temp) sConsIndex[threadIdx.x] = exclude;
			else sConsIndex[threadIdx.x] = pio[1 + threadIdx.x * 7];
		}
	}
	__syncthreads();
	thrust::stable_sort_by_key(thrust::device, sConsIndex, sConsIndex + numCondidat, (SFloat7*)pio, [](const float& a, const float& b) {return a > b; });
	numCondidat = thrust::lower_bound(thrust::device, sConsIndex, sConsIndex + numCondidat, 0, [](const float& a, const float& b) {return a > b; }) - sConsIndex;
	__syncthreads();
	if (numCondidat > threadIdx.x)
	*(pio+7*threadIdx.x) = numCondidat*(threadIdx.x==0);
}

__constant__ float adaptR[][2] = {
{0.355067475559597, 0.119262390148535 },
{0.332986547478306, 0.246449822265788 },
{-0.331071700057979, -0.146350441272887},
{-0.145853845175441, -0.399418765180059} };
__global__ void faceRefine(float* viopDetections, int vIndexFace, int* vpwidthHeight/*, int vHeight, int vWidth*/)
{
	float* pio = viopDetections + blockDim.x * 7 * blockIdx.x;
	int numBox = *pio;

	pio += 7 * threadIdx.x;
	if (threadIdx.x < numBox)
	{
		int classID = pio[1];
		if (vIndexFace == classID)
		{
			float4 xyMinMax;
			memcpy(&xyMinMax, pio + 3, sizeof(xyMinMax));
			float2 center;
			center.x = (xyMinMax.x + xyMinMax.z)*0.5f;
			center.y = (xyMinMax.y + xyMinMax.w)*0.5f;
			xyMinMax.x -= center.x;
			xyMinMax.y -= center.y;
			xyMinMax.z -= center.x;
			xyMinMax.w -= center.y;

			float length = 4/(xyMinMax.z + xyMinMax.w - xyMinMax.x - xyMinMax.y);
			xyMinMax.x *= length; xyMinMax.y *= length; 
			xyMinMax.z *= length; xyMinMax.w *= length;
			float4 bbox_dt;
			float2 r = *(float2*)adaptR[0]; bbox_dt.x = r.x * xyMinMax.x + r.y* xyMinMax.y - r.x*xyMinMax.z - r.y*xyMinMax.w;
			r = *(float2*)adaptR[1]; bbox_dt.y = r.x * xyMinMax.x + r.y* xyMinMax.y - r.x*xyMinMax.z - r.y*xyMinMax.w;
			r = *(float2*)adaptR[2]; bbox_dt.z = r.x * xyMinMax.x + r.y* xyMinMax.y - r.x*xyMinMax.z - r.y*xyMinMax.w;
			r = *(float2*)adaptR[3]; bbox_dt.w = r.x * xyMinMax.x + r.y* xyMinMax.y - r.x*xyMinMax.z - r.y*xyMinMax.w;
			
			length = 1 / length;
			bbox_dt.x *= length; bbox_dt.y *= length;
			bbox_dt.z *= length; bbox_dt.w *= length;

			bbox_dt.x += center.x; bbox_dt.z += center.x;
			bbox_dt.y += center.y; bbox_dt.w += center.y;

			r.x = bbox_dt.z - bbox_dt.x;
			r.y = bbox_dt.w - bbox_dt.y;
			r.x *= r.x; r.y *= r.y;
			length = r.x + r.y;
			length = sqrtf(length);
			length =  int(length*0.85f + 0.5f);//length*0.85f;//

			center.x = (bbox_dt.x + bbox_dt.z)*0.5f;
			center.y = (bbox_dt.y + bbox_dt.w)*0.5f;

			int2 widthHeight = *(int2*)(vpwidthHeight + (blockIdx.x << 1));
			widthHeight.x -= length + 1;
			widthHeight.y -= length + 1;
			bbox_dt.x = center.x - 0.5f*length;
			if (bbox_dt.x < 0) bbox_dt.x = 0;
			if (bbox_dt.x > widthHeight.x) bbox_dt.x = widthHeight.x;
			bbox_dt.y = center.y - 0.43f*length; 
			if (bbox_dt.y < 0) bbox_dt.y = 0;
			if (bbox_dt.y > widthHeight.y) bbox_dt.y = widthHeight.y;
			bbox_dt.z = bbox_dt.x + length; bbox_dt.w = bbox_dt.y + length;
			memcpy(pio + 3, &bbox_dt, sizeof(bbox_dt));
		}
	}
}
#include <algorithm>
#include <functional>

void postDetections(int vBatchSize, int* vpwidthHeight, float* viopDetections, SFilterPara& vthresh, int vkeep_top_k/*=100*/,  cudaStream_t vStream/*=0*/)
{
	static SFilterPara oldpara{0,0,0 };
	if (!(oldpara == vthresh));
	{
		oldpara = vthresh;
		cudaMemcpyToSymbolAsync(g_filterPara, &vthresh, sizeof(SFilterPara), 0, cudaMemcpyHostToDevice, vStream);
	}

	static thrust::device_vector<int> dwidthHeight(64 * 2);
	int* pwidthHeight = thrust::raw_pointer_cast(dwidthHeight.data());
	if (vpwidthHeight) cudaMemcpyAsync(pwidthHeight, vpwidthHeight, vBatchSize*2*sizeof(int), cudaMemcpyHostToDevice, vStream);

	filtetrBoxesAndSort << <vBatchSize, vkeep_top_k, 0, vStream >> > (viopDetections, pwidthHeight);
	faceRefine << <vBatchSize, vkeep_top_k, 0, vStream >> > (viopDetections, 2, pwidthHeight);
}


float getLengthOfIntersect(float a0, float a1, float b0, float b1)
{
	float right = std::min(a1, b1);
	float left = std::max(a0, b0);
	return  right - left;
}
float getArea(const float* vp)
{
	float x = vp[2] - vp[0];
	float y = vp[3] - vp[1];
	return x*y;
}
float caculateIoU(const float* vpA, const float* vpB)
{
	float iX = getLengthOfIntersect(vpA[0], vpA[2], vpB[0], vpB[2]);
	if (iX <= 0) return 0;
	float iY = getLengthOfIntersect(vpA[1], vpA[3], vpB[1], vpB[3]);
	if (iY <= 0) return 0;
	float i = iX * iY;
	float u0 = getArea(vpA);
	float u1 = getArea(vpB);
	float u = u0 + u1 - i;
	return i / u;
}
#include <list>
float caculateMaxIoUandErase(const float* vpA, std::list<float*>& vpBs, float vIouThresh, std::vector<float*>& vPaired)
{
	float maxIoU = 0;
	auto temp = vpBs.cend();
	for (auto p = vpBs.cbegin(); p != vpBs.cend(); ++p)
	{
		float iou = caculateIoU(vpA + 3, *p + 3);
		if (iou > vIouThresh && maxIoU < iou)
		{
			temp = p; maxIoU = iou;
		}
	}
	if (maxIoU > vIouThresh)
	{
		vPaired.emplace_back(*temp);
		vpBs.erase(temp);
	}
	return maxIoU;
}


#include "dataStructure.h"
void getheadFacePair(float* viopDetections, CImageDetections* vodetections)
{
	int headId = 1, faceId = 2;
	int numDetection = *viopDetections + 0.2;
	float* ptemp = viopDetections;
	for (int i = 0; i < numDetection; ++i)
	{
		ptemp += 3;
		*ptemp = int(*ptemp); ptemp++;
		*ptemp = int(*ptemp); ptemp++;
		*ptemp = int(*ptemp); ptemp++;
		*ptemp = int(*ptemp); ptemp++;
	}
	float* pFace = viopDetections;
	SFloat7 f7; f7.f[1] = headId;
	SFloat7* pf7 = (SFloat7*)viopDetections;
	float *pHead = (float *)std::lower_bound(pf7, pf7 + numDetection, f7, [](const SFloat7& vA, const SFloat7& vB) {return vA.f[1] > vB.f[1]; });
	int numFace = (pHead - viopDetections) / 7;
	int numHead = numDetection - numFace;

	std::list<float*> headSet;
	for (int k = 0; k < numHead; ++k)
		if (*(pHead + 7 * k+1)>0)
		headSet.push_back(pHead+7*k);
		else numDetection--;

	int faceCount = numFace;
	bool flagDeleteFace = false;
	const float iouThresh = 0.03;
	std::vector<float*> pairedHead;
	auto& dst = vodetections->detections;
	dst.resize(0);
	for (int i = 0; i < numFace; ++i)
	{
		float iou = caculateMaxIoUandErase(pFace, headSet, iouThresh, pairedHead);
		if (pFace[1]>0 &&(iou >= iouThresh || pFace[2] >= 0.9))
		{
			dst.insert(dst.end(), pFace + 1, pFace + 7);
			if (iou < iouThresh)
			{
				pFace[1] = headId;
				pairedHead.emplace_back(pFace);
				numDetection++;
			}
		}
		else
		{
			faceCount--;
			numDetection--;
		}
		pFace += 7;
	}

	vodetections->faceCount = faceCount;
	vodetections->detectionCount = numDetection;
	pairedHead.insert(pairedHead.end(), headSet.cbegin(), headSet.cend());
	for (auto ph : pairedHead)
		dst.insert(dst.end(), ph + 1, ph + 7);
}

