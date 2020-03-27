#include "dataStructure.h"
#include "Common.h"

CFactory<CImage> g_CImageCreator("CImage");
CFactory<CImageDetections> g_CImageDetectionsCreator("CImageDetections");
CFactory<CFaceKeyPoints> g_CFaceKeyPointsCreator("CFaceKeyPoints");
CFactory<CFace> g_CFaceCreator("CFace");
CFactory<CFaceInfor> g_CFaceInforCreator("CFaceInfor");
CFactory<CTrack> g_CTrackCreator("CTrack");

void CImage::initalize(int vVideoID, size_t vframID, int vWidth, int vHeight, void* vpData /*= 0*/, int vChanel /*= 3*/, int vElementBites /*= 1*/)
{
	videoID = vVideoID;
	width = vWidth;
	height = vHeight;
	chanel = vChanel;
	framID = vframID;
	setClodData(vpData);
	setClodDataSize(vWidth*vHeight*vChanel*vElementBites);
	birthTime = std::chrono::steady_clock::now();
}

#include "cudaCommon.h"
void CDataShared::transferData(bool vfromCPU2GPU /*= true*/, void* vpCudaStream/*=NULL*/)
{
	bool srcOnGPU = !vfromCPU2GPU, dstOnGPU = !srcOnGPU;
	cudaMemcpyKind cpyKind = (vfromCPU2GPU ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost);
	exitIfCudaError(cudaMemcpyAsync(getClodData(dstOnGPU), getClodData(srcOnGPU), getClodDataSize(), cpyKind, (cudaStream_t)vpCudaStream));
}

