#include "cropAndResize.h"


#define readPixelM(Pixel, X, Y) {Pixel.x = tex2D<uchar>(SrcImage, X, Y); Pixel.y = tex2D<uchar>(SrcImage, X+1.f, Y); Pixel.z = tex2D<uchar>(SrcImage, X+2.f, Y);}
__device__ void appandColor(float3& voRBG, uchar3 vPixel, float xRelativity, float yRelativity)
{
	float Relativity = xRelativity * yRelativity;
	voRBG.x += vPixel.x * Relativity;
	voRBG.y += vPixel.y * Relativity;
	voRBG.z += vPixel.z * Relativity;
}

inline __device__ int calculateOffset(int x, int y, int width)
{
	return y * width + x;
}

__global__ void preprocessKernel(SCropResizePara vPara)
{
	//if (0 == threadIdx.x) printf("dddd%d\n", blockIdx.x);
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int NumPixelResized = vPara.numPixelResized;
// 	if (tid < NumPixelResized)
	{
		//计算原坐标 和 比例
		float X = tid % vPara.dstWidth;
		float Y = tid / vPara.dstWidth;
		X = (X + 0.5f) * vPara.xScaleS_R - 0.5f;
		Y = (Y + 0.5f) * vPara.yScaleS_R - 0.5f;
		int x = X, y = Y;
		float xDistance = fabs(X - x);
		float yDistance = fabs(Y - y);
		x += x << 1;
		X = x + 0.5f;
		Y = y + 0.5f;
		X += vPara.srcXoffset;
		Y += vPara.srcYoffset;

		//读四个数
		uchar3 TLPixel, TRPixel, BLPixel, BRPixel;
		unsigned char* src = vPara.srcImage;
		int rowLength = vPara.srcWidthBites;
		__syncthreads();
		memcpy(&TLPixel, src + calculateOffset(X,    Y, rowLength), 3);
		memcpy(&TRPixel, src + calculateOffset(X+3,  Y, rowLength), 3);
		memcpy(&BLPixel, src + calculateOffset(X,  Y+1, rowLength), 3);
		memcpy(&BRPixel, src + calculateOffset(X+3,Y+1, rowLength), 3);

		//计算
		float3 RBG{ 0 };
		//memcpy(&RBG, vPara.MeanValue, sizeof(RBG));
		appandColor(RBG, TLPixel, 1.f - xDistance, 1.f - yDistance);
		appandColor(RBG, TRPixel, xDistance, 1.f - yDistance);
		appandColor(RBG, BLPixel, 1.f - xDistance, yDistance);
		appandColor(RBG, BRPixel, xDistance, yDistance);

		float* pOut = vPara.dstArray + tid;
		__syncthreads();//写回
		*pOut = RBG.x; pOut += NumPixelResized;
		*pOut = RBG.y; pOut += NumPixelResized;
		*pOut = RBG.z;
	}
}

#include "Common.h"
void cropAndResize(SCropResizePara& vpara, cudaStream_t vstream/*=0*/)
{
	preprocessKernel << <vpara.dstHeight, vpara.dstWidth, 0, vstream >> > (vpara);
}

