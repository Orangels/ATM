#pragma once
#include <cuda_runtime.h>
struct SCropResizePara
{
	unsigned char* srcImage;
	float xScaleS_R, yScaleS_R; //the size ratio between src image and resized image.
	int srcXoffset=0, srcYoffset=0;// srcXoffset may be needed * channels.
	int srcWidthBites;
	float *dstArray;
	int dstWidth, dstHeight, numPixelResized, dstWriteWidth;
};

void cropAndResize(SCropResizePara& vpara, cudaStream_t vstream=0);

