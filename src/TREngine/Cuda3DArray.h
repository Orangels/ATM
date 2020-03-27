#pragma once
#include <vector>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

class CCuda3DArray
{
public:
	virtual ~CCuda3DArray();
	cudaSurfaceObject_t offerSurfaceObject();
	void copyData2Host(void *vopHost, int vZ = -1, cudaStream_t vStream = NULL);
	void copyData2GPUArray(const void *vpHost, int vZ = -1, cudaStream_t vStream = NULL);
	const cudaExtent& getArraySize(){ return m_Size; }
	CCuda3DArray(int vWidth, int vHeight, int vDepth, cudaChannelFormatDesc& vElementType, bool vNormalizedCoords = false);
	cudaTextureObject_t offerTextureObject(bool vTexCoordsNormalized/*=false*/, cudaTextureFilterMode vFilterMode/*=cudaFilterModeLinear*/, cudaTextureReadMode vReadMode/*=cudaReadModeElementType*/, cudaTextureAddressMode vOutOfRangeSolution12D = cudaAddressModeClamp, cudaTextureAddressMode vOutOfRangeSolution3D = cudaAddressModeClamp);

private:
	cudaResourceDesc	    m_DataOfTextSurfaceLA;
	cudaChannelFormatDesc*  m_pArrayElementFormat;
	cudaTextureDesc			m_TextFormatFARN;
	cudaSurfaceObject_t		m_pSurfaceObject;
	cudaTextureObject_t     m_pTextObject;
	cudaArray_t				m_pArray3D;
	cudaExtent				m_Size;
	void _setTextureParms(bool vTexCoordsNormalized, cudaTextureFilterMode vFilterMode, cudaTextureReadMode vReadMode, cudaTextureAddressMode vOutOfRangeSolution12D, cudaTextureAddressMode vOutOfRangeSolution3D);
};