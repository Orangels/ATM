#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include "faceQuery.h"
#include "faceFeatures.pb.h"
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

CGPUFaceQuery::CGPUFaceQuery(void* vfaces) : m_handle(0)
{
	if (vfaces)
	{
		CFaceFeatures* pfaces = (CFaceFeatures*)vfaces;
		int personCount = pfaces->persons_size();
		int featureCount = pfaces->persons(0).featrue_size();
		cudaMalloc((void**)&m_faceMatrix, sizeof(float)*personCount*featureCount);
		std::vector<float> buffer(personCount*featureCount);
		float* pbuffer = buffer.data();
		for (int i = 0; i < personCount; ++i)
		{
			auto& person = pfaces->persons(i);
			for (int k = 0; k < featureCount; ++k, ++pbuffer)
				*pbuffer = person.featrue(k);
		}
		cudaMemcpyAsync(m_faceMatrix, buffer.data(), buffer.size() * sizeof(buffer.front()), cudaMemcpyHostToDevice, 0);
		cudaStreamSynchronize(0);

		cudaMalloc((void**)&m_similarities, personCount * sizeof(float));
		m_rows = personCount;
		m_cols = featureCount;
	}
}

CGPUFaceQuery::~CGPUFaceQuery()
{
	cudaFree(m_similarities);
	cudaFree(m_faceMatrix);
	if (m_handle) cublasDestroy((cublasHandle_t)m_handle);
}
#include<thrust/system/cuda/execution_policy.h> 
extern __shared__ float s_buffer[];
__global__  void L2Normalize(float* viopData)
{
	int tid = threadIdx.x;
	float* pbuffer = s_buffer;
	int step = blockDim.x;
	float tmp0 = viopData[tid];
	float tmp1 =  viopData[tid + step];
	pbuffer[tid] = tmp0;
	pbuffer[tid + step] = tmp1;
	tmp0 *= tmp0; tmp1 *= tmp1;

	pbuffer += step << 1;
	pbuffer[tid] = tmp0 + tmp1;

	step >>= 1; __syncthreads(); //support featureLength is 1024. if 2048, we should add similar codes. 
	if (step == 256 && tid < step) pbuffer[tid] += pbuffer[tid + step];

	__syncthreads();
	if (tid < 128)  pbuffer[tid] += pbuffer[tid + 128];
	__syncthreads();
	if (tid < 64)  pbuffer[tid] += pbuffer[tid + 64];
	__syncthreads();
	if (tid < 32) pbuffer[tid] += pbuffer[tid + 32];
	if (tid < 16) pbuffer[tid] += pbuffer[tid + 16];
	if (tid < 8) pbuffer[tid] += pbuffer[tid + 8];
	if (tid < 4) pbuffer[tid] += pbuffer[tid + 4];
	if (tid < 2) pbuffer[tid] += pbuffer[tid + 2];
	if (tid < 1)
	{
		float L2Lenght = sqrtf(pbuffer[0] + pbuffer[1]);
		pbuffer[tid] = 1.f / L2Lenght;
	}

	step = blockDim.x;
	__syncthreads();
	float _l2Length = pbuffer[0];
	pbuffer -= step << 1;
	viopData[tid] = pbuffer[tid] * _l2Length;
	viopData[tid + step] = pbuffer[tid + step] * _l2Length;
}

#include "FileFunction.h"
int CGPUFaceQuery::query(float* vpGPUFeature, void* vcudaStream, float vconfidence, int vfeatureLength /*= 512*/)
{
// 	cudaStream_t s = (cudaStream_t)vcudaStream;
// 	static float* m_buffer = NULL;
// 	if (m_buffer == NULL)cudaMallocHost((void**)&m_buffer, 512 * 4);
// 	cudaMemcpyAsync(m_buffer, vpGPUFeature, 512 * 4, cudaMemcpyDeviceToHost, s);
// 	cudaStreamSynchronize(s);
// 	writeFromMemory2File("bug.dat", m_buffer, 512 * 4
// 	calculateL2Normal(vpGPUFeature, vcudaStream, vfeatureLength);
	cudaStream_t cudaStream = (cudaStream_t)vcudaStream;
	cublasHandle_t handle = (cublasHandle_t)m_handle;
	static thrust::cuda_cub::execute_on_stream policy = thrust::cuda::par.on(cudaStream);
	if (handle == 0)
	{
		cublasCreate(&handle);
		cublasSetStream(handle, cudaStream);
		m_handle = handle;
	}
	const float alpha = 1, beta = 0; 
	cublasSgemv_v2(handle, CUBLAS_OP_T, 
		 m_cols, m_rows, &alpha, m_faceMatrix, m_cols,
		vpGPUFeature, 1, &beta, m_similarities, 1);
	float* pdst = thrust::max_element(policy, m_similarities, m_similarities+m_rows);

	float similarity;
	cudaMemcpyAsync(&similarity, pdst, sizeof(similarity), cudaMemcpyDeviceToHost, cudaStream);
	cudaStreamSynchronize(cudaStream);
	if (similarity >= vconfidence)
		return pdst - m_similarities;
	return -1;
}

int CGPUFaceQuery::calculateL2Normal(float* viopGPUFeature, void* vcudaStream, int vfeatureLength /*= 512*/)
{
	const int threadCount = vfeatureLength / 2;
	cudaStream_t cudaStream = (cudaStream_t)vcudaStream;
	L2Normalize << <1, threadCount, sizeof(float)*threadCount*3, cudaStream >> > (viopGPUFeature);
}
