#include "softmaxPlugin.h"
#include "Common.h"
#include "cudaUtility.h"

CFactory<CSoftmaxPlugin> softmaxCreator("Softmax");

int CSoftmaxPlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
{
	int numClass = 1;
	for (int i = m_outputDim.nbDims - 1; i >= 0; --i)
		if ((numClass = m_outputDim.d[i]) > 1)
			break;
	size_t n = batchSize*m_outputDim.c()* m_outputDim.h()* m_outputDim.w();

	/*
	//std::cout << n << ", " <<  numClass << std::endl;
	//_printGPUdata((float*)inputs[0], n, stream, "softmax src");
	float onef{ 1.0f }, zerof{ 0.0f };
	if (NULL == m_cudnnHandle) cudnnSetStream(m_cudnnHandle, stream);
	cudnnDataType_t nnDataType = (m_dataType == DataType::kHALF) ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT;
	cudnnSetTensor4dDescriptor(m_x, CUDNN_TENSOR_NCHW, nnDataType, batchSize, m_outputDim.d[0], m_outputDim.d[1], m_outputDim.d[2]);
	cudnnSetTensor4dDescriptor(m_y, CUDNN_TENSOR_NCHW, nnDataType, batchSize, m_outputDim.d[0], m_outputDim.d[1], m_outputDim.d[2]);

// 

	cudnnSoftmaxForward(m_cudnnHandle,
		cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_FAST,
		cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_CHANNEL,
		&onef,
		 m_x,
		inputs[0],
		&zerof,
		 m_y,
		outputs[0]);
	*/


	cudaSoftmax(n, numClass, (float*)inputs[0], (float*)outputs[0], stream);
//  	_printGPUdata((float*)outputs[0], n, stream, "softmax dst");

	return 0;
}

nvinfer1::IPlugin* CSoftmaxPlugin::getPlugin()
{
	return this;
}
