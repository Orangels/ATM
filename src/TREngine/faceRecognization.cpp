#include "faceRecognization.h"
#include "Common.h"
#include "dataStructure.h"
#include "FileFunction.h"
#include "cudaCommon.h"
#include "imagesFeatures.h"
#include "affineInterface.h"
#include "faceQuery.h"
CFactory<CFaceRecognization> g_faceRecognization("CFaceRecognization");

template<typename T>
static void _printGPUdataElement(const T* vpDevice, int n, cudaStream_t stream, const std::string& vInfor)
{
    std::vector<T> r(n);
    cudaMemcpyAsync(r.data(), vpDevice, sizeof(T)*n, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cv::Mat dest(96, 96, CV_32FC1, r.data());
    cv::imwrite(vInfor,dest);
}

CFaceRecognization::CFaceRecognization()
{
	std::string faceRecognizationModelPrefix = m_pconfiger->readValue("faceRecognizationModelPrefix");
	_setBatchSize(m_pconfiger->readValue<int>("faceRecognizationBatchSize"));
	_loadNetwork(m_modelPath + faceRecognizationModelPrefix, getCudaStream());
	MAX_MISMATCH_TIMES = m_pconfiger->readValue<int>("MAX_MISMATCH_TIMES");
	confidenceStep = m_pconfiger->readValue<float>("confidenceStep");
	int maxVideoCount = m_pconfiger->readValue<int>("maxVideoCount");
	ignoreThresh = m_pconfiger->readValue<float>("ignoreThresh");
//	m_namesFeatures = CImagesFeatures::getOrCreateInstance();
	m_videoFrameFaceConfidece.resize(maxVideoCount);
//	_cudaFree(m_inputViceBuffer, true);
	m_system = ("system" == m_pconfiger->readValue("on_off"));
	m_developer = ("developer" == m_pconfiger->readValue("on_off"));
	cudaMalloc((void**)&m_pointsBuffer, 219 * getBatchSize() * sizeof(float));
	if (m_system)
		splitStringAndConvert(m_pconfiger->readValue("yawPitchRol"), m_yawPitchRol, ',');

	m_pfaceQuery = new CGPUFaceQuery(NULL);
	affine_instance = *(new Affine_(m_indims.w(), m_indims.h()));
}

CFaceRecognization::~CFaceRecognization()
{
}
volatile int g_recognizationFlag = 0;
#include "caffePlugin.h"
void CFaceRecognization::preProcessV(std::vector<CDataShared *>& vsrc)
{
	static int itest = 0;
	std::vector<CDataShared *> batchFaces(vsrc.rbegin(), vsrc.rend());
	std::vector<CFaceKeyPoints *> facesOnSameImage;
	int offset = 0;
	while (batchFaces.size())
	{
		CFace* pface = (CFace*)batchFaces.back()->m_pproducer;
		facesOnSameImage.resize(0);

		facesOnSameImage.emplace_back((CFaceKeyPoints *)batchFaces.back());
		batchFaces.pop_back();
		CImage* pimage = (CImage*)pface->m_pproducer;
		while (batchFaces.size())
		{
			pface = (CFace*)batchFaces.back()->m_pproducer;
			if (pface->m_pproducer == pimage)
			{
				facesOnSameImage.emplace_back((CFaceKeyPoints *)batchFaces.back());
				batchFaces.pop_back();
			}
			else break;
		}
		float* p68keypointsCPU = (float*)facesOnSameImage.front()->getClodData(true);
		bool isContinuousMemery = (facesOnSameImage.size()==1||facesOnSameImage.back()->getClodData(true) > facesOnSameImage.front()->getClodData(true));
		if (!isContinuousMemery)
		{
			p68keypointsCPU = m_pointsBuffer;
			for (int k = 0; k < facesOnSameImage.size(); ++k)
				cudaMemcpyAsync(p68keypointsCPU+219*k, facesOnSameImage[k]->getClodData(true), 219*sizeof(float), cudaMemcpyDeviceToDevice, getViceCudaStream());
		}

// 		for (int k = 0; k < facesOnSameImage.size(); ++k)
// 			outPutSum(p68keypointsCPU + k * 219, 219, "68points");
//		std::cout << "face attribute pre "<< *((float*)facesOnSameImage.front()->getClodData()) << std::endl;
//		std::cout << "face attribute pre cpu dst "<< ((float*)facesOnSameImage.front()->getClodData()) << std::endl;
//		std::cout << "face attribute pre gpu dst "<< ((float*)facesOnSameImage.front()->getClodData(true)) << std::endl;
		CTimer timer;
		timer.start(1);
		affine(facesOnSameImage.size(), p68keypointsCPU, pimage, (float *)m_inputViceBuffer + offset*m_indims.c()*m_indims.h()*m_indims.w());
		offset += facesOnSameImage.size();
		timer.stop();
//		std::cout << "my face reco Affine times total:" << timer.getAvgMillisecond() << std::endl;
//		std::cout << "face attribute pre over cpu dst "<< ((float*)facesOnSameImage.front()->getClodData()) << std::endl;
//		std::cout << "face attribute pre over "<< *((float*)facesOnSameImage.front()->getClodData()) << std::endl;
	}
}

#include "caffePlugin.h"
void CFaceRecognization::postProcessV(std::vector<CDataShared *>& vmodelInputes, CDataShared* voutput)
{
// 	_printGPUdataElement((float*)modelIObuffers.front(), 96 * 96 * 3, getCudaStream(),  "recoginput.png");
//    visgpudata((float*)modelIObuffers.front(), 96 * 96 * 3, getCudaStream(),  "recog input");
// 	_printGPUdataElement((float*)modelIObuffers.back(), 1 * 1 * 128, getCudaStream(),  "recog output");

	if (1)
	{
		CFaceInfor* pi = (CFaceInfor*)voutput;

		// xxs add
		float* feat = (float*)modelIObuffers.back();

//		m_pfaceQuery->calculateL2Normal(feat, getCudaStream(),512);

		int out_size = int(m_dimsOut.h() *m_dimsOut.w()*m_dimsOut.c());
		std::vector<float> r(out_size);
        cudaMemcpyAsync(r.data(), feat, sizeof(float) * out_size, cudaMemcpyDeviceToHost,
                            getCudaStream());
        cudaStreamSynchronize(getCudaStream());
        _normalize(r.data(),out_size);
//        for (int i = 0; i < 128; ++i)
//            std::cout << i<<":" << r[i] << " ";


        std::vector<float>().swap(batch_face_feature);
        batch_face_feature=r;
	}

//	m_inputViceBuffer = NULL;//avoid  memory be freed twice.
}


void CFaceRecognization::affine(int vbatchSize, float* vp68points, CImage* vimage, float* vdst)
{
	clock_t start, end;
	bool show_time = false;
	start = clock();
	affine_instance.stream = getViceCudaStream();
	affine_instance.show_eclipse_time = show_time;

	affine_instance.affineInterface(vbatchSize, vp68points, (unsigned char*)vimage->getClodData(true),
		vimage->width, vimage->height, vdst);
	end = clock();
	if (show_time)
		printf("Total execution time %3.4f ms\n", (double)(end - start) / CLOCKS_PER_SEC * 1000 / vbatchSize);
}
void CFaceRecognization::_normalize(float* vp, int vfeatureLength)
{
	double sum = 0;
	float* p = vp;
	for (int i = 0; i < vfeatureLength; ++i,++p)
		sum += (*p) * (*p);
	p = vp;
	sum = sqrt(sum);
	for (int i = 0; i < vfeatureLength; ++i, ++p)
	{
		*p /= sum;
// 		if (i > 510) std::cout << sum << ", " << *p << std::endl;
	}
}
void CFaceRecognization::get_feature(std::vector<float>& fea)
{
    fea = batch_face_feature;
}